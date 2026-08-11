# This module implements baseline few-shot classifiers for comparison with the quantum prototype network.
# It builds prototypes from support examples and evaluates classical and quantum-style predictors against episode data.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances

from qpn.episodes import Episode
from qpn.quantum.encoding import QuantumFeaturePreprocessor, get_feature_map

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import Sampler

class UntrainedProtoNet:
    """
    Classical Nearest-Centroid ProtoNet (Euclidean distance).
    No meta-training. Prototypes are formed directly from the support set.
    """
    def predict(self, episode: Episode) -> np.ndarray:
        # Find unique classes in support
        classes = np.unique(episode.support_y)
        prototypes = []
        
        for c in classes:
            mask = (episode.support_y == c)
            proto = np.mean(episode.support_x[mask], axis=0)
            prototypes.append(proto)
            
        prototypes = np.array(prototypes) # (N, D)
        
        # Compute distances
        dists = euclidean_distances(episode.query_x, prototypes) # (N*Q, N)
        
        # Predict argmin distance
        preds = np.argmin(dists, axis=1)
        
        # Map back to original class indices
        return classes[preds]
        
    def __call__(self, episode: Episode) -> np.ndarray:
        return self.predict(episode)


class ScikitLearnBaseline:
    """
    Wrapper for scikit-learn classifiers (SVM, kNN, LogReg).
    Fits on the support set of an episode and predicts on the query set.
    """
    def __init__(self, model_cls, **kwargs):
        self.model_cls = model_cls
        self.kwargs = kwargs
        
    def predict(self, episode: Episode) -> np.ndarray:
        model = self.model_cls(**self.kwargs)
        model.fit(episode.support_x, episode.support_y)
        return model.predict(episode.query_x)
        
    def __call__(self, episode: Episode) -> np.ndarray:
        return self.predict(episode)


def get_classical_ml_baselines():
    return {
        "LinearSVM": ScikitLearnBaseline(SVC, kernel="linear"),
        "RBF_SVM": ScikitLearnBaseline(SVC, kernel="rbf"),
        "kNN": ScikitLearnBaseline(KNeighborsClassifier, n_neighbors=1),
        "LogReg": ScikitLearnBaseline(LogisticRegression, max_iter=1000)
    }

# ==============================================================================
# Quantum Kernel Baselines (M3)
# ==============================================================================

class QuantumKernelBaseline:
    """
    Quantum Support Vector Classifier (QSVC) using FidelityQuantumKernel.
    Evaluates exact statevector fidelity (no shots).
    """
    def __init__(self, n_qubits: int, fm_kind: str = "zz", qchba_indices: np.ndarray = None):
        self.n_qubits = min(n_qubits, 12)
        self.fm_kind = fm_kind
        self.qchba_indices = qchba_indices
        
        self.feature_map = get_feature_map(self.fm_kind, self.n_qubits)
        # Exact statevector simulator
        self.sampler = Sampler()
        self.fidelity = ComputeUncompute(sampler=self.sampler)
        self.qkernel = FidelityQuantumKernel(feature_map=self.feature_map, fidelity=self.fidelity)
        
    def predict(self, episode: Episode) -> np.ndarray:
        qsvc = QSVC(quantum_kernel=self.qkernel)
        qsvc.fit(episode.support_x, episode.support_y)
        return qsvc.predict(episode.query_x)
        
    def __call__(self, episode: Episode) -> np.ndarray:
        return self.predict(episode)

class QuantumKNNBaseline:
    """
    kNN utilizing the Quantum Kernel matrix.
    """
    def __init__(self, n_qubits: int, fm_kind: str = "zz", qchba_indices: np.ndarray = None, n_neighbors: int = 1):
        self.n_qubits = min(n_qubits, 12)
        self.fm_kind = fm_kind
        self.qchba_indices = qchba_indices
        self.n_neighbors = n_neighbors
        
        self.feature_map = get_feature_map(self.fm_kind, self.n_qubits)
        self.sampler = Sampler()
        self.fidelity = ComputeUncompute(sampler=self.sampler)
        self.qkernel = FidelityQuantumKernel(feature_map=self.feature_map, fidelity=self.fidelity)
        
    def predict(self, episode: Episode) -> np.ndarray:
        # Precompute kernel matrices
        K_train = self.qkernel.evaluate(x_vec=episode.support_x)
        K_test = self.qkernel.evaluate(x_vec=episode.query_x, y_vec=episode.support_x)
        
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric="precomputed")
        knn.fit(K_train, episode.support_y)
        return knn.predict(K_test)
        
    def __call__(self, episode: Episode) -> np.ndarray:
        return self.predict(episode)

# ==============================================================================
# Trained Classical ProtoNet
# ==============================================================================

class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class TrainedProtoNet:
    """
    Classical Trained ProtoNet.
    Trained episodically on the train relations using Snell et al. loss.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        self.encoder = MLPEncoder(input_dim, hidden_dim, output_dim)
        
    def train_epoch(self, sampler, n_episodes=100, lr=1e-3):
        self.encoder.train()
        optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        
        episodes = sampler.sample(n_episodes)
        total_loss = 0.0
        
        for ep in episodes:
            optimizer.zero_grad()
            
            s_x = torch.tensor(ep.support_x, dtype=torch.float32)
            q_x = torch.tensor(ep.query_x, dtype=torch.float32)
            
            s_z = self.encoder(s_x)
            q_z = self.encoder(q_x)
            
            classes = np.unique(ep.support_y)
            prototypes = []
            
            for c in classes:
                mask = (ep.support_y == c)
                proto = s_z[mask].mean(dim=0)
                prototypes.append(proto)
                
            prototypes = torch.stack(prototypes)
            
            # Distances (squared euclidean)
            # q_z: (N*Q, D), prototypes: (N, D)
            dists = torch.cdist(q_z, prototypes, p=2).pow(2)
            
            # Logits are negative distances
            logits = -dists
            
            # Targets are just the mapped class indices
            # Since classes are [0, 1, ..., N-1], we map query_y to [0, N-1] indices
            # We assume classes are 0..N-1 and sorted
            q_y = torch.tensor(ep.query_y, dtype=torch.long)
            
            loss = nn.CrossEntropyLoss()(logits, q_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / n_episodes
        
    def predict(self, episode: Episode) -> np.ndarray:
        self.encoder.eval()
        with torch.no_grad():
            s_x = torch.tensor(episode.support_x, dtype=torch.float32)
            q_x = torch.tensor(episode.query_x, dtype=torch.float32)
            
            s_z = self.encoder(s_x)
            q_z = self.encoder(q_x)
            
            classes = np.unique(episode.support_y)
            prototypes = []
            
            for c in classes:
                mask = (episode.support_y == c)
                proto = s_z[mask].mean(dim=0)
                prototypes.append(proto)
                
            prototypes = torch.stack(prototypes)
            dists = torch.cdist(q_z, prototypes, p=2)
            preds = torch.argmin(dists, dim=1).numpy()
            
            return classes[preds]
            
    def __call__(self, episode: Episode) -> np.ndarray:
        return self.predict(episode)
