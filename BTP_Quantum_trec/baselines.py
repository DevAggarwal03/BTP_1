"""
baselines.py
Classical and Quantum baseline models for comparison with QPN.

Provides:
- UntrainedProtoNet (Classical nearest-centroid with Euclidean distance)
- TrainedProtoNet (Classical MLP encoder + episodic training)
- ScikitLearnBaseline (Wraps any sklearn model: SVM, kNN, LogReg)
- QuantumKernelBaseline (QSVC with FidelityQuantumKernel)

All baselines expect the same Episode objects from EpisodeSampler.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances

from data.episode_sampler import EpisodeSampler, Episode
from quantum.encoding.angle_encoding import AngleEncoder

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import Sampler


# ==============================================================================
# Classical Untrained ProtoNet
# ==============================================================================

class UntrainedProtoNet:
    """
    Classical Nearest-Centroid ProtoNet (Euclidean distance).
    No meta-training. Prototypes are formed directly from the support set.
    """

    def predict(self, episode: Episode) -> np.ndarray:
        classes = np.unique(episode.support_y)
        prototypes = []

        for c in classes:
            mask = episode.support_y == c
            proto = np.mean(episode.support_x[mask], axis=0)
            prototypes.append(proto)

        prototypes = np.array(prototypes)  # (N, D)
        dists = euclidean_distances(episode.query_x, prototypes)  # (N*Q, N)
        preds = np.argmin(dists, axis=1)

        return classes[preds]

    def __call__(self, episode: Episode) -> np.ndarray:
        return self.predict(episode)


# ==============================================================================
# Scikit-Learn Baselines
# ==============================================================================

class ScikitLearnBaseline:
    """
    Wrapper for scikit-learn classifiers (SVM, kNN, LogReg).
    Fits on the support set of an episode and predicts on the query set.
    """

    def __init__(self, model_cls, **kwargs) -> None:
        self.model_cls = model_cls
        self.kwargs = kwargs

    def predict(self, episode: Episode) -> np.ndarray:
        model = self.model_cls(**self.kwargs)
        model.fit(episode.support_x, episode.support_y)
        return model.predict(episode.query_x)

    def __call__(self, episode: Episode) -> np.ndarray:
        return self.predict(episode)


def get_classical_ml_baselines() -> dict[str, ScikitLearnBaseline]:
    return {
        "LinearSVM": ScikitLearnBaseline(SVC, kernel="linear"),
        "RBF_SVM": ScikitLearnBaseline(SVC, kernel="rbf"),
        "kNN": ScikitLearnBaseline(KNeighborsClassifier, n_neighbors=1),
        "LogReg": ScikitLearnBaseline(LogisticRegression, max_iter=1000),
    }


# ==============================================================================
# Quantum Kernel Baseline (QSVC)
# ==============================================================================

class QuantumKernelBaseline:
    """
    Quantum Support Vector Classifier (QSVC) using FidelityQuantumKernel.
    Evaluates exact statevector fidelity (no shots).
    """

    def __init__(self, n_qubits: int = 8) -> None:
        self.n_qubits = min(n_qubits, 10)  # limit size

        # Angle Encoding feature map
        self.feature_map = AngleEncoder(self.n_qubits).encode(
            np.zeros(self.n_qubits)
        )
        # However, FidelityQuantumKernel needs a parameterized circuit.
        # So we build a parameterized AngleEncoder here:
        from qiskit.circuit import ParameterVector
        from qiskit import QuantumCircuit
        
        self.params = ParameterVector("x", self.n_qubits)
        qc = QuantumCircuit(self.n_qubits)
        for i, p in enumerate(self.params):
            qc.ry(np.pi * p, i)
        self.feature_map = qc

        self.sampler = Sampler()
        self.fidelity = ComputeUncompute(sampler=self.sampler)
        self.qkernel = FidelityQuantumKernel(
            feature_map=self.feature_map, fidelity=self.fidelity
        )

    def predict(self, episode: Episode) -> np.ndarray:
        qsvc = QSVC(quantum_kernel=self.qkernel)
        qsvc.fit(episode.support_x, episode.support_y)
        return qsvc.predict(episode.query_x)

    def __call__(self, episode: Episode) -> np.ndarray:
        return self.predict(episode)


# ==============================================================================
# Trained Classical ProtoNet
# ==============================================================================

class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrainedProtoNet:
    """
    Classical Trained ProtoNet.
    Trained episodically on the train dataset using Snell et al. loss.
    """

    def __init__(
        self, input_dim: int, output_dim: int = 32, hidden_dim: int = 128
    ) -> None:
        self.encoder = MLPEncoder(input_dim, hidden_dim, output_dim)

    def train_epoch(self, sampler: EpisodeSampler, n_episodes: int = 100, lr: float = 1e-3) -> float:
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
                mask = ep.support_y == c
                proto = s_z[mask].mean(dim=0)
                prototypes.append(proto)

            prototypes = torch.stack(prototypes)

            # Distances (squared euclidean)
            dists = torch.cdist(q_z, prototypes, p=2).pow(2)

            # Logits are negative distances
            logits = -dists

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
                mask = episode.support_y == c
                proto = s_z[mask].mean(dim=0)
                prototypes.append(proto)

            prototypes = torch.stack(prototypes)
            dists = torch.cdist(q_z, prototypes, p=2)
            preds = torch.argmin(dists, dim=1).numpy()

            return classes[preds]

    def __call__(self, episode: Episode) -> np.ndarray:
        return self.predict(episode)
