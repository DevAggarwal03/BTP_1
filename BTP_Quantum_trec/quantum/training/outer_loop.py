import numpy as np
import torch
from typing import Tuple, Callable
from quantum.feature_selection.qhba import QHBA, QHBAResult
from quantum.training.qpn_model import QuantumPrototypicalNetwork
from quantum.training.trainer import MetaLearningTrainer

class QPNMasterTrainer:
    """
    Master Outer Loop Controller.
    Connects the Quantum Honey Badger Algorithm (QHBA) to the VQC inner loop trainer.
    """

    def __init__(self, 
                 n_features: int, 
                 epochs_per_eval: int = 2,
                 n_qubits: int = 8,
                 learning_rate: float = 0.01,
                 num_classes: int = 50):
        """
        Args:
            n_features (int): Total classical features available.
            epochs_per_eval (int): How many epochs to train the VQC during a single 
                                   QHBA guess. We use a low number (Early Stopping)
                                   to make the search computationally feasible.
            n_qubits (int): Qubits in the VQC.
            learning_rate (float): Optimizer learning rate.
            num_classes (int): Number of target classes.
        """
        self.qhba = QHBA(n_features=n_features)
        self.epochs_per_eval = epochs_per_eval
        self.n_qubits = n_qubits
        self.learning_rate = learning_rate
        self.num_classes = num_classes

    def create_fitness_fn(self, X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], float]:
        """
        Creates a custom fitness function for the QHBA.
        
        Args:
            X (np.ndarray): Full dataset features.
            y (np.ndarray): Full dataset labels.
            
        Returns:
            Callable: A function that takes a continuous position array, trains the VQC,
                      and returns the final loss.
        """
        def fitness_fn(position: np.ndarray) -> float:
            # 1. Binarize the position to get the feature mask (which features are used)
            mask = np.where(position > 0.5, 1, 0)
            selected_features = np.where(mask == 1)[0]
            
            # If no features selected or more than n_qubits, penalize heavily
            if len(selected_features) == 0 or len(selected_features) > self.n_qubits:
                return 1e6
                
            # 2. Filter dataset using mask (Mocking support/query split for simplicity)
            X_filtered = torch.tensor(X[:, selected_features], dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            
            # Pad features to match qubits if necessary
            if X_filtered.shape[1] < self.n_qubits:
                padding = torch.zeros(X_filtered.shape[0], self.n_qubits - X_filtered.shape[1])
                X_filtered = torch.cat([X_filtered, padding], dim=1)

            # 3. Initialize fresh VQC model and inner trainer
            model = QuantumPrototypicalNetwork(num_qubits=self.n_qubits, num_classes=self.num_classes)
            trainer = MetaLearningTrainer(model=model, learning_rate=self.learning_rate)
            
            # 4. Train for a few epochs (Early-Stopping Evaluation)
            final_loss = 0.0
            for _ in range(self.epochs_per_eval):
                # In full implementation, X_filtered is split into support and query.
                # Here we pass it directly to mock the structural training step.
                final_loss = trainer.train_step(support_x=X_filtered, support_y=y_tensor, 
                                                query_x=X_filtered, query_y=y_tensor)
            
            # QHBA minimizes fitness, so lower loss is better.
            return final_loss

        return fitness_fn

    def fit(self, X: np.ndarray, y: np.ndarray) -> QHBAResult:
        """
        Runs the full Two-Loop Master Training process.
        """
        print(f"Starting QPN Master Training (Outer Loop: QHBA, Inner Loop: VQC with {self.epochs_per_eval} epochs/eval)...")
        fitness_fn = self.create_fitness_fn(X, y)
        
        # Pass the custom fitness function to override the default KNN logic
        result = self.qhba.fit(X=X, y=y, fitness_fn=fitness_fn, verbose=True)
        return result
