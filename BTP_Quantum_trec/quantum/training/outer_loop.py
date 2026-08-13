"""
outer_loop.py
Master Outer Loop: QHBA + Episodic VQC Inner Loop.

The QPNMasterTrainer connects the Quantum Honey Badger Algorithm (QHBA)
outer loop to the episodic VQC inner loop.  For each QHBA candidate
feature mask, the fitness function:

    1. Selects the corresponding feature columns.
    2. Samples a proper N-way K-shot episode from the dataset.
    3. Trains the QPN for ``epochs_per_eval`` episodic steps.
    4. Returns the final episode loss as the fitness score.

QHBA minimises the fitness, so lower loss → better feature mask.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Callable

from quantum.feature_selection.qhba import QHBA, QHBAResult
from quantum.training.qpn_model import QuantumProtoNet
from quantum.training.trainer import MetaLearningTrainer
from data.episode_sampler import EpisodeSampler, build_class_pool


class QPNMasterTrainer:
    """
    Master Outer Loop Controller.

    Args:
        n_features:      Total classical features available.
        epochs_per_eval: Episodic steps per QHBA fitness evaluation.
        n_qubits:        Number of qubits in the VQC.
        learning_rate:   Adam optimizer LR for inner loop.
        n_way:           N-way few-shot episodes inside fitness function.
        k_shot:          K-shot few-shot episodes inside fitness function.
        n_query:         Query examples per class in fitness function.
    """

    def __init__(
        self,
        n_features: int,
        epochs_per_eval: int = 2,
        n_qubits: int = 8,
        learning_rate: float = 0.01,
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 5,
    ) -> None:
        self.qhba = QHBA(n_features=n_features)
        self.epochs_per_eval = epochs_per_eval
        self.n_qubits = n_qubits
        self.learning_rate = learning_rate
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

    def create_fitness_fn(
        self, X: np.ndarray, y: np.ndarray
    ) -> Callable[[np.ndarray], float]:
        """
        Creates a QHBA fitness function that uses proper episodic training.

        Args:
            X: Full dataset features (N_samples, n_features).
            y: Full dataset integer labels (N_samples,).

        Returns:
            Callable: position → scalar loss (lower = better feature mask).
        """
        # Pre-build class pool once (reused across all fitness evaluations)
        class_pool = build_class_pool(X, y)

        n_way = self.n_way
        k_shot = self.k_shot
        n_query = self.n_query
        n_qubits = self.n_qubits
        lr = self.learning_rate
        epochs = self.epochs_per_eval

        def fitness_fn(position: np.ndarray) -> float:
            # 1. Binarize position → feature mask
            mask = np.where(position > 0.5, 1, 0)
            selected_features = np.where(mask == 1)[0]

            if len(selected_features) == 0 or len(selected_features) > n_qubits:
                return 1e6

            # 2. Build filtered class pool for this feature mask
            filtered_pool = {
                cls_id: instances[:, selected_features]
                for cls_id, instances in class_pool.items()
            }

            # Pad to n_qubits if fewer features selected
            n_sel = selected_features.shape[0]
            if n_sel < n_qubits:
                pad = n_qubits - n_sel
                filtered_pool = {
                    cls_id: np.pad(instances, ((0, 0), (0, pad)))
                    for cls_id, instances in filtered_pool.items()
                }

            # 3. Sample one proper N-way K-shot episode
            try:
                sampler = EpisodeSampler(filtered_pool, n_way, k_shot, n_query)
                episodes = sampler.sample(n_episodes=epochs)
            except ValueError:
                return 1e6  # Not enough classes with sufficient examples

            # 4. Fresh QPN model + trainer for this fitness eval
            model = QuantumProtoNet(n_qubits=n_qubits)
            trainer = MetaLearningTrainer(model=model, learning_rate=lr)

            final_loss = 0.0
            for ep in episodes:
                s_x = torch.tensor(ep.support_x, dtype=torch.float32)
                s_y = torch.tensor(ep.support_y, dtype=torch.long)
                q_x = torch.tensor(ep.query_x, dtype=torch.float32)
                q_y = torch.tensor(ep.query_y, dtype=torch.long)
                final_loss = trainer.train_step(s_x, s_y, q_x, q_y)

            return final_loss

        return fitness_fn

    def fit(self, X: np.ndarray, y: np.ndarray) -> QHBAResult:
        """Run the full Two-Loop Master Training (QHBA outer + QPN inner)."""
        print(
            f"Starting QPN Master Training "
            f"(QHBA outer loop + {self.epochs_per_eval} episodic steps/eval)..."
        )
        fitness_fn = self.create_fitness_fn(X, y)
        result = self.qhba.fit(X=X, y=y, fitness_fn=fitness_fn, verbose=True)
        return result

