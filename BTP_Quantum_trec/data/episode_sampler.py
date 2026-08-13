"""
episode_sampler.py
Few-shot episode sampler for the TREC-50 dataset.

Adapted from BTP_Quantum_few_rel/qpn/episodes.py for TREC-50's
integer-label class structure (classes 0–49) rather than relation strings.

Key difference from few_rel version:
  - Classes are integers (0..49), not relation strings.
  - Handles TREC-50 class imbalance by filtering out classes with fewer
    than (k_shot + n_query) examples before sampling.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class Episode:
    """
    A single N-way K-shot few-shot episode.

    All class labels are remapped to 0, 1, ..., N-1 within the episode.
    """
    support_x: np.ndarray   # shape: (N * K, D)
    support_y: np.ndarray   # shape: (N * K,)  — values in 0..N-1
    query_x: np.ndarray     # shape: (N * Q, D)
    query_y: np.ndarray     # shape: (N * Q,)  — values in 0..N-1
    class_ids: List[int]    # Original TREC-50 integer class indices (length N)


class EpisodeSampler:
    """
    Samples N-way K-shot episodes from a class-indexed feature pool.

    Args:
        class_pool: Dict mapping integer class_id → np.ndarray of shape (n_i, D).
                    Classes with fewer than (k_shot + n_query) examples are
                    automatically excluded.
        n_way:    Number of classes per episode.
        k_shot:   Number of support examples per class.
        n_query:  Number of query examples per class.
    """

    def __init__(
        self,
        class_pool: Dict[int, np.ndarray],
        n_way: int,
        k_shot: int,
        n_query: int,
    ) -> None:
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

        required = k_shot + n_query
        # Filter classes that don't have enough examples
        self.eligible_classes = [
            cls_id
            for cls_id, instances in class_pool.items()
            if len(instances) >= required
        ]
        self.class_pool = {
            cls_id: class_pool[cls_id]
            for cls_id in self.eligible_classes
        }

        if len(self.eligible_classes) < n_way:
            raise ValueError(
                f"Only {len(self.eligible_classes)} eligible classes "
                f"(need ≥ {k_shot + n_query} examples each), "
                f"but {n_way}-way sampling requested."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, n_episodes: int = 1) -> List[Episode]:
        """Return a list of n_episodes sampled episodes."""
        return [self._sample_one() for _ in range(n_episodes)]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_one(self) -> Episode:
        """Sample a single episode."""
        selected_classes = random.sample(self.eligible_classes, self.n_way)

        support_x_list, support_y_list = [], []
        query_x_list, query_y_list = [], []

        for local_idx, cls_id in enumerate(selected_classes):
            instances = self.class_pool[cls_id]
            n_needed = self.k_shot + self.n_query
            chosen = random.sample(range(len(instances)), n_needed)

            support_idx = chosen[: self.k_shot]
            query_idx = chosen[self.k_shot :]

            support_x_list.append(instances[support_idx])
            support_y_list.extend([local_idx] * self.k_shot)

            query_x_list.append(instances[query_idx])
            query_y_list.extend([local_idx] * self.n_query)

        return Episode(
            support_x=np.concatenate(support_x_list, axis=0),
            support_y=np.array(support_y_list, dtype=int),
            query_x=np.concatenate(query_x_list, axis=0),
            query_y=np.array(query_y_list, dtype=int),
            class_ids=selected_classes,
        )


# ------------------------------------------------------------------
# Helper: build class_pool from (X, y) arrays
# ------------------------------------------------------------------

def build_class_pool(X: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Organise feature matrix X and label vector y into a class-indexed pool.

    Args:
        X: Feature matrix, shape (N, D).
        y: Integer labels, shape (N,).

    Returns:
        Dict mapping class_id -> np.ndarray of shape (n_i, D).
    """
    pool: Dict[int, np.ndarray] = {}
    for cls_id in np.unique(y):
        pool[int(cls_id)] = X[y == cls_id]
    return pool
