"""Deterministic 5-way few-shot episode sampling."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping, Sequence

import numpy as np

from .config import EpisodeConfig


@dataclass(frozen=True)
class Episode:
    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray
    classes: tuple[str, ...]


class EpisodeSampler:
    """Sample independent episodes without changing the input pools."""

    def __init__(self, relation_pools: Mapping[str, np.ndarray], config: EpisodeConfig, seed: int = 7):
        self.relation_pools = relation_pools
        self.config = config
        self.rng = random.Random(seed)

        if len(relation_pools) < config.n_way:
            raise ValueError("There are fewer relation pools than n_way.")
        minimum = config.k_shot + config.q_queries
        too_small = [name for name, values in relation_pools.items() if len(values) < minimum]
        if too_small:
            raise ValueError(
                f"Relations need at least {minimum} vectors for this episode; too small: {too_small[:3]}"
            )

    def sample(self) -> Episode:
        class_names = tuple(self.rng.sample(list(self.relation_pools), self.config.n_way))
        support_parts: list[np.ndarray] = []
        query_parts: list[np.ndarray] = []
        support_labels: list[int] = []
        query_labels: list[int] = []

        for label, relation in enumerate(class_names):
            values = self.relation_pools[relation]
            indices = self.rng.sample(range(len(values)), self.config.k_shot + self.config.q_queries)
            support_indices = indices[: self.config.k_shot]
            query_indices = indices[self.config.k_shot :]
            support_parts.append(np.asarray(values[support_indices], dtype=np.float32))
            query_parts.append(np.asarray(values[query_indices], dtype=np.float32))
            support_labels.extend([label] * self.config.k_shot)
            query_labels.extend([label] * self.config.q_queries)

        return Episode(
            support_x=np.concatenate(support_parts, axis=0),
            support_y=np.asarray(support_labels, dtype=np.int64),
            query_x=np.concatenate(query_parts, axis=0),
            query_y=np.asarray(query_labels, dtype=np.int64),
            classes=class_names,
        )

    def sample_many(self, count: int) -> list[Episode]:
        return [self.sample() for _ in range(count)]


def episode_from_arrays(
    support_x: np.ndarray,
    support_y: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
) -> Episode:
    """Convenience helper for tests and controlled experiments."""

    n_classes = int(np.max(support_y)) + 1
    return Episode(
        support_x=np.asarray(support_x, dtype=np.float32),
        support_y=np.asarray(support_y, dtype=np.int64),
        query_x=np.asarray(query_x, dtype=np.float32),
        query_y=np.asarray(query_y, dtype=np.int64),
        classes=tuple(str(i) for i in range(n_classes)),
    )

