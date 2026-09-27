"""Fixed-episode evaluation helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import EpisodeConfig
from .data import RelationPools
from .episodes import Episode, EpisodeSampler
from .metrics import classification_metrics, mean_confidence_interval
from .model import HybridQuantumProtoNet


def evaluate_episodes(model: HybridQuantumProtoNet, episodes: Sequence[Episode]) -> dict[str, object]:
    episode_metrics: list[dict[str, float]] = []
    for episode in episodes:
        predictions = model.predict_episode(episode)
        episode_metrics.append(classification_metrics(episode.query_y, predictions))

    accuracies = [item["accuracy"] for item in episode_metrics]
    f1_values = [item["weighted_f1"] for item in episode_metrics]
    accuracy_mean, accuracy_ci = mean_confidence_interval(accuracies)
    f1_mean, f1_ci = mean_confidence_interval(f1_values)
    return {
        "episodes": len(episodes),
        "accuracy_mean": accuracy_mean,
        "accuracy_95ci": accuracy_ci,
        "weighted_f1_mean": f1_mean,
        "weighted_f1_95ci": f1_ci,
        "episode_metrics": episode_metrics,
    }


def make_fixed_episodes(
    relation_pools: RelationPools,
    episode_config: EpisodeConfig | None = None,
    count: int = 20,
    seed: int = 17,
) -> list[Episode]:
    sampler = EpisodeSampler(relation_pools, episode_config or EpisodeConfig(), seed=seed)
    return sampler.sample_many(count)

