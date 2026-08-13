"""
harness.py
Evaluation harness for few-shot episodes.

Mirrors BTP_Quantum_few_rel/qpn/eval/harness.py but adapted for TREC-50.
Evaluates any model that provides a `predict(episode)` method (works for
both QuantumProtoNet and classical baselines).
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Any

from sklearn.metrics import accuracy_score, f1_score
from data.episode_sampler import Episode, EpisodeSampler


def evaluate(
    model_fn: Callable[[Episode], np.ndarray],
    sampler: EpisodeSampler,
    n_episodes: int = 100,
) -> Dict[str, Any]:
    """
    Evaluate a model across N few-shot episodes.

    Args:
        model_fn:   A callable that takes an Episode and returns a 1D array
                    of predicted class indices (0..N-1) for the query set.
        sampler:    An initialized EpisodeSampler for the test dataset.
        n_episodes: Number of episodes to sample and evaluate.

    Returns:
        Dict with aggregated 'accuracy' and 'f1' scores, including mean,
        std, and 95% confidence intervals, plus the raw score lists.
    """
    episodes = sampler.sample(n_episodes)
    acc_scores = []
    f1_scores = []

    for ep in episodes:
        # Expected predictions shape: (N*Q,)
        preds = model_fn(ep)

        # Ground truth labels shape: (N*Q,)
        truths = ep.query_y

        acc = accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average="weighted", zero_division=0)

        acc_scores.append(acc)
        f1_scores.append(f1)

    acc_mean = np.mean(acc_scores)
    acc_std = np.std(acc_scores)
    acc_ci = 1.96 * (acc_std / np.sqrt(n_episodes))

    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)
    f1_ci = 1.96 * (f1_std / np.sqrt(n_episodes))

    return {
        "accuracy": {
            "mean": float(acc_mean * 100),
            "std": float(acc_std * 100),
            "ci95": float(acc_ci * 100),
            "raw": acc_scores,
        },
        "f1": {
            "mean": float(f1_mean * 100),
            "std": float(f1_std * 100),
            "ci95": float(f1_ci * 100),
            "raw": f1_scores,
        },
        "n_episodes": n_episodes,
    }
