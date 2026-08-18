"""
harness.py
Evaluation harness for few-shot episodes.

Mirrors BTP_Quantum_few_rel/qpn/eval/harness.py:
- Uses scipy.stats.t for t-distribution confidence intervals (not z-score).
- Prints progress at every 10% checkpoint.
- Returns 'f1_weighted' key (matching few_rel's return schema).
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Any

from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from data.episode_sampler import Episode, EpisodeSampler


def mean_confidence_interval(data: list, confidence: float = 0.95):
    """
    Compute mean and t-distribution confidence interval.

    Mirrors few_rel's harness.py mean_confidence_interval.

    Args:
        data:       List of scalar scores.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        (mean, half_interval) tuple.
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


def evaluate(
    model_fn: Callable[[Episode], np.ndarray],
    sampler: EpisodeSampler,
    n_episodes: int = 300,
) -> Dict[str, Any]:
    """
    Evaluate a model across N few-shot episodes.

    Mirrors few_rel's evaluate() in qpn/eval/harness.py:
    - Prints progress at every 10% of episodes.
    - Uses t-distribution CI (not z-score approximation).
    - Returns 'f1_weighted' key (not 'f1') to match few_rel's schema.

    Args:
        model_fn:   Callable(Episode) -> np.ndarray of predicted class indices.
        sampler:    Initialized EpisodeSampler for the test pool.
        n_episodes: Number of test episodes to evaluate.

    Returns:
        Dict with 'accuracy' and 'f1_weighted', each containing:
            mean, ci95, raw (list of per-episode scores).
    """
    episodes = sampler.sample(n_episodes)
    acc_scores = []
    f1_scores = []

    for i, ep in enumerate(episodes):
        preds = model_fn(ep)
        truths = ep.query_y

        acc = accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average="weighted", zero_division=0)

        acc_scores.append(acc)
        f1_scores.append(f1)

        # Print progress at every 10% checkpoint (matching few_rel)
        if (i + 1) % max(1, len(episodes) // 10) == 0 or (i + 1) == len(episodes):
            print(
                f"    [Eval] {i + 1}/{len(episodes)} episodes complete. "
                f"(Running Acc: {np.mean(acc_scores):.4f})"
            )

    mean_acc, ci_acc = mean_confidence_interval(acc_scores)
    mean_f1, ci_f1   = mean_confidence_interval(f1_scores)

    return {
        "accuracy": {
            "mean": float(mean_acc * 100),
            "ci95": float(ci_acc * 100),
            "raw":  acc_scores,
        },
        "f1_weighted": {
            "mean": float(mean_f1 * 100),
            "ci95": float(ci_f1  * 100),
            "raw":  f1_scores,
        },
        "n_episodes": n_episodes,
    }
