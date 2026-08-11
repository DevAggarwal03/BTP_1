# This module evaluates trained or baseline models over many episodes and reports standard metrics.
# It ties together episode sampling, predictions, and aggregate scoring for benchmark experiments.
import numpy as np
from typing import Callable, List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats

from qpn.episodes import Episode, EpisodeSampler

def mean_confidence_interval(data, confidence=0.95):
    """
    Computes the mean and confidence interval for a given array of data.
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def evaluate(model_fn: Callable[[Episode], np.ndarray], sampler: EpisodeSampler, n_episodes: int = None) -> Dict[str, Any]:
    """
    Evaluates a model function on a stream of episodes.
    
    model_fn: A function that takes an Episode and returns predicted query labels (shape: (N*Q,))
    sampler: An EpisodeSampler
    n_episodes: Overrides the sampler's default number of episodes if provided.
    
    Returns a dictionary containing mean, 95% CI, and raw per-episode scores for accuracy and weighted-F1.
    """
    episodes = sampler.sample(n_episodes)
    
    acc_scores = []
    f1_scores = []
    
    for i, episode in enumerate(episodes):
        preds = model_fn(episode)
        
        acc = accuracy_score(episode.query_y, preds)
        f1 = f1_score(episode.query_y, preds, average='weighted')
        
        acc_scores.append(acc)
        f1_scores.append(f1)
        
        if (i + 1) % max(1, len(episodes) // 10) == 0 or (i + 1) == len(episodes):
            print(f"    [Eval] {i+1}/{len(episodes)} episodes complete. (Current Acc: {np.mean(acc_scores):.4f})")

        
    mean_acc, ci_acc = mean_confidence_interval(acc_scores)
    mean_f1, ci_f1 = mean_confidence_interval(f1_scores)
    
    return {
        "accuracy": {
            "mean": mean_acc,
            "ci95": ci_acc,
            "raw": acc_scores
        },
        "f1_weighted": {
            "mean": mean_f1,
            "ci95": ci_f1,
            "raw": f1_scores
        }
    }
