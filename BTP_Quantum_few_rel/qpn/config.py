# This module stores experiment configuration dataclasses and shared defaults for episode generation.
# It centralizes seed handling and reusable training settings for the benchmark scripts.
import random
import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class EpisodeConfig:
    n_way: int
    k_shot: int
    q_queries: int = 15
    n_episodes: int = 600

@dataclass
class QCHBAConfig:
    n_features: int = 8
    per_episode: bool = False # Default: fit QCHBA once on train relations

STANDARD_SETTINGS = {
    "5w1s": EpisodeConfig(n_way=5, k_shot=1, q_queries=15, n_episodes=600),
    "5w5s": EpisodeConfig(n_way=5, k_shot=5, q_queries=15, n_episodes=600),
    "10w1s": EpisodeConfig(n_way=10, k_shot=1, q_queries=15, n_episodes=600),
    "10w5s": EpisodeConfig(n_way=10, k_shot=5, q_queries=15, n_episodes=600),
}

def set_seed(seed: int):
    """
    Sets the seed for numpy, torch, and python random to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
