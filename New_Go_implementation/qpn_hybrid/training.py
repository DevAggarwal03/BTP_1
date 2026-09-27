"""Episodic training utilities."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import EpisodeConfig, TrainConfig
from .data import RelationPools
from .episodes import Episode, EpisodeSampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _episode_tensors(episode: Episode) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(episode.support_x, dtype=torch.float32),
        torch.as_tensor(episode.support_y, dtype=torch.long),
        torch.as_tensor(episode.query_x, dtype=torch.float32),
        torch.as_tensor(episode.query_y, dtype=torch.long),
    )


def train_one_episode(
    model: nn.Module,
    episode: Episode,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    support_x, support_y, query_x, query_y = _episode_tensors(episode)
    optimizer.zero_grad(set_to_none=True)
    if hasattr(model, "episode_loss"):
        loss, _, _ = model.episode_loss(support_x, support_y, query_x, query_y)
    else:
        logits = model(support_x, support_y, query_x)
        loss = F.cross_entropy(logits, query_y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.detach().cpu())


def meta_train(
    model: nn.Module,
    relation_pools: RelationPools,
    episode_config: EpisodeConfig | None = None,
    train_config: TrainConfig | None = None,
    verbose: bool = True,
) -> list[float]:
    episode_config = episode_config or EpisodeConfig()
    train_config = train_config or TrainConfig()
    set_seed(train_config.seed)
    sampler = EpisodeSampler(relation_pools, episode_config, seed=train_config.seed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    losses: list[float] = []
    for episode_number in range(train_config.train_episodes):
        loss = train_one_episode(model, sampler.sample(), optimizer)
        losses.append(loss)
        if verbose and (episode_number == 0 or (episode_number + 1) % 10 == 0):
            print(f"episode {episode_number + 1}/{train_config.train_episodes}: loss={loss:.4f}")
    return losses
