"""
train.py
Episodic meta-training loop for the Quantum Prototypical Network.

Mirrors BTP_Quantum_few_rel/qpn/eval/train.py but adapted for the
TREC-50 integer-label class structure and the local QuantumProtoNet model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from quantum.training.qpn_model import QuantumProtoNet
from data.episode_sampler import EpisodeSampler, build_class_pool


def meta_train_qpn(
    model: QuantumProtoNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_way: int = 5,
    k_shot: int = 1,
    n_query: int = 15,
    n_train_episodes: int = 100,
    lr: float = 0.01,
    lr_step_size: int = 15,
    lr_gamma: float = 0.5,
) -> QuantumProtoNet:
    """
    Episodic meta-training loop for QuantumProtoNet.

    For each episode:
      1. Sample N random classes + K support + Q query examples per class.
      2. Run the full QPN forward: encoding → VQC → prototypes → distances → logits.
      3. Compute CrossEntropyLoss on the query logits.
      4. Backpropagate via PyTorch (gradients w.r.t. VQC angles θ and temperature β).
      5. Step Adam optimizer + StepLR scheduler.

    Args:
        model:            Initialised QuantumProtoNet instance.
        X_train:          Training feature matrix (N_samples, n_qubits).
        y_train:          Training integer labels (N_samples,).
        n_way:            Number of classes per episode.
        k_shot:           Support examples per class.
        n_query:          Query examples per class.
        n_train_episodes: Total number of training episodes.
        lr:               Adam learning rate.
        lr_step_size:     StepLR step size (episodes between decays).
        lr_gamma:         StepLR decay factor.

    Returns:
        The trained QuantumProtoNet (same object, mutated in-place).
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    loss_fn = nn.CrossEntropyLoss()

    class_pool = build_class_pool(X_train, y_train)
    sampler = EpisodeSampler(class_pool, n_way=n_way, k_shot=k_shot, n_query=n_query)
    episodes = sampler.sample(n_episodes=n_train_episodes)

    total_loss = 0.0

    for ep_idx, ep in enumerate(episodes):
        optimizer.zero_grad()

        s_x = torch.tensor(ep.support_x, dtype=torch.float32)
        q_x = torch.tensor(ep.query_x, dtype=torch.float32)
        s_y = torch.tensor(ep.support_y, dtype=torch.long)
        q_y = torch.tensor(ep.query_y, dtype=torch.long)

        # Real QPN forward: prototypes → distances → logits
        logits = model(s_x, s_y, q_x)

        loss = loss_fn(logits, q_y)
        loss.backward()

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        print(
            f"  [Train] Ep {ep_idx + 1:3d}/{n_train_episodes} "
            f"- Loss: {loss.item():.4f}"
        )

    avg_loss = total_loss / n_train_episodes
    print(f"  => Meta-Training Complete. Avg Loss: {avg_loss:.4f}")
    return model
