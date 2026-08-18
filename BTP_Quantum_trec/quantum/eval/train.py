"""
train.py
Episodic meta-training loop for the Quantum Prototypical Network.

Mirrors BTP_Quantum_few_rel/qpn/eval/train.py — accepts a global_preprocessor
to transform support/query features into amplitude-encoded vectors before the
quantum forward pass.
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
    global_preprocessor,
    n_way: int = 5,
    k_shot: int = 1,
    n_query: int = 15,
    n_train_episodes: int = 100,
    lr: float = 0.05,
    lr_step_size: int = 15,
    lr_gamma: float = 0.5,
) -> QuantumProtoNet:
    """
    Episodic meta-training loop for QuantumProtoNet.

    Mirrors few_rel's meta_train_qpn in qpn/eval/train.py:
    - Samples N-way K-shot episodes from the LDA feature pool.
    - Transforms support/query features via the global QuantumFeaturePreprocessor
      (amplitude-encoding: pad to 2^n_qubits and L2-normalize).
    - Runs the differentiable QPN forward pass (FidelityParamShift).
    - Backpropagates CrossEntropyLoss through PSR gradients.
    - Steps Adam + StepLR scheduler.

    Args:
        model:                Initialized QuantumProtoNet instance.
        X_train:              LDA feature matrix (N_samples, n_lda).
        y_train:              Integer label array (N_samples,).
        global_preprocessor:  Fitted QuantumFeaturePreprocessor for amplitude encoding.
        n_way:                Number of classes per episode.
        k_shot:               Support examples per class.
        n_query:              Query examples per class.
        n_train_episodes:     Total training episodes.
        lr:                   Adam learning rate.
        lr_step_size:         StepLR step size.
        lr_gamma:             StepLR decay factor.

    Returns:
        The trained QuantumProtoNet (mutated in-place).
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

        # Amplitude-encode support and query (mirroring few_rel's transform call)
        s_x_enc = global_preprocessor.transform(ep.support_x)
        q_x_enc = global_preprocessor.transform(ep.query_x)

        # Map support labels to contiguous 0..N-1 (as few_rel does)
        classes = list(np.unique(ep.support_y))
        s_y_mapped = np.array([classes.index(int(y)) for y in ep.support_y])
        q_y_mapped = np.array([classes.index(int(y)) for y in ep.query_y])

        s_x_t = torch.tensor(s_x_enc, dtype=torch.float32)
        q_x_t = torch.tensor(q_x_enc, dtype=torch.float32)
        s_y_t = torch.tensor(s_y_mapped, dtype=torch.long)
        q_y_t = torch.tensor(q_y_mapped, dtype=torch.long)

        logits = model(s_x_t, s_y_t, q_x_t)  # (Q, N)
        loss = loss_fn(logits, q_y_t)
        loss.backward()

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        print(f"  [Train] Ep {ep_idx + 1:3d}/{n_train_episodes} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / n_train_episodes
    print(f"  => Meta-Training Complete. Avg Loss: {avg_loss:.4f}")
    return model
