"""
trainer.py
Inner-loop trainer for the Quantum Prototypical Network.

Wraps the QuantumProtoNet model with an Adam optimizer and provides
a train_step() method that executes one episodic forward + backward pass.

This file is kept for backward compatibility with the outer_loop.py.
The canonical training loop is in quantum/eval/train.py (meta_train_qpn).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from quantum.training.qpn_model import QuantumProtoNet


class MetaLearningTrainer:
    """
    Episodic inner-loop trainer for QuantumProtoNet.

    Args:
        model:         A QuantumProtoNet instance.
        learning_rate: Adam optimizer learning rate.
        lr_step_size:  StepLR step size (episodes between LR decay).
        lr_gamma:      StepLR gamma (multiplicative decay factor).
    """

    def __init__(
        self,
        model: QuantumProtoNet,
        learning_rate: float = 0.01,
        lr_step_size: int = 15,
        lr_gamma: float = 0.5,
    ) -> None:
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(
            self.optimizer, step_size=lr_step_size, gamma=lr_gamma
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> float:
        """
        Execute one episodic training step.

        The model computes quantum prototypes from the support set and
        measures infidelity distances to each query — all real quantum ops.
        CrossEntropyLoss is then computed on those distance-based logits.

        Args:
            support_x: float tensor (N*K, n_qubits)
            support_y: long  tensor (N*K,)  — labels 0..N-1
            query_x:   float tensor (N*Q, n_qubits)
            query_y:   long  tensor (N*Q,)  — labels 0..N-1

        Returns:
            float: Loss value for this episode.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Real QPN forward: support → prototypes → distances → logits
        logits = self.model(support_x, support_y, query_x)  # (N*Q, N)

        loss = self.loss_fn(logits, query_y)
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

