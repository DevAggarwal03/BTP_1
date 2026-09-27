"""The small trainable classical compression layer."""

from __future__ import annotations

import math

import torch
from torch import nn


class CompressionNetwork(nn.Module):
    """Map a 384-dimensional sentence embedding to bounded qubit angles.

    The default path is LayerNorm -> Linear(384, 64) -> GELU ->
    Linear(64, n_qubits) -> pi*tanh.  The final tanh keeps angles in a
    predictable range for stable angle encoding.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 64,
        n_qubits: int = 4,
        mode: str = "mlp",
    ) -> None:
        super().__init__()
        if mode not in {"mlp", "linear"}:
            raise ValueError("compression mode must be 'mlp' or 'linear'")
        self.n_qubits = n_qubits
        self.mode = mode
        self.normalise = nn.LayerNorm(input_dim)
        if mode == "linear":
            self.project = nn.Linear(input_dim, n_qubits)
        else:
            self.project = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_qubits),
            )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected [batch, embedding_dim], got {tuple(embeddings.shape)}")
        raw_angles = self.project(self.normalise(embeddings))
        return math.pi * torch.tanh(raw_angles)

