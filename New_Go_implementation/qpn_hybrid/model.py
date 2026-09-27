"""End-to-end hybrid quantum prototype classifier."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .compression import CompressionNetwork
from .config import ModelConfig
from .episodes import Episode
from .prototypes import prototype_fidelity_scores
from .quantum_encoder import DataReuploadingQuantumEncoder


def inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("beta_init must be positive")
    return math.log(math.expm1(value))


class HybridQuantumProtoNet(nn.Module):
    """Compression -> exact statevector -> density prototypes -> fidelity logits."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.compression = CompressionNetwork(
            input_dim=self.config.embedding_dim,
            hidden_dim=self.config.compression_hidden_dim,
            n_qubits=self.config.n_qubits,
            mode=self.config.compression_mode,
        )
        self.quantum = DataReuploadingQuantumEncoder(
            n_qubits=self.config.n_qubits,
            reps=self.config.quantum_reps,
        )
        self.raw_beta = nn.Parameter(
            torch.tensor(inverse_softplus(self.config.beta_init), dtype=torch.float64)
        )

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_beta)

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        angles = self.compression(embeddings)
        return self.quantum(angles)

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        support_states = self.encode(support_x)
        query_states = self.encode(query_x)
        num_classes = int(support_y.max().item()) + 1
        scores = prototype_fidelity_scores(query_states, support_states, support_y, num_classes)
        logits = self.beta * scores
        return logits, scores

    def episode_loss(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, scores = self(support_x, support_y, query_x)
        loss = F.cross_entropy(logits, query_y)
        return loss, logits, scores

    @torch.no_grad()
    def predict_episode(self, episode: Episode) -> np.ndarray:
        self.eval()
        support_x = torch.as_tensor(episode.support_x, dtype=torch.float32)
        support_y = torch.as_tensor(episode.support_y, dtype=torch.long)
        query_x = torch.as_tensor(episode.query_x, dtype=torch.float32)
        logits, _ = self(support_x, support_y, query_x)
        return logits.argmax(dim=1).cpu().numpy()

