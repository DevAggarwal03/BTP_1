"""Classical comparison models sharing the same episode protocol."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .compression import CompressionNetwork
from .episodes import Episode


class ClassicalProtoNet(nn.Module):
    """A fair classical baseline using the same compression input and episodes."""

    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim: int = 64,
        output_dim: int = 4,
        distance: str = "euclidean",
        compression_mode: str = "mlp",
    ):
        super().__init__()
        if distance not in {"euclidean", "cosine"}:
            raise ValueError("distance must be 'euclidean' or 'cosine'")
        self.distance = distance
        self.compression = CompressionNetwork(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_qubits=output_dim,
            mode=compression_mode,
        )

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        support = self.compression(support_x)
        query = self.compression(query_x)
        num_classes = int(support_y.max().item()) + 1
        prototypes = torch.stack(
            [support[support_y == class_id].mean(dim=0) for class_id in range(num_classes)]
        )
        if self.distance == "euclidean":
            return -torch.cdist(query, prototypes).square()
        query = F.normalize(query, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)
        return query @ prototypes.transpose(0, 1)

    @torch.no_grad()
    def predict_episode(self, episode: Episode) -> np.ndarray:
        self.eval()
        support_x = torch.as_tensor(episode.support_x, dtype=torch.float32)
        support_y = torch.as_tensor(episode.support_y, dtype=torch.long)
        query_x = torch.as_tensor(episode.query_x, dtype=torch.float32)
        return self(support_x, support_y, query_x).argmax(dim=1).cpu().numpy()


class FixedQuantumProtoNet(nn.Module):
    """Frozen exact quantum feature map used as a comparison baseline."""

    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 64, n_qubits: int = 4):
        from .model import HybridQuantumProtoNet
        from .config import ModelConfig

        super().__init__()
        self.model = HybridQuantumProtoNet(
            ModelConfig(
                embedding_dim=embedding_dim,
                compression_hidden_dim=hidden_dim,
                n_qubits=n_qubits,
            )
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor):
        return self.model(support_x, support_y, query_x)[0]

    @torch.no_grad()
    def predict_episode(self, episode: Episode) -> np.ndarray:
        return self.model.predict_episode(episode)
