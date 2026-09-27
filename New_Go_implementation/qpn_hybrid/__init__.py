"""Hybrid classical--quantum metric-learning prototype for FewRel."""

from .config import EpisodeConfig, ModelConfig, TrainConfig
from .model import HybridQuantumProtoNet

__all__ = [
    "EpisodeConfig",
    "ModelConfig",
    "TrainConfig",
    "HybridQuantumProtoNet",
]
