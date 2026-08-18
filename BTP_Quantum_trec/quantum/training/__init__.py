"""
quantum/training/__init__.py
Module for the overarching Quantum Prototypical Network model and its Training loop.
"""
from .qpn_model import QuantumProtoNet
from .trainer import MetaLearningTrainer
from .outer_loop import QPNMasterTrainer

__all__ = ["QuantumProtoNet", "MetaLearningTrainer", "QPNMasterTrainer"]
