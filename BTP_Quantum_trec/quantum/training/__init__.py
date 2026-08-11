"""
quantum/training/__init__.py
Module for the overarching Quantum Prototypical Network model and its Training loop.
"""
from .qpn_model import QuantumPrototypicalNetwork
from .trainer import MetaLearningTrainer
from .outer_loop import QPNMasterTrainer

__all__ = ["QuantumPrototypicalNetwork", "MetaLearningTrainer", "QPNMasterTrainer"]
