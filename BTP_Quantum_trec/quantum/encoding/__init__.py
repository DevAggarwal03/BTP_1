"""
quantum/encoding/__init__.py
Quantum encoding strategies: angle encoding and amplitude encoding.
"""
from .angle_encoding import AngleEncoder
from .amplitude_encoding import AmplitudeEncoder

__all__ = ["AngleEncoder", "AmplitudeEncoder"]
