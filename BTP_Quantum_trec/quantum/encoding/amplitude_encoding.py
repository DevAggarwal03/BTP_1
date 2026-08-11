"""
amplitude_encoding.py
Amplitude embedding of a classical feature vector into a quantum state.

Encoding scheme:
    Given x ∈ R^n, prepare the quantum state:
        |ψ⟩ = (1/‖x‖) Σ_{i=0}^{n-1} x_i |i⟩

    This requires n_qubits = ⌈log₂(n)⌉ qubits and encodes n values
    as amplitudes of a 2^n_qubits dimensional state.

Uses Qiskit's ``StatePreparation`` (formerly ``Initialize``) for
exact state preparation via a sequence of uniformly controlled rotations.

Note: StatePreparation has exponential circuit depth in the worst case.
For NISQ-scale use, keep n_features ≤ 16 (n_qubits ≤ 4).
"""
from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation


class AmplitudeEncoder:
    """
    Amplitude encoding: embeds ``x ∈ R^n_features`` as quantum state amplitudes
    in a ``⌈log₂(n_features)⌉``-qubit system.

    The input vector is:
        1. Zero-padded to the next power of 2.
        2. L2-normalized (amplitudes must define a unit quantum state).

    Args:
        n_features: Dimensionality of the input feature vector.
    """

    def __init__(self, n_features: int) -> None:
        if n_features < 1:
            raise ValueError("n_features must be at least 1.")
        self.n_features = n_features
        self.n_qubits = math.ceil(math.log2(max(n_features, 2)))
        self.state_dim = 2 ** self.n_qubits  # padded state space size

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def encode(self, x: np.ndarray) -> QuantumCircuit:
        """
        Encode feature vector ``x`` as quantum amplitudes.

        Args:
            x: 1-D numpy array of length ``n_features``.
               Need not be normalized — L2 normalization is applied internally.

        Returns:
            QuantumCircuit that prepares the state |ψ⟩ ∝ Σ x_i |i⟩.

        Raises:
            ValueError: If ``x`` has wrong length.
        """
        x = np.asarray(x, dtype=complex)
        if x.ndim != 1 or len(x) != self.n_features:
            raise ValueError(
                f"Expected 1-D feature vector of length {self.n_features}, got shape {x.shape}."
            )

        statevector = self._prepare_statevector(x)

        qc = QuantumCircuit(self.n_qubits, name="AmplitudeEncoding")
        state_prep = StatePreparation(statevector)
        qc.append(state_prep, range(self.n_qubits))

        return qc

    def encode_batch(self, X: np.ndarray) -> list[QuantumCircuit]:
        """
        Encode a batch of feature vectors.

        Args:
            X: 2-D numpy array of shape ``(N, n_features)``.

        Returns:
            List of N QuantumCircuits.
        """
        return [self.encode(x) for x in X]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_statevector(self, x: np.ndarray) -> np.ndarray:
        """
        Zero-pad and L2-normalize ``x`` to produce a valid quantum statevector.

        Returns:
            Complex numpy array of length ``state_dim`` with unit L2 norm.
        """
        padded = np.zeros(self.state_dim, dtype=complex)
        padded[: self.n_features] = x

        norm = np.linalg.norm(padded)
        if norm < 1e-12:
            # Degenerate zero vector — default to |0⟩
            padded[0] = 1.0
        else:
            padded /= norm

        return padded

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AmplitudeEncoder("
            f"n_features={self.n_features}, "
            f"n_qubits={self.n_qubits}, "
            f"state_dim={self.state_dim})"
        )
