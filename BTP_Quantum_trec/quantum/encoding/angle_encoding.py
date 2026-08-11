"""
angle_encoding.py
Map classical normalized feature vectors → qubit rotations (RY gates).

Encoding scheme:
    Feature x_i ∈ [0, 1]  →  RY(π * x_i) on qubit i

This is the standard angle (basis) encoding used for quantum machine learning.
Each qubit encodes exactly one feature value.  The encoded state for qubit i is:

    |ψ_i⟩ = cos(π x_i / 2)|0⟩ + sin(π x_i / 2)|1⟩

The circuit depth is O(n_qubits) — efficient for NISQ devices.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class AngleEncoder:
    """
    Angle encoding: embeds a feature vector ``x ∈ [0, 1]^n`` into ``n`` qubits
    using single-qubit RY rotations.

    Args:
        n_qubits: Number of qubits (must equal the feature vector length).
    """

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be at least 1.")
        self.n_qubits = n_qubits

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def encode(self, x: np.ndarray) -> QuantumCircuit:
        """
        Build and return a Qiskit circuit that angle-encodes feature vector ``x``.

        Args:
            x: 1-D numpy array of length ``n_qubits`` with values in ``[0, 1]``.
               Values are clipped to ``[0, 1]`` before encoding.

        Returns:
            QuantumCircuit of depth 1 with RY(π * x[i]) on qubit i.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) != self.n_qubits:
            raise ValueError(
                f"Expected 1-D feature vector of length {self.n_qubits}, got shape {x.shape}."
            )

        x = np.clip(x, 0.0, 1.0)
        qc = QuantumCircuit(self.n_qubits, name="AngleEncoding")

        for i, val in enumerate(x):
            qc.ry(float(np.pi * val), i)

        return qc

    def encode_batch(self, X: np.ndarray) -> list[QuantumCircuit]:
        """
        Encode a batch of feature vectors.

        Args:
            X: 2-D numpy array of shape ``(N, n_qubits)``.

        Returns:
            List of N QuantumCircuits.
        """
        return [self.encode(x) for x in X]

    def build_parameterized(self) -> tuple[QuantumCircuit, ParameterVector]:
        """
        Build a *parameterized* angle encoding circuit suitable for gradient-based
        VQC optimization (used in Phase 2 — QNN training).

        Returns
        -------
        circuit : QuantumCircuit
            Parameterized circuit with RY(π * x[i]) gates.
        params : ParameterVector
            Parameter vector ``x`` of length ``n_qubits``.
        """
        params = ParameterVector("x", self.n_qubits)
        qc = QuantumCircuit(self.n_qubits, name="AngleEncoding_Param")

        for i, p in enumerate(params):
            qc.ry(np.pi * p, i)

        return qc, params

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"AngleEncoder(n_qubits={self.n_qubits})"
