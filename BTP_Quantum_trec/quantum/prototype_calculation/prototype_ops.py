"""
prototype_ops.py
Quantum Prototype Calculator for the Quantum Prototypical Network.

Computes the mixed-state Density Matrix class prototype by averaging the
pure-state density matrices of all support set examples for a given class:

    ρ_k = (1 / K) Σᵢ DensityMatrix(|ψᵢ⟩)

Uses vectorized NumPy operations — avoids the O(4^n) element-wise Python loop
from the previous implementation.
"""
from __future__ import annotations

from typing import List, Union

import numpy as np
from qiskit.quantum_info import Statevector, DensityMatrix


class QuantumPrototypeCalculator:
    """
    Quantum Prototype Calculator (Few-shot class prototypes).

    Converts a list of pure quantum states (Statevectors from the VQC) into a
    single mixed-state Density Matrix representing the class prototype.

    Args:
        num_qubits: Number of qubits used in the VQC. Determines matrix size (2^n × 2^n).
    """

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        if num_qubits > 6:
            import warnings
            warnings.warn(
                f"Density matrix for {num_qubits} qubits is "
                f"{2**num_qubits}×{2**num_qubits} = {4**num_qubits} elements. "
                "Keep n_qubits ≤ 6 for reasonable memory usage (default is 4).",
                ResourceWarning,
                stacklevel=2,
            )

    def calculate_class_prototype(
        self,
        support_states: List[Union[Statevector, DensityMatrix]],
    ) -> DensityMatrix:
        """
        Calculate the mixed-state Density Matrix prototype for a class.

        Stacks all support-set density matrices into a (K, 2^n, 2^n) array
        and takes the vectorized mean along axis 0 — no Python element loops.

        Args:
            support_states: List of Qiskit Statevector or DensityMatrix objects.

        Returns:
            DensityMatrix: The averaged mixed state ρ_k.

        Raises:
            ValueError: If support_states is empty.
        """
        if not support_states:
            raise ValueError("support_states cannot be empty.")

        # Stack all density matrices: shape (K, dim, dim), then mean over K axis.
        dm_data = np.stack(
            [DensityMatrix(state).data for state in support_states],
            axis=0,
        )
        return DensityMatrix(dm_data.mean(axis=0))
