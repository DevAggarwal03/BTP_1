"""Exact differentiable statevector simulator and data-reuploading circuit.

This is intentionally a small simulator implemented directly with PyTorch.
It represents every circuit state as a length-2**n complex statevector, so it
is exact up to floating-point arithmetic and supports autograd through both
input angles and trainable gate angles.
"""

from __future__ import annotations

import torch
from torch import nn


def _rotation_matrix(angles: torch.Tensor, kind: str) -> torch.Tensor:
    """Create a batch of RY or RZ matrices with shape [batch, 2, 2]."""

    half = angles / 2.0
    zeros = torch.zeros_like(half)
    if kind == "ry":
        cos = torch.cos(half)
        sin = torch.sin(half)
        matrix = torch.stack(
            [
                torch.stack([cos, -sin], dim=-1),
                torch.stack([sin, cos], dim=-1),
            ],
            dim=-2,
        )
    elif kind == "rz":
        phase = torch.complex(zeros, half)
        matrix = torch.zeros((*angles.shape, 2, 2), dtype=torch.complex128, device=angles.device)
        matrix[..., 0, 0] = torch.exp(-phase)
        matrix[..., 1, 1] = torch.exp(phase)
    else:
        raise ValueError(f"Unsupported rotation kind: {kind}")
    return matrix.to(torch.complex128)


def _apply_batched_single_qubit_gate(
    states: torch.Tensor,
    gates: torch.Tensor,
    qubit: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply one possibly different 2x2 gate to each state in a batch."""

    batch_size = states.shape[0]
    tensor = states.reshape(batch_size, *([2] * n_qubits))
    axis = qubit + 1
    permutation = [0] + [index for index in range(1, n_qubits + 1) if index != axis] + [axis]
    moved = tensor.permute(permutation).reshape(batch_size, -1, 2)
    transformed = torch.einsum("bij,bmj->bmi", gates, moved)
    inverse = [permutation.index(index) for index in range(len(permutation))]
    return transformed.reshape([batch_size] + [2] * n_qubits).permute(inverse).reshape(batch_size, -1)


def _apply_cnot(states: torch.Tensor, control: int, target: int, n_qubits: int) -> torch.Tensor:
    """Apply CNOT using a fixed permutation of computational-basis amplitudes."""

    dimension = 2**n_qubits
    indices = torch.arange(dimension, device=states.device)
    control_bits = (indices >> (n_qubits - 1 - control)) & 1
    source_indices = indices ^ (control_bits << (n_qubits - 1 - target))
    return states.index_select(1, source_indices)


class DataReuploadingQuantumEncoder(nn.Module):
    """Angle encoding followed by trainable data re-uploading layers.

    The circuit begins with input-dependent RY rotations. Each trainable
    RY/RZ block and circular CNOT chain is followed by another input encoding.
    The final operation is therefore input-dependent, so no trainable block
    is hidden inside a common final unitary that cancels from fidelities.
    """

    def __init__(self, n_qubits: int = 4, reps: int = 2) -> None:
        super().__init__()
        if n_qubits < 2:
            raise ValueError("Use at least two qubits so the entangling layer is meaningful.")
        if reps < 1:
            raise ValueError("reps must be at least one so the circuit has trainable layers.")
        self.n_qubits = n_qubits
        self.reps = reps
        self.theta = nn.Parameter(torch.empty(reps, 2, n_qubits, dtype=torch.float64))
        nn.init.uniform_(self.theta, -0.05, 0.05)

    @property
    def state_dimension(self) -> int:
        return 2**self.n_qubits

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        if angles.ndim != 2 or angles.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected angles with shape [batch, {self.n_qubits}], got {tuple(angles.shape)}"
            )
        angles = angles.to(dtype=torch.float64)
        batch_size = angles.shape[0]
        states = torch.zeros(
            batch_size,
            self.state_dimension,
            dtype=torch.complex128,
            device=angles.device,
        )
        states[:, 0] = 1.0 + 0.0j

        # Initial angle encoding, followed by data re-uploading after every
        # trainable block. In particular, keep the final re-upload: a shared
        # trainable unitary after the last input encoding would leave all
        # support/query fidelities invariant to that final block.
        for qubit in range(self.n_qubits):
            input_gate = _rotation_matrix(angles[:, qubit], "ry")
            states = _apply_batched_single_qubit_gate(states, input_gate, qubit, self.n_qubits)

        for repetition in range(self.reps):
            for qubit in range(self.n_qubits):
                trainable_ry = _rotation_matrix(
                    self.theta[repetition, 0, qubit].expand(batch_size), "ry"
                )
                trainable_rz = _rotation_matrix(
                    self.theta[repetition, 1, qubit].expand(batch_size), "rz"
                )
                states = _apply_batched_single_qubit_gate(
                    states, trainable_ry, qubit, self.n_qubits
                )
                states = _apply_batched_single_qubit_gate(
                    states, trainable_rz, qubit, self.n_qubits
                )

            for qubit in range(self.n_qubits):
                states = _apply_cnot(states, qubit, (qubit + 1) % self.n_qubits, self.n_qubits)

            for qubit in range(self.n_qubits):
                input_gate = _rotation_matrix(angles[:, qubit], "ry")
                states = _apply_batched_single_qubit_gate(states, input_gate, qubit, self.n_qubits)

        norm = states.abs().square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
        return states / norm
