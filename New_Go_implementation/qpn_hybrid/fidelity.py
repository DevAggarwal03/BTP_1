"""Fidelity calculations used consistently in training and evaluation."""

from __future__ import annotations

import torch


def pure_state_fidelity(states_a: torch.Tensor, states_b: torch.Tensor) -> torch.Tensor:
    """Return pairwise pure-state fidelities between two state batches."""

    overlaps = states_a.conj() @ states_b.transpose(-2, -1)
    return overlaps.abs().square().real


def density_matrix(states: torch.Tensor) -> torch.Tensor:
    """Build the average density matrix for a batch of statevectors."""

    if states.ndim != 2:
        raise ValueError("states must have shape [examples, state_dimension]")
    return states.transpose(0, 1) @ states.conj() / states.shape[0]


def state_to_density_fidelity(states: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
    """Compute <psi|rho|psi> for every state and one density prototype."""

    values = torch.einsum("bi,ij,bj->b", states.conj(), prototype, states)
    return values.real.clamp(0.0, 1.0)


def validate_state_norms(states: torch.Tensor, atol: float = 1e-5) -> bool:
    norms = states.abs().square().sum(dim=-1)
    return bool(torch.allclose(norms, torch.ones_like(norms), atol=atol, rtol=atol))

