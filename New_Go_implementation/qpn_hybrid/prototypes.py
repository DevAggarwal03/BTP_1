"""Prototype construction and prototype-based class scores."""

from __future__ import annotations

import torch

from .fidelity import density_matrix, state_to_density_fidelity


def class_density_matrices(
    support_states: torch.Tensor,
    support_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    prototypes = []
    for class_id in range(num_classes):
        members = support_states[support_labels == class_id]
        if members.shape[0] == 0:
            raise ValueError(f"No support example found for local class {class_id}.")
        prototypes.append(density_matrix(members))
    return torch.stack(prototypes, dim=0)


def prototype_fidelity_scores(
    query_states: torch.Tensor,
    support_states: torch.Tensor,
    support_labels: torch.Tensor,
    num_classes: int | None = None,
) -> torch.Tensor:
    if num_classes is None:
        num_classes = int(support_labels.max().item()) + 1
    prototypes = class_density_matrices(support_states, support_labels, num_classes)
    return torch.stack(
        [state_to_density_fidelity(query_states, prototype) for prototype in prototypes],
        dim=1,
    )

