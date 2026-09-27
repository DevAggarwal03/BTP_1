import torch

from qpn_hybrid.fidelity import density_matrix, pure_state_fidelity, state_to_density_fidelity


def test_pure_state_and_density_fidelity_are_in_range():
    states = torch.eye(4, dtype=torch.complex128)
    values = pure_state_fidelity(states[:2], states[1:3])
    assert torch.all(values >= 0)
    assert torch.all(values <= 1)
    prototype = density_matrix(states[:2])
    scores = state_to_density_fidelity(states[:2], prototype)
    assert torch.allclose(scores, torch.tensor([0.5, 0.5], dtype=torch.float64))

