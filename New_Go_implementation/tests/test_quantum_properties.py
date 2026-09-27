import torch

from qpn_hybrid.fidelity import density_matrix, state_to_density_fidelity
from qpn_hybrid.quantum_encoder import DataReuploadingQuantumEncoder


def test_density_matrix_has_trace_one_and_is_positive_semidefinite():
    states = DataReuploadingQuantumEncoder(4, 1)(torch.randn(3, 4))
    prototype = density_matrix(states)
    eigenvalues = torch.linalg.eigvalsh(prototype).real
    assert torch.allclose(torch.trace(prototype).real, torch.tensor(1.0, dtype=torch.float64))
    assert torch.all(eigenvalues >= -1e-10)


def test_one_shot_density_prototype_is_pure():
    states = DataReuploadingQuantumEncoder(4, 1)(torch.randn(1, 4))
    prototype = density_matrix(states)
    assert torch.allclose(torch.trace(prototype @ prototype).real, torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(state_to_density_fidelity(states, prototype), torch.ones(1, dtype=torch.float64))
