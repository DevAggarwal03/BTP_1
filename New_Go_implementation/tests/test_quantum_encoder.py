import torch

from qpn_hybrid.fidelity import validate_state_norms
from qpn_hybrid.quantum_encoder import DataReuploadingQuantumEncoder


def test_exact_statevector_is_normalised_and_input_dependent():
    encoder = DataReuploadingQuantumEncoder(n_qubits=4, reps=2)
    first = encoder(torch.zeros(1, 4))
    second = encoder(torch.full((1, 4), 0.4))
    assert first.shape == (1, 16)
    assert validate_state_norms(first)
    assert validate_state_norms(second)
    assert not torch.allclose(first, second)


def test_quantum_parameters_receive_gradient():
    encoder = DataReuploadingQuantumEncoder(n_qubits=4, reps=2)
    angles = torch.randn(5, 4, requires_grad=True)
    states = encoder(angles)
    loss = states.real.square().sum() + states.imag.square().sum()
    loss.backward()
    assert encoder.theta.grad is not None
    assert torch.isfinite(encoder.theta.grad).all()

