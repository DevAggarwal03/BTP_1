import torch

from qpn_hybrid.quantum_encoder import DataReuploadingQuantumEncoder


def test_quantum_gradient_agrees_with_finite_difference_away_from_zero():
    encoder = DataReuploadingQuantumEncoder(3, 1)
    with torch.no_grad():
        encoder.theta.fill_(0.17)
    angles = torch.tensor([[0.31, -0.27, 0.19]], dtype=torch.float64)

    def objective() -> torch.Tensor:
        state = encoder(angles)
        return state[0, 1].real.square() + state[0, 2].imag.square()

    analytic = torch.autograd.grad(objective(), encoder.theta)[0][0, 0, 0].item()
    epsilon = 1e-5
    with torch.no_grad():
        encoder.theta[0, 0, 0] += epsilon
    plus = objective().item()
    with torch.no_grad():
        encoder.theta[0, 0, 0] -= 2 * epsilon
    minus = objective().item()
    with torch.no_grad():
        encoder.theta[0, 0, 0] += epsilon
    finite_difference = (plus - minus) / (2 * epsilon)
    assert abs(analytic - finite_difference) < 1e-5
