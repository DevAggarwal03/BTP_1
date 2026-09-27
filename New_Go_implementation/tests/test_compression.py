import torch

from qpn_hybrid.compression import CompressionNetwork


def test_compression_outputs_bounded_qubit_angles():
    network = CompressionNetwork(input_dim=384, hidden_dim=64, n_qubits=4)
    angles = network(torch.randn(3, 384))
    assert angles.shape == (3, 4)
    assert torch.all(angles <= torch.pi)
    assert torch.all(angles >= -torch.pi)

