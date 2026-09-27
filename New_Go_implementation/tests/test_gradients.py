import numpy as np
import torch

from qpn_hybrid.episodes import episode_from_arrays
from qpn_hybrid.model import HybridQuantumProtoNet


def test_end_to_end_loss_reaches_compression_and_quantum_parameters():
    rng = np.random.default_rng(11)
    support_x = rng.normal(size=(5, 384)).astype(np.float32)
    query_x = rng.normal(size=(10, 384)).astype(np.float32)
    episode = episode_from_arrays(
        support_x,
        np.arange(5),
        query_x,
        np.repeat(np.arange(5), 2),
    )
    model = HybridQuantumProtoNet()
    support_x_t = torch.tensor(episode.support_x)
    support_y_t = torch.tensor(episode.support_y)
    query_x_t = torch.tensor(episode.query_x)
    query_y_t = torch.tensor(episode.query_y)
    loss, _, _ = model.episode_loss(support_x_t, support_y_t, query_x_t, query_y_t)
    loss.backward()
    assert model.compression.project[0].weight.grad is not None
    assert model.quantum.theta.grad is not None
    assert torch.isfinite(model.quantum.theta.grad).all()

