import numpy as np
import torch

from qpn_hybrid.baselines import ClassicalProtoNet, FixedQuantumProtoNet
from qpn_hybrid.episodes import episode_from_arrays
from qpn_hybrid.training import train_one_episode


def _episode():
    rng = np.random.default_rng(2)
    return episode_from_arrays(
        rng.normal(size=(5, 384)).astype(np.float32),
        np.arange(5),
        rng.normal(size=(10, 384)).astype(np.float32),
        np.repeat(np.arange(5), 2),
    )


def test_classical_baselines_accept_the_same_episode_input():
    episode = _episode()
    model = ClassicalProtoNet(distance="cosine")
    logits = model(
        torch.tensor(episode.support_x),
        torch.tensor(episode.support_y),
        torch.tensor(episode.query_x),
    )
    assert logits.shape == (10, 5)


def test_trainer_can_train_classical_baseline():
    model = ClassicalProtoNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_episode(model, _episode(), optimizer)
    assert np.isfinite(loss)


def test_fixed_quantum_baseline_has_no_trainable_parameters():
    model = FixedQuantumProtoNet()
    assert not any(parameter.requires_grad for parameter in model.parameters())
