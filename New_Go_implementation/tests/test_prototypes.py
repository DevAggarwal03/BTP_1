import torch

from qpn_hybrid.prototypes import prototype_fidelity_scores


def test_prototype_scores_have_one_column_per_class():
    states = torch.eye(4, dtype=torch.complex128)
    support_labels = torch.tensor([0, 1])
    query_states = states[[0, 1, 2]]
    scores = prototype_fidelity_scores(query_states, states[:2], support_labels)
    assert scores.shape == (3, 2)
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 1] > scores[1, 0]

