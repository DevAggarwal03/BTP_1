import torch

from qpn_hybrid.prototypes import prototype_fidelity_scores


def test_same_prototype_objective_is_used_for_repeated_calls():
    states = torch.eye(8, dtype=torch.complex128)
    support_labels = torch.tensor([0, 0, 1, 1])
    query = states[[0, 2, 4]]
    first = prototype_fidelity_scores(query, states[:4], support_labels)
    second = prototype_fidelity_scores(query, states[:4], support_labels)
    assert torch.allclose(first, second)


def test_global_pairwise_average_matches_global_mixture_fidelity():
    states = torch.eye(4, dtype=torch.complex128)[:3]
    mixture = states.transpose(0, 1) @ states.conj() / states.shape[0]
    mixture_scores = torch.einsum("bi,ij,bj->b", states.conj(), mixture, states).real
    pairwise_average = states.conj() @ states.transpose(0, 1)
    pairwise_average = pairwise_average.abs().square().mean(dim=1)
    assert torch.allclose(mixture_scores, pairwise_average)
