from qpn_hybrid.config import EpisodeConfig


def test_benchmark_episode_protocol_is_explicit():
    config = EpisodeConfig(n_way=5, k_shot=1, q_queries=15)
    assert config.support_size == 5
    assert config.query_size == 75
    assert config.total_size == 80

