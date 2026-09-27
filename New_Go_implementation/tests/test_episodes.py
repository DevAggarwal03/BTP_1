import numpy as np

from qpn_hybrid.config import EpisodeConfig
from qpn_hybrid.episodes import EpisodeSampler


def test_5w1s_episode_has_expected_size():
    pools = {f"r{i}": np.random.randn(20, 384).astype(np.float32) for i in range(8)}
    episode = EpisodeSampler(pools, EpisodeConfig(), seed=3).sample()
    assert episode.support_x.shape == (5, 384)
    assert episode.query_x.shape == (75, 384)
    assert len(episode.classes) == 5
    assert np.bincount(episode.query_y).tolist() == [15] * 5

