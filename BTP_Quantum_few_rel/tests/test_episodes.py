# This test module checks that episode generation and remapping behave correctly.
# It validates the shapes and labels produced by the sampler under different configurations.
import pytest
import numpy as np
from qpn.config import EpisodeConfig, set_seed
from qpn.episodes import EpisodeSampler, Episode

def test_episode_sampler_shapes():
    # Mock data pool
    # 15 relations, each with 20 instances, embedding size 10
    pool = {f"R{i}": np.random.randn(20, 10) for i in range(15)}
    
    config = EpisodeConfig(n_way=5, k_shot=1, q_queries=5, n_episodes=2)
    sampler = EpisodeSampler(pool, config)
    episodes = sampler.sample()
    
    assert len(episodes) == 2
    
    ep = episodes[0]
    assert ep.support_x.shape == (5 * 1, 10)
    assert ep.support_y.shape == (5 * 1,)
    assert ep.query_x.shape == (5 * 5, 10)
    assert ep.query_y.shape == (5 * 5,)
    assert len(ep.classes) == 5
    
    # Check y values are 0..4
    assert set(ep.support_y) == {0, 1, 2, 3, 4}
    assert set(ep.query_y) == {0, 1, 2, 3, 4}

def test_sampler_determinism():
    pool = {f"R{i}": np.random.randn(20, 10) for i in range(10)}
    config = EpisodeConfig(n_way=5, k_shot=1, q_queries=5, n_episodes=1)
    
    set_seed(42)
    sampler1 = EpisodeSampler(pool, config)
    ep1 = sampler1.sample()[0]
    
    set_seed(42)
    sampler2 = EpisodeSampler(pool, config)
    ep2 = sampler2.sample()[0]
    
    assert ep1.classes == ep2.classes
    assert np.allclose(ep1.support_x, ep2.support_x)
    assert np.array_equal(ep1.support_y, ep2.support_y)
    assert np.allclose(ep1.query_x, ep2.query_x)
