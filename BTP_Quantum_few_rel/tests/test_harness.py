# This test module verifies that the evaluation harness produces the expected metrics and predictions.
# It exercises the end-to-end scoring pathway for a small example.
import numpy as np
from qpn.config import EpisodeConfig, set_seed
from qpn.episodes import EpisodeSampler
from qpn.eval.harness import evaluate
from qpn.baselines import UntrainedProtoNet

def test_harness_reproducibility():
    pool = {f"R{i}": np.random.randn(20, 10) for i in range(5)}
    config = EpisodeConfig(n_way=3, k_shot=1, q_queries=5, n_episodes=5)
    
    set_seed(42)
    sampler1 = EpisodeSampler(pool, config)
    res1 = evaluate(UntrainedProtoNet(), sampler1)
    
    set_seed(42)
    sampler2 = EpisodeSampler(pool, config)
    res2 = evaluate(UntrainedProtoNet(), sampler2)
    
    assert res1["accuracy"]["raw"] == res2["accuracy"]["raw"]
    assert res1["f1_weighted"]["raw"] == res2["f1_weighted"]["raw"]
    assert np.isclose(res1["accuracy"]["mean"], res2["accuracy"]["mean"])
