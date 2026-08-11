# This script launches a small interactive demonstration of the few-shot pipeline.
# It wires together configuration, episode generation, and baseline evaluation for a quick run.
import os
import numpy as np
from qpn.config import STANDARD_SETTINGS, set_seed
from qpn.episodes import EpisodeSampler
from qpn.baselines import UntrainedProtoNet, get_classical_ml_baselines
from qpn.eval.harness import evaluate
from qpn.data import get_relation_pools

def main():
    set_seed(42)
    
    print("Loading data...")
    try:
        train_pool, val_pool = get_relation_pools()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run scripts.download_fewrel and scripts.build_embeddings first.")
        # Create a mock pool for demo purposes if real data is missing
        print("Using mock data for demo...")
        val_pool = {f"R{i}": np.random.randn(20, 384) for i in range(10)}

    config = STANDARD_SETTINGS["5w1s"]
    config.n_episodes = 10 # Small number for demo
    
    sampler = EpisodeSampler(val_pool, config)
    
    print(f"\nEvaluating on {config.n_way}-way {config.k_shot}-shot ({config.n_episodes} episodes)")
    
    model = UntrainedProtoNet()
    res = evaluate(model, sampler)
    
    mean = res["accuracy"]["mean"]
    ci = res["accuracy"]["ci95"]
    print(f"Untrained ProtoNet Accuracy: {mean:.4f} ± {ci:.4f}")
    
    baselines = get_classical_ml_baselines()
    for name, ml_model in baselines.items():
        # ScikitLearn classifiers need to be re-initialized for each episode inside evaluate wrapper
        # The wrapper handles it.
        # Wait, EpisodeSampler needs to sample same episodes.
        # We need to set the seed before each evaluate call to ensure same episodes are evaluated.
        
        set_seed(42)
        sampler = EpisodeSampler(val_pool, config)
        ml_res = evaluate(ml_model, sampler)
        
        ml_mean = ml_res["accuracy"]["mean"]
        ml_ci = ml_res["accuracy"]["ci95"]
        print(f"{name} Accuracy: {ml_mean:.4f} ± {ml_ci:.4f}")

if __name__ == "__main__":
    main()
