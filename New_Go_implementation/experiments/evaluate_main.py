"""Evaluate a saved model on a newly sampled validation episode list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qpn_hybrid.config import EpisodeConfig, ModelConfig
from qpn_hybrid.data import load_relation_pools
from qpn_hybrid.evaluation import evaluate_episodes, make_fixed_episodes
from qpn_hybrid.model import HybridQuantumProtoNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = HybridQuantumProtoNet(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    pools = load_relation_pools(args.data_dir, args.split)
    episodes = make_fixed_episodes(pools, EpisodeConfig(**checkpoint["episode_config"]), args.episodes, args.seed)
    print(json.dumps(evaluate_episodes(model, episodes), indent=2))


if __name__ == "__main__":
    main()
