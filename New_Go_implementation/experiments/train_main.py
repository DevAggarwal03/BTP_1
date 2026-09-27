"""Run the first small end-to-end training/evaluation smoke test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qpn_hybrid.config import EpisodeConfig, ModelConfig, TrainConfig
from qpn_hybrid.data import (
    load_relation_pools,
    validate_disjoint_relation_sets,
    validate_relation_pools,
)
from qpn_hybrid.evaluation import evaluate_episodes, make_fixed_episodes
from qpn_hybrid.model import HybridQuantumProtoNet
from qpn_hybrid.training import meta_train, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results" / "smoke_checkpoint.pt",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_pools = load_relation_pools(args.data_dir, args.train_split)
    eval_pools = load_relation_pools(args.data_dir, args.eval_split)
    validate_relation_pools(train_pools)
    validate_relation_pools(eval_pools)
    validate_disjoint_relation_sets(train_pools, eval_pools)

    # Seed before constructing the model so its initial weights are reproducible too.
    set_seed(args.seed)
    episode_config = EpisodeConfig()
    model = HybridQuantumProtoNet(ModelConfig(n_qubits=args.n_qubits))
    train_config = TrainConfig(
        train_episodes=args.train_episodes,
        validation_episodes=args.eval_episodes,
        seed=args.seed,
    )
    losses = meta_train(
        model,
        train_pools,
        episode_config=episode_config,
        train_config=train_config,
        verbose=not args.quiet,
    )
    validation_episodes = make_fixed_episodes(
        eval_pools,
        episode_config=episode_config,
        count=args.eval_episodes,
        seed=args.seed + 1,
    )
    results = evaluate_episodes(model, validation_episodes)
    results["last_train_loss"] = losses[-1] if losses else None
    results["n_qubits"] = args.n_qubits
    results["train_episodes"] = args.train_episodes
    results["validation_episodes"] = args.eval_episodes
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model.config),
            "episode_config": asdict(episode_config),
            "train_config": asdict(train_config),
            "losses": losses,
            "validation_results": results,
        },
        args.checkpoint,
    )
    results["checkpoint"] = str(args.checkpoint)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
