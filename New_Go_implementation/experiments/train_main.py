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

from qpn_hybrid.config import (
    EpisodeConfig,
    ModelConfig,
    TrainConfig,
    dataclass_from_config,
    load_yaml_config,
    quantum_simulation_metadata,
)
from qpn_hybrid.data import (
    embedding_dataset_metadata,
    load_relation_pools,
    validate_disjoint_relation_sets,
    validate_relation_pools,
)
from qpn_hybrid.evaluation import evaluate_episodes, make_fixed_episodes
from qpn_hybrid.model import HybridQuantumProtoNet
from qpn_hybrid.training import meta_train, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "four_qubit.yaml")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results" / "reruns" / "smoke_checkpoint.pt",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yaml_config = load_yaml_config(args.config)
    train_pools = load_relation_pools(args.data_dir, args.train_split)
    eval_pools = load_relation_pools(args.data_dir, args.eval_split)
    validate_relation_pools(train_pools)
    validate_relation_pools(eval_pools)
    validate_disjoint_relation_sets(train_pools, eval_pools)

    model_values = dict(yaml_config)
    if args.n_qubits is not None:
        model_values["n_qubits"] = args.n_qubits
    model_config = dataclass_from_config(ModelConfig, model_values)
    episode_config = dataclass_from_config(EpisodeConfig, yaml_config)
    train_values = dict(yaml_config)
    if args.train_episodes is not None:
        train_values["train_episodes"] = args.train_episodes
    if args.eval_episodes is not None:
        train_values["validation_episodes"] = args.eval_episodes
    if args.seed is not None:
        train_values["seed"] = args.seed
    train_config = dataclass_from_config(TrainConfig, train_values)
    # Seed before constructing the model so its initial weights are reproducible too.
    set_seed(train_config.seed)
    model = HybridQuantumProtoNet(model_config)
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
        count=train_config.validation_episodes,
        seed=train_config.seed + 1,
    )
    results = evaluate_episodes(model, validation_episodes)
    results["last_train_loss"] = losses[-1] if losses else None
    results["n_qubits"] = model_config.n_qubits
    results["model_config"] = asdict(model_config)
    results["quantum_simulation"] = quantum_simulation_metadata(model_config)
    results["data"] = embedding_dataset_metadata(
        args.data_dir, args.train_split, args.eval_split, train_pools, eval_pools
    )
    results["train_episodes"] = train_config.train_episodes
    results["validation_episodes"] = train_config.validation_episodes
    results["episode_config"] = asdict(episode_config)
    results["train_config"] = asdict(train_config)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model.config),
            "episode_config": asdict(episode_config),
            "train_config": asdict(train_config),
            "quantum_simulation": quantum_simulation_metadata(model_config),
            "losses": losses,
            "validation_results": results,
        },
        args.checkpoint,
    )
    results["checkpoint"] = str(args.checkpoint)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
