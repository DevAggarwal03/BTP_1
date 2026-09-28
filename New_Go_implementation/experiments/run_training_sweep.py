"""Run longer-training and learning-rate experiments for the hybrid model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "eight_qubit.yaml")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--train-episodes", nargs="+", type=int, default=[2000])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.0003, 0.001, 0.003])
    parser.add_argument("--eval-episodes", type=int, default=600)
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "results" / "reruns" / "training_sweep_checkpoints",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "reruns" / "training_sweep.json",
    )
    return parser.parse_args()


def run_one(
    train_pools,
    eval_pools,
    episode_config: EpisodeConfig,
    seed: int,
    model_config: ModelConfig,
    train_episodes: int,
    learning_rate: float,
    eval_episodes: int,
    checkpoint_dir: Path,
) -> dict[str, float | int | str]:
    set_seed(seed)
    model = HybridQuantumProtoNet(model_config)
    train_config = TrainConfig(
        learning_rate=learning_rate,
        train_episodes=train_episodes,
        validation_episodes=eval_episodes,
        seed=seed,
    )
    start = time.perf_counter()
    losses = meta_train(
        model,
        train_pools,
        episode_config=episode_config,
        train_config=train_config,
        verbose=False,
    )
    validation_episodes = make_fixed_episodes(
        eval_pools,
        episode_config=episode_config,
        count=eval_episodes,
        seed=seed + 1,
    )
    evaluation = evaluate_episodes(model, validation_episodes)
    runtime_seconds = time.perf_counter() - start

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rate_name = f"{learning_rate:.6g}".replace(".", "p")
    checkpoint_path = checkpoint_dir / (
        f"{model_config.n_qubits}q_{train_episodes}ep_lr_{rate_name}_seed_{seed}.pt"
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_config),
            "quantum_simulation": quantum_simulation_metadata(model_config),
            "episode_config": asdict(episode_config),
            "train_config": asdict(train_config),
            "losses": losses,
            "validation_results": evaluation,
        },
        checkpoint_path,
    )
    return {
        "n_qubits": model_config.n_qubits,
        "seed": seed,
        "train_episodes": train_episodes,
        "validation_episodes": eval_episodes,
        "learning_rate": learning_rate,
        "accuracy_mean": float(evaluation["accuracy_mean"]),
        "accuracy_95ci": float(evaluation["accuracy_95ci"]),
        "weighted_f1_mean": float(evaluation["weighted_f1_mean"]),
        "weighted_f1_95ci": float(evaluation["weighted_f1_95ci"]),
        "initial_train_loss": float(losses[0]),
        "last_train_loss": float(losses[-1]),
        "minimum_train_loss": float(min(losses)),
        "runtime_seconds": runtime_seconds,
        "checkpoint": str(checkpoint_path),
    }


def main() -> None:
    args = parse_args()
    yaml_config = load_yaml_config(args.config)
    model_values = dict(yaml_config)
    if args.n_qubits is not None:
        model_values["n_qubits"] = args.n_qubits
    model_config = dataclass_from_config(ModelConfig, model_values)
    episode_config = dataclass_from_config(EpisodeConfig, yaml_config)
    train_pools = load_relation_pools(args.data_dir, args.train_split)
    eval_pools = load_relation_pools(args.data_dir, args.eval_split)
    validate_relation_pools(train_pools)
    validate_relation_pools(eval_pools)
    validate_disjoint_relation_sets(train_pools, eval_pools)

    checkpoint_dir = args.checkpoint_dir
    records: list[dict[str, float | int | str]] = []
    total = len(args.seeds) * len(args.train_episodes) * len(args.learning_rates)
    completed = 0
    for train_episodes in args.train_episodes:
        for learning_rate in args.learning_rates:
            for seed in args.seeds:
                completed += 1
                print(
                    f"starting setting {completed}/{total}: "
                    f"{train_episodes} episodes, lr={learning_rate}, seed={seed}"
                )
                record = run_one(
                    train_pools,
                    eval_pools,
                    episode_config,
                    seed,
                    model_config,
                    train_episodes,
                    learning_rate,
                    args.eval_episodes,
                    checkpoint_dir,
                )
                records.append(record)
                print(
                    f"completed: accuracy={float(record['accuracy_mean']):.4f}, "
                    f"weighted_f1={float(record['weighted_f1_mean']):.4f}, "
                    f"runtime={float(record['runtime_seconds']) / 60:.1f} min"
                )

    output = {
        "n_qubits": model_config.n_qubits,
        "model_config": asdict(model_config),
        "quantum_simulation": quantum_simulation_metadata(model_config),
        "seeds": args.seeds,
        "train_episodes": args.train_episodes,
        "learning_rates": args.learning_rates,
        "optimizer": {"name": "Adam", "weight_decay": TrainConfig().weight_decay},
        "validation_episodes": args.eval_episodes,
        "episode_config": asdict(episode_config),
        "data": embedding_dataset_metadata(
            args.data_dir, args.train_split, args.eval_split, train_pools, eval_pools
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
