"""Run the main model across several seeds and aggregate the results."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
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
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--qubits", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--train-episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=600)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "results" / "reruns" / "multi_seed_checkpoints",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "reruns" / "multi_seed_summary.json",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def run_one_setting(
    train_pools,
    eval_pools,
    episode_config: EpisodeConfig,
    seed: int,
    model_config: ModelConfig,
    train_episodes: int,
    eval_episodes: int,
    checkpoint_dir: Path,
    quiet: bool,
) -> dict[str, float | int | str]:
    # Seed before model construction so initial weights, episodes, and validation
    # sampling are all controlled by the same seed.
    set_seed(seed)
    model = HybridQuantumProtoNet(model_config)
    train_config = TrainConfig(
        train_episodes=train_episodes,
        validation_episodes=eval_episodes,
        seed=seed,
    )
    losses = meta_train(
        model,
        train_pools,
        episode_config=episode_config,
        train_config=train_config,
        verbose=not quiet,
    )
    validation_episodes = make_fixed_episodes(
        eval_pools,
        episode_config=episode_config,
        count=eval_episodes,
        seed=seed + 1,
    )
    evaluation = evaluate_episodes(model, validation_episodes)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_config.n_qubits}qubit_seed_{seed}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_config),
            "episode_config": asdict(episode_config),
            "quantum_simulation": quantum_simulation_metadata(model_config),
            "train_config": asdict(train_config),
            "losses": losses,
            "validation_results": evaluation,
        },
        checkpoint_path,
    )
    return {
        "seed": seed,
        "n_qubits": model_config.n_qubits,
        "train_episodes": train_episodes,
        "validation_episodes": eval_episodes,
        "accuracy_mean": float(evaluation["accuracy_mean"]),
        "accuracy_95ci": float(evaluation["accuracy_95ci"]),
        "weighted_f1_mean": float(evaluation["weighted_f1_mean"]),
        "weighted_f1_95ci": float(evaluation["weighted_f1_95ci"]),
        "last_train_loss": float(losses[-1]),
        "checkpoint": str(checkpoint_path),
    }


def aggregate(records: list[dict[str, float | int | str]]) -> list[dict[str, float | int]]:
    summaries = []
    for n_qubits in sorted({int(record["n_qubits"]) for record in records}):
        selected = [record for record in records if int(record["n_qubits"]) == n_qubits]
        accuracy = np.asarray([float(record["accuracy_mean"]) for record in selected])
        weighted_f1 = np.asarray([float(record["weighted_f1_mean"]) for record in selected])
        summaries.append(
            {
                "n_qubits": n_qubits,
                "seeds": [int(record["seed"]) for record in selected],
                "accuracy_mean_across_seeds": float(accuracy.mean()),
                "accuracy_std_across_seeds": float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0,
                "weighted_f1_mean_across_seeds": float(weighted_f1.mean()),
                "weighted_f1_std_across_seeds": float(weighted_f1.std(ddof=1)) if len(weighted_f1) > 1 else 0.0,
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    train_pools = load_relation_pools(args.data_dir, args.train_split)
    eval_pools = load_relation_pools(args.data_dir, args.eval_split)
    validate_relation_pools(train_pools)
    validate_relation_pools(eval_pools)
    validate_disjoint_relation_sets(train_pools, eval_pools)

    config_stems = {4: "four_qubit", 8: "eight_qubit"}
    unsupported = sorted(set(args.qubits) - set(config_stems))
    if unsupported:
        raise ValueError(f"No YAML model config is defined for qubits={unsupported}.")
    config_files = {
        n_qubits: load_yaml_config(args.config_dir / f"{config_stems[n_qubits]}.yaml")
        for n_qubits in args.qubits
    }
    model_configs = {
        n_qubits: dataclass_from_config(
            ModelConfig,
            {**config_files[n_qubits], "n_qubits": n_qubits},
        )
        for n_qubits in args.qubits
    }
    episode_config = dataclass_from_config(
        EpisodeConfig,
        config_files[args.qubits[0]],
    )
    records: list[dict[str, float | int | str]] = []
    checkpoint_dir = args.checkpoint_dir
    total = len(args.qubits) * len(args.seeds)
    completed = 0
    for n_qubits in args.qubits:
        for seed in args.seeds:
            completed += 1
            print(f"starting setting {completed}/{total}: {n_qubits} qubits, seed={seed}")
            record = run_one_setting(
                train_pools,
                eval_pools,
                episode_config,
                seed,
                model_configs[n_qubits],
                args.train_episodes,
                args.eval_episodes,
                checkpoint_dir,
                args.quiet,
            )
            records.append(record)
            print(
                f"completed: accuracy={float(record['accuracy_mean']):.4f}, "
                f"weighted_f1={float(record['weighted_f1_mean']):.4f}"
            )

    output = {
        "seeds": args.seeds,
        "qubits": args.qubits,
        "train_episodes": args.train_episodes,
        "validation_episodes": args.eval_episodes,
        "optimizer": {
            "name": "Adam",
            "learning_rate": TrainConfig().learning_rate,
            "weight_decay": TrainConfig().weight_decay,
        },
        "episode_config": asdict(episode_config),
        "data": embedding_dataset_metadata(
            args.data_dir, args.train_split, args.eval_split, train_pools, eval_pools
        ),
        "model_configs": {str(key): asdict(value) for key, value in model_configs.items()},
        "quantum_simulation_by_qubit": {
            str(key): quantum_simulation_metadata(value)
            for key, value in model_configs.items()
        },
        "records": records,
        "aggregates": aggregate(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["aggregates"], indent=2))


if __name__ == "__main__":
    main()
