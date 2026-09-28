"""Run trained classical ProtoNet baselines on the shared FewRel protocol."""

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

from qpn_hybrid.baselines import ClassicalProtoNet
from qpn_hybrid.config import (
    EpisodeConfig,
    ModelConfig,
    TrainConfig,
    dataclass_from_config,
    load_yaml_config,
)
from qpn_hybrid.data import (
    embedding_dataset_metadata,
    load_relation_pools,
    validate_disjoint_relation_sets,
    validate_relation_pools,
)
from qpn_hybrid.evaluation import evaluate_episodes, make_fixed_episodes
from qpn_hybrid.training import meta_train, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "eight_qubit.yaml")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--train-episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=600)
    parser.add_argument("--output-dim", type=int, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "results" / "reruns" / "baseline_checkpoints",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "reruns" / "classical_baselines_600_600.json",
    )
    return parser.parse_args()


def run_one_baseline(
    train_pools,
    eval_pools,
    episode_config: EpisodeConfig,
    seed: int,
    distance: str,
    output_dim: int,
    hidden_dim: int,
    compression_mode: str,
    train_episodes: int,
    eval_episodes: int,
    checkpoint_dir: Path,
) -> dict[str, float | int | str]:
    set_seed(seed)
    model = ClassicalProtoNet(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        distance=distance,
        compression_mode=compression_mode,
    )
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
        verbose=False,
    )
    validation_episodes = make_fixed_episodes(
        eval_pools,
        episode_config=episode_config,
        count=eval_episodes,
        seed=seed + 1,
    )
    evaluation = evaluate_episodes(model, validation_episodes)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{distance}_seed_{seed}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "distance": distance,
            "output_dim": output_dim,
            "compression_hidden_dim": hidden_dim,
            "compression_mode": compression_mode,
            "episode_config": asdict(episode_config),
            "train_config": asdict(train_config),
            "losses": losses,
            "validation_results": evaluation,
        },
        checkpoint_path,
    )
    return {
        "model": f"classical_{distance}",
        "distance": distance,
        "seed": seed,
        "output_dim": output_dim,
        "train_episodes": train_episodes,
        "validation_episodes": eval_episodes,
        "accuracy_mean": float(evaluation["accuracy_mean"]),
        "accuracy_95ci": float(evaluation["accuracy_95ci"]),
        "weighted_f1_mean": float(evaluation["weighted_f1_mean"]),
        "weighted_f1_95ci": float(evaluation["weighted_f1_95ci"]),
        "last_train_loss": float(losses[-1]),
        "checkpoint": str(checkpoint_path),
    }


def aggregate(records: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    summaries = []
    for model_name in sorted({str(record["model"]) for record in records}):
        selected = [record for record in records if record["model"] == model_name]
        accuracy = np.asarray([float(record["accuracy_mean"]) for record in selected])
        weighted_f1 = np.asarray([float(record["weighted_f1_mean"]) for record in selected])
        summaries.append(
            {
                "model": model_name,
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
    yaml_config = load_yaml_config(args.config)
    model_config = dataclass_from_config(ModelConfig, yaml_config)
    output_dim = args.output_dim if args.output_dim is not None else model_config.n_qubits
    episode_config = dataclass_from_config(EpisodeConfig, yaml_config)
    train_pools = load_relation_pools(args.data_dir, args.train_split)
    eval_pools = load_relation_pools(args.data_dir, args.eval_split)
    validate_relation_pools(train_pools)
    validate_relation_pools(eval_pools)
    validate_disjoint_relation_sets(train_pools, eval_pools)

    records: list[dict[str, float | int | str]] = []
    checkpoint_dir = args.checkpoint_dir
    distances = ["euclidean", "cosine"]
    total = len(distances) * len(args.seeds)
    completed = 0
    for distance in distances:
        for seed in args.seeds:
            completed += 1
            print(f"starting baseline {completed}/{total}: {distance}, seed={seed}")
            record = run_one_baseline(
                train_pools,
                eval_pools,
                episode_config,
                seed,
                distance,
                output_dim,
                model_config.compression_hidden_dim,
                model_config.compression_mode,
                args.train_episodes,
                args.eval_episodes,
                checkpoint_dir,
            )
            records.append(record)
            print(
                f"completed: accuracy={float(record['accuracy_mean']):.4f}, "
                f"weighted_f1={float(record['weighted_f1_mean']):.4f}"
            )

    output = {
        "seeds": args.seeds,
        "distances": distances,
        "output_dim": output_dim,
        "compression_hidden_dim": model_config.compression_hidden_dim,
        "compression_mode": model_config.compression_mode,
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
        "records": records,
        "aggregates": aggregate(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["aggregates"], indent=2))


if __name__ == "__main__":
    main()
