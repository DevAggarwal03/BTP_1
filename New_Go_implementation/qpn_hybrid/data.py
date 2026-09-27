"""FewRel relation-pool loading.

The old exploratory project stores precomputed sentence embeddings as a
pickle mapping relation name -> list of vectors.  This module reads that
format without changing the old project.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np


RelationPools = Dict[str, np.ndarray]


def default_data_dir() -> Path:
    """Return the data directory used by the existing FewRel project."""

    return Path(__file__).resolve().parents[2] / "BTP_Quantum_few_rel" / "data"


def _find_pickle(data_dir: Path, split: str) -> Path:
    candidates = [
        data_dir / f"{split}_embeddings.pkl",
        data_dir / f"{split}_relation_embeddings.pkl",
        data_dir / f"{split}.pkl",
        data_dir / f"{split}_data.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(data_dir.glob(f"*{split}*.pkl"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find a pickle file for split={split!r} in {data_dir}. "
        "Pass --data-dir explicitly or prepare the FewRel embeddings first."
    )


def _find_npz(data_dir: Path, split: str) -> Path | None:
    candidates = [
        data_dir / f"{split}_embeddings.npz",
        data_dir / f"{split}.npz",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _normalise_mapping(raw: Mapping[str, Iterable[Iterable[float]]]) -> RelationPools:
    pools: RelationPools = {}
    for relation, vectors in raw.items():
        array = np.asarray(list(vectors), dtype=np.float32)
        if array.ndim != 2 or array.shape[0] == 0:
            continue
        pools[str(relation)] = array
    if not pools:
        raise ValueError("The embedding file did not contain any non-empty relation pools.")
    return pools


def load_relation_pools(
    data_dir: str | Path | None = None,
    split: str = "train",
) -> RelationPools:
    """Load relation -> embedding arrays from the existing pickle format."""

    directory = Path(data_dir) if data_dir is not None else default_data_dir()
    npz_path = _find_npz(directory, split)
    if npz_path is not None:
        loaded = np.load(npz_path, allow_pickle=True)
        if "embeddings" not in loaded or "labels" not in loaded:
            raise ValueError(f"Expected embeddings and labels arrays in {npz_path}.")
        pools: dict[str, list[np.ndarray]] = {}
        for vector, relation in zip(loaded["embeddings"], loaded["labels"]):
            pools.setdefault(str(relation), []).append(vector)
        return _normalise_mapping(pools)

    with _find_pickle(directory, split).open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("Expected the embedding pickle to contain a mapping of relations to vectors.")
    return _normalise_mapping(raw)


def validate_relation_pools(
    pools: Mapping[str, np.ndarray],
    embedding_dim: int = 384,
    minimum_examples: int = 16,
) -> None:
    """Fail early when the data cannot support the requested episodes."""

    if len(pools) < 5:
        raise ValueError(f"At least five relations are required, found {len(pools)}.")
    for relation, values in pools.items():
        if values.ndim != 2 or values.shape[1] != embedding_dim:
            raise ValueError(
                f"Relation {relation!r} has shape {values.shape}; expected (?, {embedding_dim})."
            )
        if values.shape[0] < minimum_examples:
            raise ValueError(
                f"Relation {relation!r} has {values.shape[0]} examples; "
                f"need at least {minimum_examples} for a 1-shot/15-query episode."
            )


def validate_disjoint_relation_sets(
    train_pools: Mapping[str, np.ndarray],
    validation_pools: Mapping[str, np.ndarray],
) -> None:
    """Prevent relation leakage between train and validation splits."""

    overlap = set(train_pools).intersection(validation_pools)
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValueError(f"Train and validation relations overlap: {preview}")
