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
            raise ValueError(f"Relation {relation!r} must contain a non-empty 2D embedding array.")
        if not np.isfinite(array).all():
            raise ValueError(f"Relation {relation!r} contains NaN or infinite embedding values.")
        if pools and array.shape[1] != next(iter(pools.values())).shape[1]:
            raise ValueError(f"Relation {relation!r} has an inconsistent embedding dimension.")
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
        embeddings = np.asarray(loaded["embeddings"], dtype=np.float32)
        labels = np.asarray(loaded["labels"])
        if embeddings.ndim != 2 or labels.ndim != 1:
            raise ValueError(
                f"Expected a 2D embedding matrix and 1D labels in {npz_path}; "
                f"got {embeddings.shape} and {labels.shape}."
            )
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Embeddings/labels length mismatch in {npz_path}: "
                f"{embeddings.shape[0]} vs {labels.shape[0]}."
            )
        if not np.isfinite(embeddings).all():
            raise ValueError(f"Embeddings in {npz_path} contain NaN or infinite values.")
        pools: dict[str, list[np.ndarray]] = {}
        for vector, relation in zip(embeddings, labels):
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


def relation_pool_summary(pools: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Return compact split-size metadata for experiment result files."""

    counts = [int(values.shape[0]) for values in pools.values()]
    dimensions = sorted({int(values.shape[1]) for values in pools.values()})
    return {
        "relations": len(pools),
        "examples": int(sum(counts)),
        "embedding_dimensions": dimensions,
        "examples_per_relation_min": min(counts) if counts else 0,
        "examples_per_relation_max": max(counts) if counts else 0,
    }


def embedding_dataset_metadata(
    data_dir: str | Path | None,
    train_split: str,
    eval_split: str,
    train_pools: Mapping[str, np.ndarray],
    eval_pools: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Record the cached embedding source and split dimensions for a run."""

    directory = Path(data_dir) if data_dir is not None else default_data_dir()
    train_source = _find_npz(directory, train_split) or _find_pickle(directory, train_split)
    eval_source = _find_npz(directory, eval_split) or _find_pickle(directory, eval_split)
    return {
        "data_directory": str(directory),
        "source_files": {train_split: str(train_source), eval_split: str(eval_source)},
        "sentence_encoder": "sentence-transformers/all-MiniLM-L6-v2 (frozen cached embeddings)",
        "embeddings_normalized": False,
        "entity_markers": "[H]...[/H] and [T]...[/T] per Implementation Context",
        "splits": {
            train_split: relation_pool_summary(train_pools),
            eval_split: relation_pool_summary(eval_pools),
        },
    }
