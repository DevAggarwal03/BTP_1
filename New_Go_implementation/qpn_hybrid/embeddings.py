"""Frozen Sentence-BERT embedding utilities.

The model is loaded lazily so tests and experiments using cached embeddings do
not download or initialise a language model unnecessarily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


class FrozenSentenceEmbedder:
    """Create fixed `all-MiniLM-L6-v2` sentence embeddings."""

    model_name = "all-MiniLM-L6-v2"
    output_dim = 384

    def __init__(self, model_name: str = model_name, cache_folder: str | Path | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - depends on optional install
            raise ImportError(
                "sentence-transformers is required to create embeddings; "
                "use the cached .npz files for tests if it is unavailable."
            ) from error
        kwargs = {} if cache_folder is None else {"cache_folder": str(cache_folder)}
        self.model = SentenceTransformer(model_name, **kwargs)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, sentences: Iterable[str], batch_size: int = 32) -> np.ndarray:
        values = self.model.encode(
            list(sentences),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.output_dim:
            raise ValueError(f"Expected [batch, 384] embeddings, got {values.shape}")
        return values

