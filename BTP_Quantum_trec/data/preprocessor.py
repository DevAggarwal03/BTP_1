"""
preprocessor.py
Text preprocessing pipeline for quantum encoding.

Pipeline stages:
    1. Sentence embedding    → dense feature matrix (N, embedding_dim)
                               via a pretrained SentenceTransformer model
    2. Min-Max normalization → values in [0, 1]
    3. LDA (supervised)      → (N, n_lda)  class-discriminative axes
    4. Re-normalization      → [0, 1] after LDA (LDA output is unbounded)

Why a pretrained encoder instead of TF-IDF?
    TF-IDF is a bag-of-words method — it captures word frequency but
    ignores word order, grammar, and semantics entirely. A pretrained
    sentence encoder (e.g. all-MiniLM-L6-v2) produces dense 384-dim
    vectors that encode *meaning*, enabling the downstream LDA to find
    more semantically meaningful discriminant directions for TREC-50.

Why LDA over PCA?
    PCA finds directions of maximum *variance* — ignoring class structure.
    LDA finds directions that maximally *separate classes*, making the output
    features directly discriminative. For TREC-50 (classification task), this
    is a better fit. With C=50 classes, LDA can produce up to C−1=49 components.
    We use n_lda=32, which is well within this limit.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler


class TRECPreprocessor:
    """
    Full prepr``jjocessing pipeline: text → quantum-ready feature matrix.

    Uses a pretrained SentenceTransformer for dense sentence embeddings,
    followed by **Linear Discriminant Analysis (LDA)** for supervised
    dimensionality reduction. LDA requires class labels during fitting
    (``fit_transform``) but not during inference (``transform``).

    Args:
        encoder_model:  HuggingFace model name or local path for the
                        SentenceTransformer encoder.
                        Defaults to ``"all-MiniLM-L6-v2"`` (384-dim,
                        Apache-2.0, ~90 MB, CPU-friendly).
        n_features_lda: Number of LDA discriminant components.
                        Must satisfy: n_features_lda ≤ min(n_classes − 1, embedding_dim).
                        For TREC-50 (50 classes) the ceiling is 49.
                        Set to ``None`` to skip LDA entirely.
        batch_size:     Batch size for the sentence encoder.
    """

    def __init__(
        self,
        encoder_model: str = "all-MiniLM-L6-v2",
        n_features_lda: Optional[int] = 32,
        batch_size: int = 64,
    ) -> None:
        self.encoder_model = encoder_model
        self.n_features_lda = n_features_lda
        self.batch_size = batch_size

        # Lazy-loaded on first use to avoid long import times when only
        # instantiating the class without encoding.
        self._encoder: Optional[SentenceTransformer] = None

        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self.lda: Optional[LinearDiscriminantAnalysis] = (
            LinearDiscriminantAnalysis(n_components=n_features_lda)
            if n_features_lda is not None
            else None
        )

        self._fitted = False
        self._embedding_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Core transform API
    # ------------------------------------------------------------------

    def fit_transform(self, texts: list[str], y: np.ndarray) -> np.ndarray:
        """
        Fit the full pipeline on *training* data and return transformed features.

        LDA is supervised and needs class labels at fit time.

        Args:
            texts: Raw text strings (training set).
            y:     Integer label array of shape ``(N,)``.
                   Required by LDA to compute class scatter matrices.

        Returns:
            X: ``np.ndarray`` of shape ``(N, n_output_features)`` in ``[0, 1]``.
        """
        y = np.asarray(y, dtype=int)

        # Stage 1 — Sentence embeddings  (N, embedding_dim)
        X = self._encode(texts)
        self._embedding_dim = X.shape[1]

        # Stage 2 — Min-Max normalize to [0, 1]
        X = self.scaler.fit_transform(X)                     # (N, embedding_dim)

        # Stage 3 — LDA: supervised dim-reduction (uses y)
        if self.lda is not None:
            X = self.lda.fit_transform(X, y)                 # (N, n_lda)
            X = self._clip_normalize(X)                      # back to [0, 1]

        self._fitted = True
        return X.astype(np.float32)

    def transform(self, texts: list[str]) -> np.ndarray:
        """
        Transform *unseen* texts using the already-fitted pipeline.

        Labels are **not** required at inference time — LDA.transform() applies
        the learned projection without needing y.

        Args:
            texts: Raw text strings (test / validation set).

        Returns:
            X: ``np.ndarray`` of shape ``(N, n_output_features)`` in ``[0, 1]``.

        Raises:
            RuntimeError: If ``fit_transform`` has not been called yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted. "
                "Call fit_transform(texts, y) on training data first."
            )

        X = self._encode(texts)
        X = self.scaler.transform(X)

        if self.lda is not None:
            X = self.lda.transform(X)
            X = self._clip_normalize(X)

        return X.astype(np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_output_features(self) -> int:
        """Dimensionality of the output feature vectors."""
        if self.n_features_lda is not None:
            return self.n_features_lda
        if self._embedding_dim is not None:
            return self._embedding_dim
        # Return encoder's native embedding dim before fitting
        return self._get_encoder().get_sentence_embedding_dimension()

    @property
    def embedding_dim(self) -> int:
        """Native embedding dimension of the sentence encoder."""
        return self._get_encoder().get_embedding_dimension()

    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        """
        LDA explained variance ratio — proportion of between-class variance
        captured by each discriminant axis.

        Returns ``None`` if LDA is disabled or not yet fitted.
        """
        if self.lda is None or not self._fitted:
            return None
        return getattr(self.lda, "explained_variance_ratio_", None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_encoder(self) -> SentenceTransformer:
        """Lazily load the SentenceTransformer model."""
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.encoder_model)
        return self._encoder

    def _encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into dense embedding vectors.

        Returns:
            ``np.ndarray`` of shape ``(N, embedding_dim)``, dtype float32.
        """
        encoder = self._get_encoder()
        embeddings = encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    @staticmethod
    def _clip_normalize(X: np.ndarray) -> np.ndarray:
        """
        Per-feature min-max rescale to [0, 1].
        Applied after LDA because discriminant projections are unbounded.
        """
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        denom = X_max - X_min
        denom[denom == 0.0] = 1.0          # constant features → set to 0
        return (X - X_min) / denom

    def __repr__(self) -> str:
        return (
            f"TRECPreprocessor("
            f"encoder_model='{self.encoder_model}', "
            f"n_features_lda={self.n_features_lda}, "
            f"fitted={self._fitted})"
        )
