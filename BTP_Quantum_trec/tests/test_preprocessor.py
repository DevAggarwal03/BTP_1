"""
test_preprocessor.py
Unit tests for the TRECPreprocessor class (LDA-based pipeline).

Note on LDA constraints in tests:
    LDA max components = n_classes − 1.
    We use 5 unique classes in SAMPLE_LABELS, so max n_features_lda = 4.
    Tests set n_features_lda=4 to stay within this limit.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessor import TRECPreprocessor


# 10 sample texts with 5 unique integer labels (2 samples per class)
SAMPLE_TEXTS = [
    "What is the capital of France?",
    "Who wrote Hamlet?",
    "How many planets are in the solar system?",
    "What country has the largest population?",
    "Who invented the telephone?",
    "When did World War II end?",
    "What is the speed of light?",
    "Where is the Eiffel Tower located?",
    "Who was the first US president?",
    "What year was Python created?",
]
# 5 classes, 2 samples each → LDA max n_components = 4
SAMPLE_LABELS = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])


class TestTRECPreprocessor:
    """Tests for TRECPreprocessor with LDA."""

    # ------------------------------------------------------------------
    # Output shape
    # ------------------------------------------------------------------

    def test_output_shape_with_lda(self):
        """fit_transform with LDA should return (N, n_lda) shaped array."""
        pp = TRECPreprocessor(n_features_tfidf=32, n_features_lda=4)
        X = pp.fit_transform(SAMPLE_TEXTS, SAMPLE_LABELS)
        assert X.shape == (len(SAMPLE_TEXTS), 4), f"Got {X.shape}"

    def test_output_shape_no_lda(self):
        """fit_transform without LDA should return (N, n_tfidf) shaped array."""
        pp = TRECPreprocessor(n_features_tfidf=32, n_features_lda=None)
        X = pp.fit_transform(SAMPLE_TEXTS, SAMPLE_LABELS)
        assert X.shape == (len(SAMPLE_TEXTS), 32), f"Got {X.shape}"

    # ------------------------------------------------------------------
    # Value range (angle encoding requires [0, 1])
    # ------------------------------------------------------------------

    def test_values_in_unit_range_with_lda(self):
        """All feature values after LDA must be in [0, 1]."""
        pp = TRECPreprocessor(n_features_tfidf=32, n_features_lda=4)
        X = pp.fit_transform(SAMPLE_TEXTS, SAMPLE_LABELS)
        assert X.min() >= -1e-6, f"Min value {X.min()} below 0"
        assert X.max() <= 1.0 + 1e-6, f"Max value {X.max()} above 1"

    def test_values_in_unit_range_no_lda(self):
        """All feature values without LDA must be in [0, 1]."""
        pp = TRECPreprocessor(n_features_tfidf=32, n_features_lda=None)
        X = pp.fit_transform(SAMPLE_TEXTS, SAMPLE_LABELS)
        assert X.min() >= -1e-6
        assert X.max() <= 1.0 + 1e-6

    # ------------------------------------------------------------------
    # dtype
    # ------------------------------------------------------------------

    def test_dtype_is_float32(self):
        """Output dtype must be float32."""
        pp = TRECPreprocessor(n_features_tfidf=32, n_features_lda=4)
        X = pp.fit_transform(SAMPLE_TEXTS, SAMPLE_LABELS)
        assert X.dtype == np.float32

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_transform_before_fit_raises(self):
        """transform() before fit_transform() must raise RuntimeError."""
        pp = TRECPreprocessor(n_features_tfidf=32, n_features_lda=4)
        with pytest.raises(RuntimeError, match="not been fitted"):
            pp.transform(SAMPLE_TEXTS)

    # ------------------------------------------------------------------
    # Transform consistency
    # ------------------------------------------------------------------

    def test_transform_shape_matches_fit(self):
        """transform() on a subset must return the same feature dimension."""
        pp = TRECPreprocessor(n_features_tfidf=32, n_features_lda=4)
        pp.fit_transform(SAMPLE_TEXTS, SAMPLE_LABELS)
        X_test = pp.transform(SAMPLE_TEXTS[:3])
        assert X_test.shape == (3, 4)

    def test_transform_values_in_unit_range(self):
        """transform() output must also be in [0, 1]."""
        pp = TRECPreprocessor(n_features_tfidf=32, n_features_lda=4)
        pp.fit_transform(SAMPLE_TEXTS, SAMPLE_LABELS)
        X_test = pp.transform(SAMPLE_TEXTS[:5])
        assert X_test.min() >= -1e-6
        assert X_test.max() <= 1.0 + 1e-6

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def test_n_output_features_with_lda(self):
        """n_output_features must equal n_features_lda."""
        pp = TRECPreprocessor(n_features_tfidf=32, n_features_lda=4)
        pp.fit_transform(SAMPLE_TEXTS, SAMPLE_LABELS)
        assert pp.n_output_features == 4

    def test_n_output_features_no_lda(self):
        """n_output_features without LDA must equal n_features_tfidf."""
        pp = TRECPreprocessor(n_features_tfidf=32, n_features_lda=None)
        pp.fit_transform(SAMPLE_TEXTS, SAMPLE_LABELS)
        assert pp.n_output_features == 32

    def test_repr_contains_class_name(self):
        pp = TRECPreprocessor()
        assert "TRECPreprocessor" in repr(pp)

    def test_repr_contains_lda_key(self):
        """repr should mention n_features_lda (not n_features_pca)."""
        pp = TRECPreprocessor(n_features_tfidf=32, n_features_lda=4)
        assert "n_features_lda" in repr(pp)

    def test_explained_variance_available_after_fit(self):
        """LDA explained variance ratio must be accessible after fitting."""
        pp = TRECPreprocessor(n_features_tfidf=32, n_features_lda=4)
        pp.fit_transform(SAMPLE_TEXTS, SAMPLE_LABELS)
        evr = pp.explained_variance_ratio
        assert evr is not None
        assert len(evr) == 4
        assert (evr >= 0).all()

    def test_explained_variance_none_without_lda(self):
        """explained_variance_ratio must be None when LDA is disabled."""
        pp = TRECPreprocessor(n_features_tfidf=32, n_features_lda=None)
        pp.fit_transform(SAMPLE_TEXTS, SAMPLE_LABELS)
        assert pp.explained_variance_ratio is None

    def test_vocabulary_size_after_fit(self):
        """vocabulary_size must be ≤ n_features_tfidf."""
        pp = TRECPreprocessor(n_features_tfidf=32, n_features_lda=4)
        pp.fit_transform(SAMPLE_TEXTS, SAMPLE_LABELS)
        assert pp.vocabulary_size <= 32
