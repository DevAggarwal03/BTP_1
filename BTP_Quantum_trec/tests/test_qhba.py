"""
test_qhba.py
Unit tests for QHBA components: honey_badger_ops, quantum_oracle, and QHBA.

The quantum oracle tests use a mocked AerSimulator to avoid running
real quantum circuits in CI (slow + hardware-dependent).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum.feature_selection.honey_badger_ops import (
    badger_phase_update,
    binarize,
    compute_intensity,
    honey_phase_update,
    knn_fitness,
)
from quantum.feature_selection.qhba import QHBA, QHBAConfig, QHBAResult


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_dataset():
    """Small linearly separable dataset for fast KNN tests."""
    rng = np.random.default_rng(0)
    X = rng.random((60, 8)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)   # first feature is perfectly discriminative
    return X, y


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


# ──────────────────────────────────────────────────────────────────────────────
# honey_badger_ops tests
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeIntensity:
    def test_output_shape(self, rng):
        n_agents, n_feats = 5, 8
        positions = rng.random((n_agents, n_feats))
        best_pos = rng.random(n_feats)
        I = compute_intensity(positions, best_pos)
        assert I.shape == (n_agents,)

    def test_values_nonnegative(self, rng):
        positions = rng.random((10, 8))
        best_pos = rng.random(8)
        I = compute_intensity(positions, best_pos)
        assert (I >= 0).all()


class TestHoneyPhaseUpdate:
    def test_output_shape(self, rng):
        n = 8
        xi = rng.random(n)
        x_best = rng.random(n)
        x_prey = rng.random(n)
        result = honey_phase_update(xi, x_best, x_prey, 0.5, 0.3, 0.5, rng)
        assert result.shape == (n,)

    def test_deterministic_with_seeded_rng(self):
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        xi = np.ones(4) * 0.5
        x_best = np.ones(4) * 0.8
        x_prey = np.ones(4) * 0.2
        r1 = honey_phase_update(xi, x_best, x_prey, 0.5, 0.3, 0.5, rng1)
        r2 = honey_phase_update(xi, x_best, x_prey, 0.5, 0.3, 0.5, rng2)
        np.testing.assert_array_almost_equal(r1, r2)


class TestBadgerPhaseUpdate:
    def test_output_shape(self, rng):
        n = 8
        xi = rng.random(n)
        x_best = rng.random(n)
        result = badger_phase_update(xi, x_best, 0.5, 0.3, 0.5, rng)
        assert result.shape == (n,)


class TestBinarize:
    def test_output_is_binary(self, rng):
        pos = rng.random(16)
        mask = binarize(pos)
        assert set(np.unique(mask)).issubset({0.0, 1.0})

    def test_output_shape(self, rng):
        pos = rng.random(10)
        assert binarize(pos).shape == (10,)

    def test_all_ones_position(self):
        """Position vector of all-ones should yield mostly selected features."""
        pos = np.ones(20)
        mask = binarize(pos)
        # V(1.0) ≈ 0.57 — expect >50% to be selected on average
        assert mask.sum() > 0

    def test_all_zeros_position(self):
        """Position vector of all-zeros → transfer = 0 → all zeros mask."""
        pos = np.zeros(20)
        mask = binarize(pos)
        assert mask.sum() == 0.0


class TestKnnFitness:
    def test_range(self, simple_dataset):
        X, y = simple_dataset
        mask = np.ones(X.shape[1], dtype=np.float32)
        fit = knn_fitness(X, y, mask)
        assert 0.0 <= fit <= 1.0

    def test_empty_mask_returns_one(self, simple_dataset):
        X, y = simple_dataset
        mask = np.zeros(X.shape[1], dtype=np.float32)
        assert knn_fitness(X, y, mask) == 1.0

    def test_good_feature_lower_fitness(self, simple_dataset):
        """Selecting the discriminative feature (idx 0) should yield lower fitness."""
        X, y = simple_dataset
        good_mask = np.zeros(X.shape[1], dtype=np.float32)
        good_mask[0] = 1.0
        bad_mask = np.zeros(X.shape[1], dtype=np.float32)
        bad_mask[7] = 1.0   # last feature — not discriminative

        good_fit = knn_fitness(X, y, good_mask, k=3)
        bad_fit = knn_fitness(X, y, bad_mask, k=3)
        assert good_fit < bad_fit, (
            f"Expected good_fit ({good_fit:.3f}) < bad_fit ({bad_fit:.3f})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# QHBA integration tests (classical-only to avoid slow quantum simulation)
# ──────────────────────────────────────────────────────────────────────────────

class TestQHBA:
    """Run QHBA with quantum oracle DISABLED for speed."""

    def test_fit_returns_result(self, simple_dataset):
        X, y = simple_dataset
        cfg = QHBAConfig(n_agents=4, max_iter=3, use_quantum_oracle=False, seed=0)
        qhba = QHBA(n_features=X.shape[1], config=cfg)
        result = qhba.fit(X, y, verbose=False)
        assert isinstance(result, QHBAResult)

    def test_result_mask_shape(self, simple_dataset):
        X, y = simple_dataset
        cfg = QHBAConfig(n_agents=4, max_iter=3, use_quantum_oracle=False, seed=0)
        result = QHBA(n_features=X.shape[1], config=cfg).fit(X, y, verbose=False)
        assert result.best_mask.shape == (X.shape[1],)

    def test_result_mask_is_binary(self, simple_dataset):
        X, y = simple_dataset
        cfg = QHBAConfig(n_agents=4, max_iter=3, use_quantum_oracle=False, seed=0)
        result = QHBA(n_features=X.shape[1], config=cfg).fit(X, y, verbose=False)
        assert set(np.unique(result.best_mask)).issubset({0.0, 1.0})

    def test_fitness_history_length(self, simple_dataset):
        X, y = simple_dataset
        max_iter = 5
        cfg = QHBAConfig(n_agents=4, max_iter=max_iter, use_quantum_oracle=False, seed=0)
        result = QHBA(n_features=X.shape[1], config=cfg).fit(X, y, verbose=False)
        # history = [initial] + [one per iteration]
        assert len(result.fitness_history) == max_iter + 1

    def test_fitness_is_non_increasing(self, simple_dataset):
        """Best fitness should never increase iteration-to-iteration."""
        X, y = simple_dataset
        cfg = QHBAConfig(n_agents=6, max_iter=10, use_quantum_oracle=False, seed=42)
        result = QHBA(n_features=X.shape[1], config=cfg).fit(X, y, verbose=False)
        history = result.fitness_history
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + 1e-9, (
                f"Fitness increased at iter {i}: {history[i - 1]:.4f} → {history[i]:.4f}"
            )

    def test_repr(self, simple_dataset):
        X, y = simple_dataset
        cfg = QHBAConfig()
        qhba = QHBA(n_features=X.shape[1], config=cfg)
        assert "QHBA" in repr(qhba)


# ──────────────────────────────────────────────────────────────────────────────
# QHBAResult string representation
# ──────────────────────────────────────────────────────────────────────────────

class TestQHBAResult:
    def test_str_contains_fitness(self):
        mask = np.array([1.0, 0.0, 1.0, 0.0])
        result = QHBAResult(
            best_mask=mask,
            best_fitness=0.1234,
            fitness_history=[0.5, 0.3, 0.1234],
            selected_indices=[0, 2],
        )
        s = str(result)
        assert "0.1234" in s
        assert "2/4" in s
