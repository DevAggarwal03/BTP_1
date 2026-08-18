"""
qhba.py
Quantum Honey Badger Algorithm (QHBA) for feature selection.

Algorithm overview
------------------
QHBA extends the classical Honey Badger Algorithm (HBA) by replacing the
classical fitness function with a *hybrid* evaluator:

    fitness(x) = 0.5 · quantum_oracle(x)  +  0.5 · knn_fitness(binarize(x))

Each agent maintains a continuous position vector in [0, 1]^n_features.
Binary feature masks are derived per-evaluation via a V-shaped transfer
function (``binarize``), allowing gradient-free combinatorial search.

Iteration steps
---------------
    1. Compute smell intensity I_i = r² / (4π d²) for each agent.
    2. For each agent, randomly choose:
          Honey phase  (50%): exploit toward global best + prey.
          Badger phase (50%): explore via intensity-driven digging.
    3. Clip new positions to [0, 1].
    4. Evaluate hybrid fitness for all agents.
    5. Update global best if improvement found.
    6. Repeat for max_iter iterations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .honey_badger_ops import (
    badger_phase_update,
    binarize,
    compute_intensity,
    honey_phase_update,
    knn_fitness,
)
from .quantum_oracle import QuantumOracle


# ──────────────────────────────────────────────────────────────────────────────
# Configuration & Result dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QHBAConfig:
    """Hyperparameters for the QHBA optimizer."""

    n_agents: int = 10
    """Number of agents (population size)."""

    max_iter: int = 30
    """Maximum number of optimization iterations."""

    c1: float = 0.5
    """Honey phase coefficient — controls exploitation strength."""

    c2: float = 0.5
    """Badger phase coefficient — controls exploration decay rate."""

    n_qubits: int = 4
    """Number of qubits for the quantum oracle."""

    shots: int = 1024
    """Measurement shots per oracle call."""

    use_quantum_oracle: bool = True
    """If False, use only the classical KNN fitness (baseline mode)."""

    quantum_weight: float = 0.5
    """Weight of quantum fitness in hybrid score (1 − weight for KNN)."""

    seed: Optional[int] = 42
    """Random seed for reproducibility."""


@dataclass
class QHBAResult:
    """Results returned by ``QHBA.fit()``."""

    best_mask: np.ndarray
    """Binary feature mask of shape ``(n_features,)``."""

    best_fitness: float
    """Best (lowest) fitness value achieved."""

    fitness_history: list[float] = field(default_factory=list)
    """Best fitness per iteration (length = max_iter + 1)."""

    selected_indices: list[int] = field(default_factory=list)
    """Indices of selected features (where ``best_mask > 0.5``)."""

    def __str__(self) -> str:
        n_sel = len(self.selected_indices)
        n_tot = len(self.best_mask)
        return (
            f"QHBAResult(\n"
            f"  best_fitness    = {self.best_fitness:.4f}\n"
            f"  selected        = {n_sel}/{n_tot} features\n"
            f"  feature_indices = {self.selected_indices}\n"
            f")"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Core QHBA class
# ──────────────────────────────────────────────────────────────────────────────

class QHBA:
    """
    Quantum Honey Badger Algorithm for wrapper-based feature selection.

    Args:
        n_features: Total number of input features (dimensionality of search space).
        config:     ``QHBAConfig`` hyperparameters. Uses defaults if ``None``.
    """

    def __init__(
        self,
        n_features: int,
        config: Optional[QHBAConfig] = None,
    ) -> None:
        self.n_features = n_features
        self.cfg = config or QHBAConfig()
        self.rng = np.random.default_rng(seed=self.cfg.seed)

        if self.cfg.use_quantum_oracle:
            self.oracle = QuantumOracle(
                n_qubits=min(self.cfg.n_qubits, n_features),
                shots=self.cfg.shots,
                seed=self.cfg.seed or 42,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitness_fn: Optional[Callable[[np.ndarray], float]] = None,
        verbose: bool = True,
    ) -> QHBAResult:
        """
        Run QHBA to find the optimal feature subset.

        Args:
            X:          Feature matrix ``(N_samples, n_features)``.
            y:          Integer label array ``(N_samples,)``.
            fitness_fn: Optional external fitness function ``f(position) → float``.
                        If provided, overrides the quantum + KNN hybrid evaluator.
            verbose:    Print per-iteration progress.

        Returns:
            :class:`QHBAResult` with the best binary mask and fitness history.
        """
        n = self.n_features
        n_agents = self.cfg.n_agents

        # ── Initialise population ──────────────────────────────────────
        positions = self.rng.uniform(size=(n_agents, n))       # (n_agents, n_features)
        fitnesses = self._evaluate_all(positions, X, y, fitness_fn)

        best_idx = int(np.argmin(fitnesses))
        best_pos = positions[best_idx].copy()
        best_fit = float(fitnesses[best_idx])
        history: list[float] = [best_fit]

        if verbose:
            print(f"[QHBA] Initial best fitness: {best_fit:.4f}")

        # ── Main loop ──────────────────────────────────────────────────
        for t in range(1, self.cfg.max_iter + 1):
            # Decreasing factor: starts large (exploration) → small (exploitation)
            alpha = self.cfg.c1 * np.exp(-self.cfg.c2 * t / self.cfg.max_iter)

            # Smell intensities for all agents
            intensities = compute_intensity(positions, best_pos)

            new_positions = np.empty_like(positions)
            for i in range(n_agents):
                if self.rng.random() < 0.5:
                    # ── Honey phase (exploitation) ──
                    prey_idx = int(self.rng.integers(0, n_agents))
                    new_pos = honey_phase_update(
                        xi=positions[i],
                        x_best=best_pos,
                        x_prey=positions[prey_idx],
                        intensity=float(intensities[i]),
                        alpha=alpha,
                        c1=self.cfg.c1,
                        rng=self.rng,
                    )
                else:
                    # ── Badger phase (exploration) ──
                    new_pos = badger_phase_update(
                        xi=positions[i],
                        x_best=best_pos,
                        intensity=float(intensities[i]),
                        alpha=alpha,
                        c2=self.cfg.c2,
                        rng=self.rng,
                    )

                new_positions[i] = np.clip(new_pos, 0.0, 1.0)

            positions = new_positions
            fitnesses = self._evaluate_all(positions, X, y, fitness_fn)

            curr_best_idx = int(np.argmin(fitnesses))
            if fitnesses[curr_best_idx] < best_fit:
                best_fit = float(fitnesses[curr_best_idx])
                best_pos = positions[curr_best_idx].copy()

            history.append(best_fit)

            if verbose:
                print(
                    f"[QHBA] Iter {t:3d}/{self.cfg.max_iter}"
                    f"  |  alpha={alpha:.4f}"
                    f"  |  best_fitness={best_fit:.4f}"
                )

        # ── Extract binary mask from best continuous position ──────────
        best_mask = binarize(best_pos)
        selected = list(np.where(best_mask > 0.5)[0])

        return QHBAResult(
            best_mask=best_mask,
            best_fitness=best_fit,
            fitness_history=history,
            selected_indices=selected,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_all(
        self,
        positions: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        fitness_fn: Optional[Callable],
    ) -> np.ndarray:
        """Evaluate hybrid fitness for all agents."""
        fitnesses = np.zeros(len(positions), dtype=float)

        for i, pos in enumerate(positions):
            if fitness_fn is not None:
                # User-supplied fitness overrides everything
                fitnesses[i] = fitness_fn(pos)

            elif self.cfg.use_quantum_oracle:
                # Hybrid: 50% quantum oracle + 50% classical KNN
                q_fit = self.oracle.evaluate(pos)
                mask = binarize(pos)
                c_fit = knn_fitness(X, y, mask)
                w = self.cfg.quantum_weight
                fitnesses[i] = w * q_fit + (1.0 - w) * c_fit

            else:
                # Classical-only baseline
                mask = binarize(pos)
                fitnesses[i] = knn_fitness(X, y, mask)

        return fitnesses

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"QHBA("
            f"n_features={self.n_features}, "
            f"n_agents={self.cfg.n_agents}, "
            f"max_iter={self.cfg.max_iter})"
        )
