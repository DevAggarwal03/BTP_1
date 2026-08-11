"""
honey_badger_ops.py
Classical Honey Badger Algorithm (HBA) update operators and fitness functions.

The Honey Badger Algorithm is a swarm-intelligence metaheuristic that models
the foraging behaviour of honey badgers via two complementary strategies:

    Honey phase  (exploitation): agents move toward the best-known position,
                                  guided by smell intensity and a prey target.
    Badger phase (exploration):  agents perform intensity-driven digging,
                                  biased away from the current position.

Reference
---------
Hashim, F. A., Houssein, E. H., Hussain, K., Mabrouk, M. S., & Al-Atabany, W.
"Honey Badger Algorithm: New metaheuristic algorithm for solving optimization problems."
Mathematics and Computers in Simulation, 192, 84–110. (2022)
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# ──────────────────────────────────────────────────────────────────────────────
# Smell / intensity computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_intensity(
    positions: np.ndarray,
    best_pos: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Compute the smell intensity for each agent toward the best (prey) position.

    Modelled after the inverse-square law of scent diffusion:
        I_i = r² / (4π d²)   where d = ‖x_i − x_best‖

    Args:
        positions: Current agent positions, shape ``(n_agents, n_features)``.
        best_pos:  Global best position, shape ``(n_features,)``.
        epsilon:   Numerical stability floor added to squared distances.

    Returns:
        intensity: Shape ``(n_agents,)`` non-negative intensity values.
    """
    diff = positions - best_pos                         # (n_agents, n_features)
    dist_sq = np.sum(diff ** 2, axis=1) + epsilon       # (n_agents,)
    r = np.random.default_rng().random(size=len(positions))
    return (r ** 2) / (4.0 * np.pi * dist_sq)


# ──────────────────────────────────────────────────────────────────────────────
# Phase update operators
# ──────────────────────────────────────────────────────────────────────────────

def honey_phase_update(
    xi: np.ndarray,
    x_best: np.ndarray,
    x_prey: np.ndarray,
    intensity: float,
    alpha: float,
    c1: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Honey phase update (exploitation).

    Moves agent ``xi`` toward ``x_best`` using prey attraction and
    a random perturbation modulated by smell intensity:

        x_new = x_best + F · β · I · x_best + F · r₁ · α · |x_i − x_prey|

    Args:
        xi:        Current agent position ``(n_features,)``.
        x_best:    Global best position ``(n_features,)``.
        x_prey:    Random other agent's position (prey) ``(n_features,)``.
        intensity: Scalar smell intensity I_i for this agent.
        alpha:     Decreasing factor α = c₁ · exp(−c₂ · t / T).
        c1:        Honey phase scaling coefficient.
        rng:       NumPy random Generator for reproducibility.

    Returns:
        New position (before clipping to ``[0, 1]``).
    """
    F = float(rng.choice([-1, 1]))      # random directional flag
    r1 = rng.random()
    beta = rng.random()

    return (
        x_best
        + F * beta * intensity * x_best
        + F * r1 * alpha * np.abs(xi - x_prey)
    )


def badger_phase_update(
    xi: np.ndarray,
    x_best: np.ndarray,
    intensity: float,
    alpha: float,
    c2: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Badger phase update (exploration).

    Models the digging behaviour — a more aggressive random exploration
    biased by intensity away from the current position:

        x_new = x_best + F · r₂ · α · I · |x_best − x_i|

    Args:
        xi:        Current agent position ``(n_features,)``.
        x_best:    Global best position ``(n_features,)``.
        intensity: Scalar smell intensity I_i for this agent.
        alpha:     Decreasing factor.
        c2:        Badger phase scaling coefficient.
        rng:       NumPy random Generator.

    Returns:
        New position (before clipping to ``[0, 1]``).
    """
    F = float(rng.choice([-1, 1]))
    r2 = rng.random()

    return (
        x_best
        + F * r2 * alpha * intensity * np.abs(x_best - xi)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Binarization (continuous position → binary feature mask)
# ──────────────────────────────────────────────────────────────────────────────

def binarize(position: np.ndarray) -> np.ndarray:
    """
    Convert a continuous position vector to a binary feature mask using the
    V-shaped transfer function based on arctan:

        V(x) = |2/π · arctan(π/2 · x)|
        mask[i] = 1  if  rand_i < V(x_i)  else  0

    Args:
        position: Continuous position vector, typically in ``[0, 1]``.

    Returns:
        Binary mask array of the same shape as ``position``.
    """
    transfer = np.abs((2.0 / np.pi) * np.arctan((np.pi / 2.0) * position))
    rand_vals = np.random.default_rng().random(size=position.shape)
    return (rand_vals < transfer).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Classical fitness function (KNN wrapper, used as baseline / hybrid component)
# ──────────────────────────────────────────────────────────────────────────────

def knn_fitness(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    k: int = 3,
    lambda_sparsity: float = 0.01,
) -> float:
    """
    Classical KNN-based wrapper fitness function for a binary feature mask.

    Fitness =  (1 − CV_accuracy)  +  λ · |selected| / |total|

    A lower fitness means a better feature subset:
        - Higher accuracy → lower error term.
        - Fewer features  → lower sparsity penalty (Occam's razor).

    Args:
        X:               Feature matrix ``(N, D)``.
        y:               Integer label array ``(N,)``.
        mask:            Binary mask ``(D,)``; values > 0.5 are treated as selected.
        k:               Number of KNN neighbours.
        lambda_sparsity: Sparsity penalty weight λ.

    Returns:
        Scalar fitness value in approximately ``[0, 1]`` (lower = better).
    """
    selected = mask > 0.5

    if selected.sum() == 0:
        return 1.0          # penalise empty feature selection heavily

    X_sub = X[:, selected]
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_sub, y, cv=3, scoring="accuracy")
    accuracy = float(scores.mean())

    sparsity_penalty = lambda_sparsity * int(selected.sum()) / len(mask)
    return float(1.0 - accuracy + sparsity_penalty)
