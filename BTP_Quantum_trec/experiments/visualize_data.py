"""
visualize_data.py
Generate two key exploratory plots for the TREC-50 dataset:

    1. Class Distribution Bar Chart
       Horizontal bars for all 50 fine-grained labels, colored by coarse
       category. Instantly shows class imbalance.

    2. 2D LDA Projection Scatter Plot
       First two LDA discriminant components of the TF-IDF features,
       colored by coarse category. Validates that the preprocessing
       pipeline produces separable clusters for downstream quantum encoding.

Usage:
    python experiments/visualize_data.py
    python experiments/visualize_data.py --save-dir gen_artifacts
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import TRECLoader, TRECPreprocessor

# ── Color palette for the 6 coarse categories ───────────────────────────────
COARSE_COLORS: dict[str, str] = {
    "Abbreviation": "#e6194b",   # red
    "Description":  "#3cb44b",   # green
    "Entity":       "#4363d8",   # blue
    "Human":        "#f58231",   # orange
    "Location":     "#911eb4",   # purple
    "Numeric":      "#42d4f4",   # cyan
}


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 — Class Distribution
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    labels: list[int],
    split_name: str = "Train",
    save_path: Path | None = None,
) -> None:
    """Horizontal bar chart of sample counts per fine-grained label."""

    counts = Counter(labels)

    # Sort by coarse group, then by fine-grained name within each group
    sorted_indices = sorted(
        range(TRECLoader.n_classes()),
        key=lambda i: TRECLoader.label_name(i),
    )

    label_names = [TRECLoader.label_name(i) for i in sorted_indices]
    label_counts = [counts.get(i, 0) for i in sorted_indices]
    bar_colors = [COARSE_COLORS[TRECLoader.coarse_label(i)] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 14))
    y_pos = np.arange(len(label_names))

    bars = ax.barh(y_pos, label_counts, color=bar_colors, edgecolor="white", linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(label_names, fontsize=8, fontfamily="monospace")
    ax.invert_yaxis()  # top-to-bottom reading order
    ax.set_xlabel("Number of Samples", fontsize=11)
    ax.set_title(
        f"TREC-50 Class Distribution ({split_name} Split)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    # Add count labels on bars
    max_count = max(label_counts) if label_counts else 1
    for bar, count in zip(bars, label_counts):
        ax.text(
            bar.get_width() + max_count * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=7,
            color="#333333",
        )

    # Legend for coarse categories
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color, label=coarse)
        for coarse, color in COARSE_COLORS.items()
    ]
    ax.legend(
        handles=legend_handles,
        title="Coarse Category",
        loc="lower right",
        fontsize=9,
        title_fontsize=10,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2 — 2D LDA Scatter
# ──────────────────────────────────────────────────────────────────────────────

def plot_lda_scatter(
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "Train",
    save_path: Path | None = None,
) -> None:
    """Scatter plot of the first 2 LDA components, colored by coarse label."""

    if X.shape[1] < 2:
        print("  ⚠  Need at least 2 LDA components for scatter plot, skipping.")
        return

    fig, ax = plt.subplots(figsize=(11, 8))

    # Plot each coarse category separately for legend grouping
    for coarse, color in COARSE_COLORS.items():
        mask = np.array([TRECLoader.coarse_label(int(yi)) == coarse for yi in y])
        if not mask.any():
            continue
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=color,
            label=coarse,
            alpha=0.55,
            s=18,
            edgecolors="white",
            linewidths=0.3,
        )

    ax.set_xlabel("LDA Component 1", fontsize=11)
    ax.set_ylabel("LDA Component 2", fontsize=11)
    ax.set_title(
        f"2D LDA Projection of TREC-50 ({split_name} Split)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    ax.legend(
        title="Coarse Category",
        fontsize=9,
        title_fontsize=10,
        markerscale=1.5,
        loc="best",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle="--")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TREC-50 dataset visualizations.")
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save plot PNGs (relative to project root). "
             "If omitted, plots are shown interactively only.",
    )
    args = parser.parse_args()

    save_dir: Path | None = None
    if args.save_dir:
        save_dir = PROJECT_ROOT / args.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading TREC-50 dataset…")
    loader = TRECLoader(cache_dir="data/cache")
    train, test = loader.load()
    y_train = np.array(train["label"], dtype=int)

    print(f"  Train: {len(train['text'])} samples, Test: {len(test['text'])} samples")

    # ── Plot 1: Class Distribution ────────────────────────────────────────────
    print("\n📊 Plot 1 — Class Distribution")
    plot_class_distribution(
        train["label"],
        split_name="Train",
        save_path=save_dir / "class_distribution.png" if save_dir else None,
    )

    # ── Plot 2: 2D LDA Scatter ────────────────────────────────────────────────
    print("\n📊 Plot 2 — 2D LDA Projection")
    # Use only 2 LDA components for visualization
    preprocessor = TRECPreprocessor(encoder_model="all-MiniLM-L6-v2", n_features_lda=2)
    X_train_2d = preprocessor.fit_transform(train["text"], y_train)

    plot_lda_scatter(
        X_train_2d,
        y_train,
        split_name="Train",
        save_path=save_dir / "lda_scatter_2d.png" if save_dir else None,
    )

    print("\n✅  Done.")


if __name__ == "__main__":
    main()
