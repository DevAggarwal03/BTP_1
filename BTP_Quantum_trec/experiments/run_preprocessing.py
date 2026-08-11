"""
run_preprocessing.py
Entry point for Blocks 1 & 2 of the Quantum Prototypical Network pipeline:

    Block 1 — Input Data & Preprocessing
        ├── Load TREC-50 from HuggingFace
        ├── SBERT vectorization (384 features)
        ├── Min-Max normalization → [0, 1]
        └── LDA reduction (384 → 32 features)

    Block 2 — Quantum Feature Selection (QHBA)
        ├── Initialize population of feature-mask agents
        ├── Evaluate hybrid fitness (quantum oracle + KNN)
        ├── Honey phase / Badger phase position updates
        └── Return optimal binary feature mask

Usage:
    python experiments/run_preprocessing.py
    python experiments/run_preprocessing.py --config config/config.yaml --no-quantum
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import TRECLoader, TRECPreprocessor
from quantum.feature_selection.qhba import QHBA, QHBAConfig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Config not found: {full_path}")
    with open(full_path) as f:
        return yaml.safe_load(f)


def print_banner(title: str) -> None:
    width = 62
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_section(label: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print("─" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run quantum-proto-net Blocks 1 & 2 (preprocessing + QHBA)."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML (relative to project root).",
    )
    parser.add_argument(
        "--no-quantum",
        action="store_true",
        help="Disable quantum oracle — use classical KNN fitness only (faster, for testing).",
    )
    args = parser.parse_args()

    print_banner("Quantum Prototypical Network — Pipeline Blocks 1 & 2")

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    q_cfg = cfg["quantum"]
    hba_cfg = cfg["qhba"]

    # ── Block 1A: Load TREC-6 ─────────────────────────────────────────────────
    print_section("Block 1A — Loading TREC-6 Dataset")
    t0 = time.perf_counter()

    loader = TRECLoader(cache_dir=data_cfg["cache_dir"])
    train, test = loader.load()

    label_counts = {TRECLoader.label_name(i): 0 for i in range(TRECLoader.n_classes())}
    for lbl in train["label"]:
        label_counts[TRECLoader.label_name(lbl)] += 1

    print(f"  Train samples : {len(train['text'])}")
    print(f"  Test  samples : {len(test['text'])}")
    print(f"  Class dist.   : {label_counts}")
    print(f"  Load time     : {time.perf_counter() - t0:.2f}s")

    # ── Block 1B: Preprocess ──────────────────────────────────────────────────
    print_section("Block 1B — Preprocessing (Embed → Normalize → LDA)")
    t0 = time.perf_counter()

    preprocessor = TRECPreprocessor(
        encoder_model=data_cfg["encoder_model"],
        n_features_lda=data_cfg["n_features_lda"],
    )

    y_train = np.array(train["label"], dtype=int)
    y_test  = np.array(test["label"],  dtype=int)

    # LDA requires labels at fit time (supervised)
    X_train = preprocessor.fit_transform(train["text"], y_train)
    X_test  = preprocessor.transform(test["text"])

    print(f"  Encoder model      : {preprocessor.encoder_model}")
    print(f"  Embedding dim      : {preprocessor.embedding_dim}")
    print(f"  X_train shape      : {X_train.shape}")
    print(f"  X_test  shape      : {X_test.shape}")
    print(f"  Value range        : [{X_train.min():.4f}, {X_train.max():.4f}]")
    if preprocessor.explained_variance_ratio is not None:
        total_var = preprocessor.explained_variance_ratio.sum() * 100
        print(f"  LDA variance kept  : {total_var:.1f}%")
    print(f"  Preprocess time    : {time.perf_counter() - t0:.2f}s")

    # ── Block 2: QHBA Feature Selection ───────────────────────────────────────
    print_section("Block 2 — Quantum Feature Selection (QHBA)")

    use_quantum = not args.no_quantum
    if not use_quantum:
        print("Quantum oracle DISABLED (--no-quantum flag set)")

    qhba_config = QHBAConfig(
        n_agents=hba_cfg["n_agents"],
        max_iter=hba_cfg["max_iter"],
        c1=hba_cfg["c1"],
        c2=hba_cfg["c2"],
        n_qubits=q_cfg["n_qubits"],
        shots=q_cfg["shots"],
        use_quantum_oracle=use_quantum,
        seed=42,
    )

    print(f"\n  Config: {qhba_config}\n")
    t0 = time.perf_counter()

    qhba = QHBA(n_features=X_train.shape[1], config=qhba_config)
    result = qhba.fit(X_train, y_train, verbose=True)

    elapsed = time.perf_counter() - t0

    # ── Results ───────────────────────────────────────────────────────────────
    print_banner("Results")
    print(result)
    print(f"\n  QHBA runtime       : {elapsed:.1f}s")
    print(f"  Fitness history    : {[f'{v:.3f}' for v in result.fitness_history]}")

    # Reduced feature matrix for downstream use
    X_train_sel = X_train[:, result.selected_indices]
    X_test_sel = X_test[:, result.selected_indices]
    print(f"\n  X_train (selected) : {X_train_sel.shape}")
    print(f"  X_test  (selected) : {X_test_sel.shape}")
    # print(f"\n  ✅  Ready for Block 3 — Quantum Feature Extractor (QNN)\n")


if __name__ == "__main__":
    main()
