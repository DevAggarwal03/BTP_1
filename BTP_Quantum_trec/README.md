# Quantum Prototypical Network (quantum-proto-net)

A hybrid quantum-classical few-shot text classification system built on the **TREC-50** fine-grained question classification dataset.
Implements the full pipeline from the **Quantum Prototypical Network** architecture.

## Architecture

```
Input Data & Preprocessing
        ↓
Quantum Feature Selection (QCHBA)
        ↓
Quantum Feature Extractor (VQC / QNN)   ← [Phase 2]
        ↓
Quantum Prototype Calculation           ← [Phase 2]
        ↓
Quantum Distance Measurement            ← [Phase 2]
        ↓
Quantum Classifier → OUTPUT             ← [Phase 2]
```

## Project Structure

```
quantum-proto-net/
├── config/config.yaml              # All hyperparameters
├── data/
│   ├── trec_loader.py              # TREC-6 download & load
│   ├── preprocessor.py             # TF-IDF → Normalize → PCA
│   └── episode_sampler.py          # N-way K-shot (Phase 3)
├── quantum/
│   ├── encoding/
│   │   ├── angle_encoding.py       # RY rotation encoding
│   │   └── amplitude_encoding.py   # StatePreparation encoding
│   └── feature_selection/
│       ├── qhba.py                 # Core QHBA loop
│       ├── honey_badger_ops.py     # HBA update operators
│       └── quantum_oracle.py       # Qiskit fitness oracle
├── models/                         # [Phase 2] QNN, Prototype, Classifier
├── training/                       # [Phase 2] Training loop
├── experiments/
│   └── run_preprocessing.py        # Block 1+2 entry point
└── tests/
```

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Block 1 + 2 pipeline
python experiments/run_preprocessing.py
```

## Current Status

| Block | Status |
|-------|--------|
| Block 1 — Input Data & Preprocessing | ✅ Implemented |
| Block 2 — QHBA Feature Selection | ✅ Implemented |
| Block 3 — QNN Feature Extractor | 🔲 Scaffold only |
| Block 4 — Prototype Calculation | 🔲 Scaffold only |
| Block 5 — Quantum Classifier | 🔲 Scaffold only |
| Block 6 — Training & Optimization | 🔲 Scaffold only |
