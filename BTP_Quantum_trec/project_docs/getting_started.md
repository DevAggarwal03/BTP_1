# Getting Started — BTP_Quantum_trec

This guide walks you through setting up the **Quantum Prototypical Network for TREC-50** from scratch on a fresh machine (or GPU VM).

---

## 1. Prerequisites

- **Python**: 3.11 (required — Qiskit has strict version constraints)
- **Git**: to clone the repository
- **GPU (Recommended)**: NVIDIA GPU with CUDA 12+ for `qiskit-aer-gpu` acceleration
- **RAM**: At least 8 GB (the SBERT model + density matrix simulations are memory-intensive)

---

## 2. Clone the Repository

```bash
git clone <your-repo-url>
cd BTP_Quantum_trec
```

---

## 3. Create a Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows
```

---

## 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### GPU-Accelerated Aer (Optional but Recommended)

If you have a CUDA GPU, install the GPU-enabled Aer simulator. This dramatically speeds up the QHBA oracle evaluation:

```bash
pip install qiskit-aer-gpu
```

> The code automatically detects and falls back to CPU if no GPU is available — no code changes needed.

---

## 5. Verify the Installation

Run a quick sanity check to make sure all packages are importable:

```bash
python -c "
import qiskit; print('Qiskit:', qiskit.__version__)
import qiskit_aer; print('Aer:', qiskit_aer.__version__)
import sentence_transformers; print('SentenceTransformers: OK')
import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
import sklearn; print('Scikit-learn:', sklearn.__version__)
"
```

---

## 6. Configuration

All hyperparameters are centralized in [`config/config.yaml`](../config/config.yaml). Open it and review the defaults before running:

```yaml
data:
  dataset: "FastFit/trec_50"
  encoder_model: "all-MiniLM-L6-v2"
  n_features_lda: 32             # LDA dimensions before QHBA (max 49 for 50 classes)
  cache_dir: "data/cache"

quantum:
  n_qubits: 4                    # QHBA selects 4 from 32 LDA features
  shots: 1024                    # Measurement shots per oracle call

qhba:
  n_agents: 10
  max_iter: 30
  use_quantum_oracle: true       # Set false for faster CPU-only debug runs

training:
  n_way: 5
  k_shot: 1
  n_query: 15
  n_train_episodes: 100
  learning_rate: 0.01

evaluation:
  n_way: 5
  k_shot: 1
  n_query: 15
  n_test_episodes: 300
```

---

## 7. Running the Pipeline

### Option A: Full Research Pipeline (Recommended)

This runs the complete end-to-end pipeline: data loading → preprocessing → QHBA feature selection → episodic meta-training → evaluation → visualizations.

```bash
python experiments/run_experiments.py
```

**What to expect:** The terminal will print progress for each stage. The full pipeline (with default config) takes ~2-4 hours on CPU, or ~15-30 minutes on a GPU VM.

Output files will be saved to the project root:
- `confusion_matrix.png` — Heatmap of classification results
- `tsne_plot.png` — 2D t-SNE visualization of VQC quantum embeddings
- `results/` — JSON + Markdown file with accuracy and F1 scores

---

### Option B: Preprocessing + Feature Selection Only (Block 1 & 2)

Runs only the classical preprocessing and QHBA feature selection. Fast, useful for debugging the data pipeline.

```bash
python experiments/run_preprocessing.py

# To skip the quantum oracle (faster):
python experiments/run_preprocessing.py --no-quantum
```

---

### Option C: Data Visualization Only

Generates the class distribution bar chart and 2D LDA scatter plot.

```bash
python experiments/visualize_data.py

# To save to disk instead of displaying interactively:
python experiments/visualize_data.py --save-dir gen_artifacts
```

---

## 8. Running Tests

```bash
pytest tests/ -v
```

---

## 9. Project Directory Overview

```
BTP_Quantum_trec/
│
├── config/config.yaml              ← All hyperparameters (edit here)
│
├── data/
│   ├── trec_loader.py              ← Downloads & loads TREC-50 from HuggingFace
│   ├── preprocessor.py             ← SBERT → MinMax → LDA pipeline
│   └── episode_sampler.py          ← N-way K-shot episode sampling for meta-training
│
├── quantum/
│   ├── encoding/                   ← Angle & Amplitude encoding circuits
│   ├── feature_selection/          ← QHBA + quantum oracle + HBA update ops
│   ├── vqc/                        ← Parameterized VQC ansatz builder
│   ├── prototype_calculation/      ← Density matrix prototype computation
│   ├── distance_measurement/       ← Quantum infidelity (1 - fidelity)
│   ├── classifier/                 ← Softmax classifier with temperature
│   ├── training/                   ← QuantumProtoNet model + outer loop
│   ├── evaluation/                 ← Metrics, confusion matrix, t-SNE visualizer
│   └── eval/                       ← meta_train_qpn() + evaluate() harness
│
├── experiments/
│   ├── run_experiments.py          ← Full pipeline entry point (START HERE)
│   ├── run_preprocessing.py        ← Blocks 1 & 2 only
│   └── visualize_data.py           ← Exploratory data visualization
│
├── project_docs/                   ← Architecture docs, block descriptions
├── tests/                          ← Unit tests
└── requirements.txt
```

---

## 10. Common Issues & Fixes

| Problem | Fix |
|---|---|
| `LDA requires more samples than components` | Increase sample size or reduce `n_features_lda` in config |
| `CUDA not available` | Install `qiskit-aer-gpu` and verify CUDA drivers are installed |
| `HuggingFace dataset download fails` | Check internet connection; data caches locally after first run |
| `OutOfMemoryError` for large n_qubits | Reduce `n_qubits` to 6 or 4 in config; density matrix size = 2^n × 2^n |
| `ModuleNotFoundError` | Ensure you are in the project root and the `.venv` is activated |
