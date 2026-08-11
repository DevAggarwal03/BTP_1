# Codebase Documentation — Quantum Prototypical Network (Full Pipeline)

> This document explains what each part of the codebase does, how the architecture flows from end to end, and how to run the final training pipeline. It covers all Blocks (1 through 8) of the Two-Loop System.

---

## 1. Pipeline Overview

The full architecture consists of an **Outer Loop** (Classical Feature Selection) and an **Inner Loop** (Quantum Neural Network Training).

```
┌────────────────────────────────────────────────────────┐
│  Phase 1 — Input Data & Preprocessing (Block 1)        │
│  Load TREC-50 Text → SBERT Embeddings → LDA Reduction  │
└──────────────────────────┬─────────────────────────────┘
                           │  feature matrix (N, 20)
                           ▼
┌────────────────────────────────────────────────────────┐
│  Phase 2 — Quantum Honey Badger Outer Loop (Block 2)   │
│  Guesses the best binary feature mask (8 features)     │
└──────────────────────────┬─────────────────────────────┘
                           │  filters down to (N, 8)
                           ▼
┌────────────────────────────────────────────────────────┐
│  Phase 3 — Quantum Inner Loop (Blocks 3, 4, 5, 8)      │
│  Angle Encoding → VQC → Prototypes → Cross-Entropy Loss│
│  PyTorch updates VQC angles via Parameter-Shift Rule   │
└──────────────────────────┬─────────────────────────────┘
                           │  Loss returned to Honey Badger
                           ▼
┌────────────────────────────────────────────────────────┐
│  Phase 4 — Final Evaluation (Block 7)                  │
│  Calculates Accuracy/F1, plots Confusion Matrix & t-SNE│
└────────────────────────────────────────────────────────┘
```

---

## 2. Project Structure

```text
quantum-proto-net/
├── data/
│   ├── trec_loader.py           # Loads raw TREC-50 text datasets
│   └── preprocessor.py          # Classical embeddings + LDA dimensionality reduction
├── quantum/
│   ├── encoding/                # Block 1: Angle & Amplitude encoding circuits
│   ├── vqc/                     # Block 3: Parameterized Quantum Circuits (EfficientSU2)
│   ├── prototype_calculation/   # Block 4: Calculates class centroids from quantum states
│   ├── distance_measurement/    # Block 5: Quantum State Fidelity distances
│   ├── classifier/              # Block 6: Softmax activation
│   ├── evaluation/              # Block 7: Metrics, Confusion Matrix, and t-SNE plots
│   ├── feature_selection/       # Block 2: Quantum Honey Badger Algorithm
│   └── training/                # Block 8: PyTorch wrappers & the Two-Loop Master Trainer
├── experiments/
│   ├── run_preprocessing.py     # Runs only Blocks 1 and 2
│   └── run_training_pipeline.py # Runs the FULL end-to-end pipeline
├── project_docs/                # In-depth architectural design docs and logs
├── requirements.txt
└── docs.md                      # This file
```

---

## 3. How to Run

Before running, ensure your virtual environment is activated and dependencies are installed:

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (first time only)
pip install -r requirements.txt
```

### Run the Full Two-Loop Pipeline
This command executes the entire architecture from end to end. It loads the dataset, uses the QCHBA to search for optimal features, uses PyTorch to calculate quantum gradients to train the VQC, and finally evaluates the model.

```bash
python experiments/run_training_pipeline.py
```

> **Note on Performance**: Calculating quantum gradients via the parameter-shift rule requires running the simulation thousands of times per epoch. By default, `run_training_pipeline.py` uses a "fast-run" configuration (fewer samples, few agents) to make it computationally feasible to run on a laptop CPU.

### Run Preprocessing & Feature Selection Only
If you only want to test the classical embedding (SentenceTransformers), LDA reduction, and the baseline Honey Badger optimization without training the VQC inner loop:

```bash
python experiments/run_preprocessing.py --no-quantum
```

---

## 4. Key Concepts

- **TREC-50 Dataset**: A fine-grained question classification dataset with 50 categories. Perfect for testing quantum few-shot learning.
- **Angle Encoding**: We map numerical feature values (normalized between 0 and 1) to rotational angles on the qubits' Bloch spheres using RY gates.
- **VQC (Variational Quantum Circuit)**: The "brain" of the network. It entangles the qubits. PyTorch trains the angles of this circuit.
- **TorchConnector**: A bridge from `qiskit_machine_learning` that allows PyTorch's classical `Adam` optimizer to backpropagate through the quantum circuit using the Parameter-Shift rule.
- **Quantum Prototypical Network (QPN)**: It learns to group similar questions tightly together in the quantum Hilbert space (forming Prototypes) and pushes different categories far apart.
