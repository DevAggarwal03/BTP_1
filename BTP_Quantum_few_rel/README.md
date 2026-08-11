# Quantum Prototypical Networks for FewRel

A research-grade, reproducible experimental pipeline for a quantum prototypical network for few-shot relation classification in NLP.
Sentences are embedded as quantum states, class prototypes are density-matrix centroids, and inference is fidelity (swap-test) based.

## Installation

1. Requires Python 3.11
2. Create a virtual environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Setup and Data Preparation

Run the following scripts to download the dataset and build the embeddings:
```bash
python -m scripts.download_fewrel
python -m scripts.build_embeddings
```

## Running Experiments

*To be implemented in later milestones.*

## Acknowledgements / Reuse

- **FewRel Data**: Han et al. 2018 (EMNLP), Gao et al. 2019 (FewRel 2.0). Data splits sourced from THUNLP/FewRel.
- **Prototypical Networks**: Snell, Swersky, Zemel 2017 (NeurIPS).
- **Sentence Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`.

## Just some extra notes

Angle vs ZZ (Sweeps the fm_kind parameter)
Qubit Count (Sweeps 2, 4, 6, 8 qubits, dynamically refitting QCHBA for each)
Feature Selector (Evaluates QCHBA vs baseline ANOVA-F)
Circuit Depth (Sweeps ansatz_reps from 1 to 4)
Cost Function (Evaluates global vs local trace fidelity)


1. Head-to-Head Benchmark Comparisons
For every setting (5-way 1-shot, 5-way 5-shot, 10-way 1-shot, 10-way 5-shot), the proposed Quantum ProtoNet is pitted directly against 5 baselines.

Crucially, to ensure a 100% fair comparison, every single one of these models receives the exact same input: the 8-dimensional scaled features strictly selected by QCHBA.

Classical ProtoNet (Untrained): The classical equivalent of our model (Nearest-Centroid using Euclidean distance).
Classical SVM (RBF): A standard Support Vector Machine with a non-linear RBF kernel.
Classical LogReg: Standard Logistic Regression.
Classical kNN (k=1): A 1-Nearest Neighbor classifier.
Quantum QSVC (ZZ): A state-of-the-art quantum baseline using a Fidelity Quantum Kernel and an SVM backend.
Quantum ProtoNet (Ours): Our proposed parameter-shift trained, fidelity-based quantum prototype network.
For all classical models above, statistical significance (p-values) is calculated against our Quantum ProtoNet to prove if our model is definitively better.

2. Internal Ablation Comparisons
In this phase, we lock the setting to 5-way 1-shot and isolate individual components of our proposed architecture to prove why we designed it the way we did.

Feature Selector Ablation:
QCHBA (Our proposed heuristic) vs. ANOVA F-Test (The standard statistical baseline)
Quantum Encoding Ablation:
ZZ Feature Map (Highly entangled data encoding) vs. Angle Encoding (Independent qubit rotations with zero entanglement)
Qubit Count Sweep:
Comparing the model's accuracy when constrained to 2, 4, 6, and 8 qubits (and subsequently 2, 4, 6, and 8 features).
Circuit Depth (Ansatz) Sweep:
Comparing the expressivity of the parameterized quantum circuit by varying the repetitions (layers) from 1 to 4.
Cost Function Ablation:
Global Fidelity (Computing the overlap of the entire 8-qubit state) vs. Local Fidelity (Tracing out all qubits except the first one to combat barren plateaus in barren landscapes).