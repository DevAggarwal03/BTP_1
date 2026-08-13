# Baseline Models — Classical & Quantum Comparisons

## Why Baselines Matter

A research paper proposing a new method cannot make any claim without measuring it against alternatives. The entire purpose of including baselines is to answer:

> *"Is the Quantum Prototypical Network (QPN) actually better than what already exists, and is that improvement statistically significant?"*

All baselines in this project receive **the exact same 8 QHBA-selected features** as the QPN. This is the only way to guarantee that any performance difference is due to the model architecture (quantum vs. classical), not differences in input representation.

All models are evaluated on **the same set of few-shot episodes**, drawn from the same `EpisodeSampler`. Per-episode accuracy scores are collected and compared using a **Wilcoxon signed-rank paired test** to determine statistical significance (p < 0.05).

---

## The Models

### 1. Classical Untrained ProtoNet (Nearest Centroid)

**File:** `baselines.py` — `UntrainedProtoNet`

This is the **direct classical equivalent of QPN** — using the same prototypical network idea, but with purely classical operations.

| Component | Classical ProtoNet | Quantum ProtoNet (Ours) |
|---|---|---|
| Encoder | Identity (raw features) | VQC `U(θ)` circuit |
| Prototype | Mean feature vector `c_k = mean(support_x)` | Density Matrix `ρ_k = (1/K) Σ \|ψᵢ⟩⟨ψᵢ\|` |
| Distance | Euclidean distance `\|\|z - c_k\|\|²` | Quantum Infidelity `1 - F(\|ψ⟩, ρ_k)` |

**No training** — prototypes are computed directly from the support set at inference time.

**Why include it?** This is the most natural baseline. If QPN doesn't beat this, nothing quantum is being gained.

---

### 2. Classical Trained ProtoNet (MLP Encoder)

**File:** `baselines.py` — `TrainedProtoNet`

Same prototypical network idea, but the encoder is a small **Multi-Layer Perceptron (MLP)** trained episodically using the same Snell et al. (2017) loss function:

```
MLPEncoder:  Linear(8 → 128) → ReLU → Linear(128 → 32)
```

Trained with Adam optimizer over the same number of training episodes as QPN. Uses Euclidean distance on the MLP embeddings.

**Why include it?** This is the strongest classical baseline — a fully trained classical representation. If QPN beats this, it demonstrates quantum-specific learning. If it doesn't, the quantum advantage claim weakens.

---

### 3. Support Vector Machine — RBF Kernel

**File:** `baselines.py` — `ScikitLearnBaseline(SVC, kernel='rbf')`

A standard SVM with a radial basis function (Gaussian) kernel. Fit on the K support examples, predicts on the Q query examples.

**Why include it?** SVMs are extremely competitive in low-data regimes (exactly the few-shot setting). The RBF kernel is the standard go-to non-linear classifier.

---

### 4. Logistic Regression

**File:** `baselines.py` — `ScikitLearnBaseline(LogisticRegression, max_iter=1000)`

A simple linear classifier. With only 8 features and K=1 or K=5 support examples per class, this is a very weak baseline but provides a floor to beat.

**Why include it?** Represents the simplest possible learned linear boundary. Even beating this convincingly with QPN is worth reporting.

---

### 5. k-Nearest Neighbors (k=1)

**File:** `baselines.py` — `ScikitLearnBaseline(KNeighborsClassifier, n_neighbors=1)`

A 1-NN classifier. For 1-shot learning with K=1 support, 1-NN is actually equivalent to the Classical Untrained ProtoNet since there's only one neighbor per class.

**Why include it?** Provides a distance-based comparison point that complements both the SVM and the ProtoNet. Also useful for 5-shot comparisons where 1-NN and ProtoNet diverge.

---

### 6. Quantum Kernel SVM — QSVC (Competing Quantum Baseline)

**File:** `baselines.py` — `QuantumKernelBaseline`

Uses Qiskit's `FidelityQuantumKernel` with **Angle Encoding** as the feature map. The quantum kernel computes the fidelity (inner product) between all pairs of encoded quantum states, producing a kernel matrix that is fed to a standard SVM.

```
K(xᵢ, xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|²    (quantum fidelity kernel)
```

This is the main competing **quantum** baseline. It also uses quantum circuits, but uses them to define a kernel function for a classical SVM — a very different architectural choice from QPN.

**Why include it?** This baseline tests whether the *prototypical network structure* (density matrix prototypes + fidelity inference) offers an advantage over simply using quantum kernels with a classical SVM. If QPN beats QSVC, it supports the architectural novelty claim of the paper.

---

## Experimental Settings

All models are evaluated across all four standard few-shot settings:

| Setting | N-way | K-shot | Query/class | Total support | Total queries |
|---|---|---|---|---|---|
| **5w1s** | 5 | 1 | 15 | 5 | 75 |
| **5w5s** | 5 | 5 | 15 | 25 | 75 |
| **10w1s** | 10 | 1 | 15 | 10 | 150 |
| **10w5s** | 10 | 5 | 15 | 50 | 150 |

---

## The Benchmark Output

Running `python experiments/run_benchmarks.py` generates a `RESULTS.md` table like this:

### 5-way 1-shot (Example)

| Model | Accuracy | Weighted F1 | Time (s) | p-value (vs QPN) |
|---|---|---|---|---|
| Classical ProtoNet (Untrained) | 45.2 ± 3.1 | 43.8 ± 3.0 | 12.4 | **0.003** |
| Classical ProtoNet (Trained MLP) | 51.7 ± 2.9 | 50.2 ± 2.8 | 84.1 | **0.021** |
| Classical SVM (RBF) | 48.3 ± 3.2 | 47.1 ± 3.1 | 8.7 | **0.012** |
| Classical LogReg | 41.6 ± 3.4 | 40.2 ± 3.3 | 3.2 | **0.001** |
| Classical kNN (k=1) | 44.8 ± 3.1 | 43.5 ± 3.0 | 2.1 | **0.004** |
| Quantum QSVC (Angle) | 53.1 ± 2.8 | 51.9 ± 2.7 | 142.3 | 0.214 |
| **Quantum ProtoNet (Ours)** | **57.4 ± 2.6** | **55.8 ± 2.5** | 310.7 | — |

> **Bold p-values** indicate statistically significant difference (p < 0.05). A p-value of "-" means this is the reference model.

---

## Important Design Decision: Same Features for All

A critical point for research validity: **all models in the benchmark receive the same 8 QHBA-selected feature indices**. The features are selected once by QHBA on the full training set, and all models — classical and quantum — receive this same restricted feature view.

This design ensures:
- Any accuracy difference is due to the model architecture, not the input representation.
- Classical baselines cannot cheat by accessing more features than the quantum model.
- The QHBA feature selection itself is validated as part of the contribution.
