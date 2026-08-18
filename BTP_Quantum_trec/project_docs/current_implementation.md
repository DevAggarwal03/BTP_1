# Current Implementation Status — BTP_Quantum_trec

**Last Updated:** 2026-08-16  
**Branch:** `main` (merged from `dev_agr` via fast-forward, no conflicts)

---

## Architecture Overview

```
Raw Text (TREC-50)
      │
      ▼
[Block 1] Preprocessing
  SentenceTransformer → 384-dim embeddings → MinMax → LDA → 32-dim features
      │
      ▼
[Block 2] Quantum Honey Badger Algorithm (QHBA) — Outer Feature Selection Loop
  QHBA Swarm (n_agents=10, max_iter=30)
    ├── Honey Phase (exploitation)
    ├── Badger Phase (exploration)
    └── Hybrid Fitness = 0.5 × QuantumOracle + 0.5 × KNN Fitness
    → Outputs: 8 best feature indices out of 32
      │
      ▼
[Block 3] VQC Feature Extractor (EfficientSU2 Ansatz)
  Angle Encoding (RY gates) + Trainable Ansatz Layers
      │
      ▼
[Block 4] Quantum Prototype Calculation
  Density Matrix ρ_k = mean(DensityMatrix(|ψ_i⟩)) over K support states
      │
      ▼
[Block 5] Quantum Distance Measurement
  Infidelity = 1 − state_fidelity(|ψ_query⟩, ρ_k)
      │
      ▼
[Block 6] Quantum Classifier
  Softmax(−β × distances) → class probabilities
      │
      ▼
[Block 7] Output & Evaluation Suite
  Accuracy / F1 / Confusion Matrix / t-SNE
```

---

## File-by-File Implementation Status

### `data/` — Data Pipeline

| File | Status | Description |
|---|---|---|
| [`trec_loader.py`](../data/trec_loader.py) | ✅ Complete | Loads `FastFit/trec_50` from HuggingFace with local disk caching. Returns `{'text': [...], 'label': [...]}`. |
| [`preprocessor.py`](../data/preprocessor.py) | ✅ Complete | Full pipeline: SentenceTransformer encoding → MinMaxScaler → LDA (supervised). `fit_transform(texts, y)` for training, `transform(texts)` for inference. Has a typo in the class docstring (`prepr``jjocessing`) — harmless but cosmetic. |
| [`episode_sampler.py`](../data/episode_sampler.py) | ✅ Complete | `Episode` dataclass + `EpisodeSampler` for N-way K-shot sampling. Handles class imbalance by filtering classes with fewer than `k_shot + n_query` examples. `build_class_pool(X, y)` helper to build the pool dict from arrays. |

---

### `quantum/feature_selection/` — QHBA Outer Loop

| File | Status | Description |
|---|---|---|
| [`qhba.py`](../quantum/feature_selection/qhba.py) | ✅ Complete | Main QHBA orchestrator. `QHBAConfig` dataclass, `QHBAResult` dataclass, `QHBA.fit(X, y, fitness_fn)`. Implements the full swarm loop: Honey Phase + Badger Phase + V-shaped binarization. Supports an optional custom `fitness_fn` override (used by `outer_loop.py`). |
| [`honey_badger_ops.py`](../quantum/feature_selection/honey_badger_ops.py) | ✅ Complete | All pure math ops: `compute_intensity`, `honey_phase_update`, `badger_phase_update`, `binarize`, `knn_fitness`. |
| [`quantum_oracle.py`](../quantum/feature_selection/quantum_oracle.py) | ✅ Complete (GPU) | Builds angle-encoded + RealAmplitudes circuits for each agent. **GPU-aware:** auto-detects `qiskit-aer-gpu` via `AerSimulator(device='GPU')` with CPU fallback. **Batched:** `evaluate_batch` submits all circuits in one `backend.run()` call. Fitness proxy = `1 − P(|0…0⟩)`. |

---

### `quantum/encoding/` — Angle Encoding

| File | Status | Description |
|---|---|---|
| [`angle_encoding.py`](../quantum/encoding/angle_encoding.py) | ✅ Complete | `AngleEncoder` class. Encodes feature vector `x ∈ [0,1]^n` as `RY(π × x_i)` on qubit `i`. Has both `encode(x)` (concrete circuit) and `build_parameterized()` (returns `ParameterVector`-based circuit for VQC integration). |

---

### `quantum/vqc/` — Variational Quantum Circuit

| File | Status | Description |
|---|---|---|
| [`vqc_extractor.py`](../quantum/vqc/vqc_extractor.py) | ✅ Complete | `VQCFeatureExtractor` wraps Qiskit circuit library ansatze. Supported types: `EfficientSU2`, `RealAmplitudes`, `TwoLocal`, `PauliTwoDesign`. Configurable `reps` and `entanglement` topology. |

---

### `quantum/prototype_calculation/` — Block 4

| File | Status | Description |
|---|---|---|
| [`prototype_ops.py`](../quantum/prototype_calculation/prototype_ops.py) | ✅ Complete (Vectorized) | `QuantumPrototypeCalculator.calculate_class_prototype(support_states)`. Uses vectorized NumPy: `np.stack([DensityMatrix(s).data ...]).mean(axis=0)`. Replaces the old `O(4^n)` Python nested loops. Warns if `n_qubits > 10` due to memory. |

---

### `quantum/distance_measurement/` — Block 5

| File | Status | Description |
|---|---|---|
| [`fidelity_ops.py`](../quantum/distance_measurement/fidelity_ops.py) | ✅ Complete | `QuantumDistanceCalculator`. `calculate_fidelity(query, prototype)` calls Qiskit's `state_fidelity`. `calculate_distance()` returns `1 − fidelity`. Used standalone for testing; the `QuantumProtoNet.forward()` calls `state_fidelity` directly. |

---

### `quantum/classifier/` — Block 6

| File | Status | Description |
|---|---|---|
| [`quantum_classifier.py`](../quantum/classifier/quantum_classifier.py) | ✅ Complete (Legacy) | `QuantumPrototypicalClassifier.classify(distances)`. Applies numerically-stable softmax over negative distances. **Note:** This module is now superseded by the temperature-scaled logit computation inside `QuantumProtoNet.forward()`. It is retained for standalone testing or single-sample usage. |

---

### `quantum/training/` — Block 8 (Core QPN Model)

| File | Status | Description |
|---|---|---|
| [`qpn_model.py`](../quantum/training/qpn_model.py) | ✅ Complete | **The central model.** Contains `QPNFunction` (custom `torch.autograd.Function` implementing the **Parameter-Shift Rule**) and `QuantumProtoNet(nn.Module)`. Both `theta` (VQC angles) and `log_beta` (softmax temperature) receive real gradients. `forward()` delegates to `QPNFunction.apply()`. `_compute_distances()` is the shared helper used by both the forward pass and PSR backward shifts. |
| [`trainer.py`](../quantum/training/trainer.py) | ✅ Complete | `MetaLearningTrainer` wraps `QuantumProtoNet` with Adam + StepLR. `train_step(support_x, support_y, query_x, query_y)` runs a single episode through `model.forward()` and backpropagates CrossEntropyLoss. Kept for use by `outer_loop.py`. Canonical meta-training is in `quantum/eval/train.py`. |
| [`outer_loop.py`](../quantum/training/outer_loop.py) | ✅ Complete | `QPNMasterTrainer`. Bridges QHBA feature search with the episodic inner loop. Fitness function: builds filtered class pool → samples N-way K-shot episode via `EpisodeSampler` → trains QPN for `epochs_per_eval` steps → returns final episode loss to QHBA. Penalises empty or oversized feature masks with `1e6`. |

---

### `quantum/eval/` — Episodic Training, Harness & Stats (New)

| File | Status | Description |
|---|---|---|
| [`train.py`](../quantum/eval/train.py) | ✅ Complete | `meta_train_qpn(model, X_train, y_train, ...)`. The canonical full episodic meta-training loop. Builds class pool, creates EpisodeSampler, runs Adam + StepLR optimizer over `n_train_episodes`. Prints per-episode loss. |
| [`harness.py`](../quantum/eval/harness.py) | ✅ Complete | `evaluate(model_fn, sampler, n_episodes)`. Model-agnostic evaluation harness. `model_fn` is any callable `episode → np.ndarray`. Computes per-episode accuracy and weighted F1, returns mean ± std ± 95% CI. |
| [`stats.py`](../quantum/eval/stats.py) | ✅ Complete | `paired_test(a_scores, b_scores)` — Wilcoxon signed-rank test for statistical significance. Returns `(p_value, effect_size)`. Handles edge cases (zero-difference arrays). |
| [`__init__.py`](../quantum/eval/__init__.py) | ✅ Complete | Package marker. |

---

### `quantum/evaluation/` — Visualisation & Metrics (Block 7)

| File | Status | Description |
|---|---|---|
| [`metrics.py`](../quantum/evaluation/metrics.py) | ✅ Complete | `QuantumEvaluator.evaluate(y_true, y_pred)` — Accuracy, Precision, Recall, F1. `plot_confusion_matrix(y_true, y_pred, save_path)` — Seaborn heatmap. |
| [`visualizer.py`](../quantum/evaluation/visualizer.py) | ✅ Complete | `QuantumSpaceVisualizer.plot_tsne(statevectors, labels, class_names, save_path)` — 2D t-SNE scatter of VQC statevectors (concatenated real+imag parts). |

---

### `baselines.py` — Comparison Models (New)

| Class | Status | Description |
|---|---|---|
| `UntrainedProtoNet` | ✅ Complete | Classical nearest-centroid (Euclidean). No training needed. |
| `ScikitLearnBaseline` | ✅ Complete | Generic wrapper. Fits any sklearn model on the support set per episode. Instantiated for: Linear SVM, RBF SVM, kNN (k=1), Logistic Regression. |
| `get_classical_ml_baselines()` | ✅ Complete | Returns a `dict` of all 4 sklearn baseline instances. |
| `QuantumKernelBaseline` | ✅ Complete | QSVC with `FidelityQuantumKernel` (parameterized angle-encoding feature map). Uses Qiskit's `ComputeUncompute` fidelity estimator. |
| `TrainedProtoNet` | ✅ Complete | Classical `MLPEncoder` (Linear→ReLU→Linear) trained episodically using Snell et al. loss (squared Euclidean distances, CrossEntropy). `train_epoch(sampler, n_episodes)` runs training. |


---

### `experiments/` — Entry Points

| File | Status | Description |
|---|---|---|
| [`run_experiments.py`](../experiments/run_experiments.py) | ✅ Complete | Single QPN end-to-end pipeline. Phases: Load → Preprocess → QHBA → meta_train_qpn → evaluate → confusion matrix + t-SNE. Reads all hyperparameters from `config.yaml`. Uses a small fixed subset (1000 samples, 5 QHBA agents, 10 training episodes) for a quick demo run. |
| [`run_benchmarks.py`](../experiments/run_benchmarks.py) | ✅ Complete | Full head-to-head benchmark across 4 settings (5w1s, 5w5s, 10w1s, 10w5s). Trains QPN + TrainedProtoNet, evaluates all 6 models. Outputs `results/<timestamp>/benchmark_results.json` and `results/<timestamp>/RESULTS.md` with Wilcoxon p-values. |
| [`run_preprocessing.py`](../experiments/run_preprocessing.py) | ✅ Pre-existing | Standalone script to run preprocessing and cache the feature matrix. |
| [`visualize_data.py`](../experiments/visualize_data.py) | ✅ Pre-existing | Standalone data visualization script (class distribution, embeddings). |

---

### `config/config.yaml` — Hyperparameter Config

```yaml
data:
  dataset: "FastFit/trec_50"
  encoder_model: "all-MiniLM-L6-v2"
  n_features_lda: 32       # 384-dim → 32 LDA dims. Max possible = 49 (C-1).
  cache_dir: "data/cache"

quantum:
  n_qubits: 4              # QHBA selects 4 features from the 32 LDA features.
  shots: 1024              # Oracle measurement shots.

qhba:
  n_agents: 10             # Swarm size. Increase to 30+ for full research runs.
  max_iter: 30             # Iterations. Increase to 50+ for full research runs.
  c1: 0.5                  # Honey phase coefficient (exploitation).
  c2: 0.5                  # Badger phase coefficient (exploration).
  use_quantum_oracle: true

training:
  n_way: 5                 # Classes per episode during QPN meta-training.
  k_shot: 1                # Support examples per class.
  n_query: 15              # Query examples per class.
  n_episodes: 100          # Training episodes. Increase for better convergence.
  learning_rate: 0.01
  lr_step_size: 15
  lr_gamma: 0.5
  temperature: 1.0         # Initial softmax β (learned via log_beta parameter).

evaluation:
  n_way: 5
  k_shot: 1
  n_query: 15
  n_episodes: 300          # Test episodes for statistical significance.
```

---

## Known Issues & Caveats

### 🟢 Minor: `preprocessor.py` Typo
Line 38 has a typo (`prepr``jjocessing`) in the class docstring — harmless, cosmetic only.

---

## How to Run

```bash
# 1. Install dependencies (scipy now included)
pip install -r requirements.txt

# 2. (GPU VM only) Replace CPU Aer with GPU version
pip uninstall qiskit-aer && pip install qiskit-aer-gpu

# 3. Single QPN end-to-end run
python experiments/run_experiments.py

# 4. Full head-to-head benchmark
python experiments/run_benchmarks.py
```

> **Note on PSR compute cost:** The Parameter-Shift Rule runs `2 × n_params` Qiskit forward passes per training step backward pass. For `EfficientSU2` with 4 qubits and `reps=2`, this is `2 × 24 = 48` circuit evaluations per backward. On a GPU VM this is parallelisable — consider using `n_way=3, k_shot=1, n_query=5` for fast iteration during development.