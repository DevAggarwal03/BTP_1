# Full Pipeline Flow — Step by Step

A concrete, example-driven walkthrough of everything that happens from the moment you run the pipeline to the moment you get your research results.

**Entry point:** `python experiments/run_experiments.py`

---

## Stage 0: Setup

The script loads `config/config.yaml` and seeds all random number generators for reproducibility (`numpy`, `torch`, `random`).

```
Config loaded:
  n_qubits         = 8
  n_features_lda   = 32
  QHBA agents      = 10, iterations = 30
  Training: 5-way 1-shot, 100 episodes
  Evaluation: 5-way 1-shot, 300 episodes
```

---

## Stage 1 — Data Loading (Block 1A)

**File:** `data/trec_loader.py`

The `TRECLoader` downloads the **TREC-50** dataset from HuggingFace (`FastFit/trec_50`) on first run. Subsequent runs load from the local cache in `data/cache/`.

TREC-50 is a fine-grained question classification dataset. It has:
- **5,452 training** questions
- **500 test** questions
- **50 fine-grained classes** (e.g., "Entity: Animal", "Numeric: Date", "Location: City")
- **6 coarse categories**: Abbreviation, Description, Entity, Human, Location, Numeric

**Example raw data:**
```
Text:  "What country produces the most iron ore?"
Label: "Location: Country."  →  encoded as integer 33
```

All 50 string labels are mapped to integers 0–49 using a fixed alphabetical lookup.

```
Train samples : 5,452
Test  samples : 500
```

---

## Stage 2 — Classical Preprocessing (Block 1B)

**File:** `data/preprocessor.py`

The raw text questions cannot be fed into a quantum circuit directly. This stage converts them into fixed-size numerical feature vectors in three steps:

### Step 2a: Sentence Embedding (384 dimensions)

The `all-MiniLM-L6-v2` SentenceTransformer model encodes each question into a 384-dimensional dense vector capturing semantic meaning.

```
Input:   "What country produces the most iron ore?"
Output:  [0.023, -0.11, 0.045, ..., 0.098]   # shape: (384,)
```

This takes ~1–2 minutes on CPU for the full 5,452 training questions.

### Step 2b: Min-Max Normalization

All 384 features are scaled to the range `[0, 1]` so they are compatible with quantum angle encoding (which expects values in `[0, 1]` to map to rotation angles in `[0, π]`).

### Step 2c: LDA Dimensionality Reduction (384 → 32)

Linear Discriminant Analysis (LDA) is a **supervised** dimensionality reduction — it finds the 32 directions in the 384-dim space that best separate the 50 classes. This is better than PCA (which is unsupervised and ignores class labels) for a classification task.

```
Input:  (5452, 384)
Output: (5452, 32)   # 32 class-discriminative features
```

After LDA, the features are re-normalized to `[0, 1]`. LDA variance retained: ~91%.

---

## Stage 3 — Quantum Feature Selection: QHBA (Block 2)

**Files:** `quantum/feature_selection/qhba.py`, `honey_badger_ops.py`, `quantum_oracle.py`

We now have 32 features per question, but our quantum circuit only has **8 qubits** — meaning it can only process 8 features at a time. The **Quantum Honey Badger Algorithm (QHBA)** finds the best 8 out of 32.

### How QHBA works:

QHBA is a **swarm intelligence** metaheuristic. 10 agents (like virtual honey badgers) each hold a continuous position vector in `[0,1]^32` representing how strongly each feature is selected.

**Each iteration:**

1. **Smell Intensity** — Each agent computes how attracted it is to the current global best.
2. **Honey Phase (50%)** — Exploitation: move toward the best known feature mask.
3. **Badger Phase (50%)** — Exploration: make aggressive random moves to find better masks.
4. **Fitness Evaluation** — Score each agent's feature mask using a **hybrid fitness**:
   - **Quantum fitness (50%)**: Angle-encode the 8-feature position → shallow RealAmplitudes ansatz → measure → `1 - P(|00000000⟩)`. This rewards feature configurations that produce concentrated quantum measurement distributions.
   - **Classical fitness (50%)**: KNN cross-validation accuracy on the selected features. `(1 - accuracy) + 0.01 × (n_selected / 32)`.

**After 30 iterations:**

```
Best feature mask found: [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, ...]
Selected feature indices: [1, 4, 7, 9, 13, 15, 22, 27]   # 8 features
Best fitness: 0.183
QHBA runtime: ~45 seconds (GPU) / ~8 minutes (CPU)
```

The 8 selected feature indices represent the LDA dimensions that are most discriminative for the 50 classes.

---

## Stage 4 — Episodic Meta-Training (Block 8 + New)

**Files:** `quantum/training/qpn_model.py`, `quantum/eval/train.py`, `data/episode_sampler.py`

This is the **heart of the Prototypical Network**. The VQC's internal angles (`θ`) are trained episodically over 100 training episodes.

### What is an episode?

An episode is a **mini few-shot task** sampled from the training data. For 5-way 1-shot:

```
Random 5 classes selected: 
  Class 0 → "Entity: Animal"
  Class 1 → "Location: Country"
  Class 2 → "Numeric: Date"
  Class 3 → "Human: Individual"
  Class 4 → "Description: Reason"

Support Set (1 example per class → 5 examples total):
  Class 0: "What is the fastest land animal?"         → features: [0.23, 0.11, ...]
  Class 1: "What country has the most Nobel prizes?"  → features: [0.67, 0.03, ...]
  Class 2: "When did World War II end?"               → features: [0.44, 0.89, ...]
  Class 3: "Who invented the telephone?"              → features: [0.12, 0.55, ...]
  Class 4: "Why does ice float on water?"             → features: [0.78, 0.21, ...]

Query Set (15 examples per class → 75 examples to classify):
  "What animal lives the longest?"   → true label: Class 0 (Animal)
  "Which continent is Brazil on?"    → true label: Class 1 (Country)
  ...  (73 more queries)
```

### What happens inside each episode:

#### Step 4a: Angle Encoding
Each of the 5 support features `∈ [0,1]^8` is mapped to 8 qubit rotations:

```
Feature x = [0.23, 0.67, 0.12, ...]
Circuit:   RY(π·0.23) on qubit 0
           RY(π·0.67) on qubit 1
           RY(π·0.12) on qubit 2
           ...
```

This produces a product state `|ψ(x)⟩ = |ψ₀⟩ ⊗ |ψ₁⟩ ⊗ ... ⊗ |ψ₇⟩`.

#### Step 4b: VQC Feature Extraction
The `EfficientSU2` ansatz (2 layers, 8 qubits, 48 trainable parameters `θ`) is applied to the encoded state, creating entanglement between qubits and producing a rich quantum embedding:

```
|ψ_out(x; θ)⟩  =  VQC(θ) · |ψ_encoded(x)⟩
                =  256-dimensional complex state vector
```

#### Step 4c: Prototype Calculation (Block 4)
For each class, all K support quantum states are averaged into a **Density Matrix prototype**:

```
Class 0 (Animal) support state: |ψ_animal⟩
ρ_animal = |ψ_animal⟩⟨ψ_animal|     # (256×256 complex matrix)

With K=1, ρ_k = pure state density matrix.
With K=5, ρ_k = (1/5) Σᵢ |ψᵢ⟩⟨ψᵢ|   # mixed state — captures class variance
```

#### Step 4d: Distance Measurement (Block 5)
For each of the 75 query states, we compute the **Quantum Infidelity** to each of the 5 prototypes:

```
Query: "What animal lives the longest?"  →  |ψ_query⟩

Distance to ρ_animal   = 1 - F(|ψ_query⟩, ρ_animal)   = 0.08  ← small (good match)
Distance to ρ_country  = 1 - F(|ψ_query⟩, ρ_country)  = 0.91
Distance to ρ_date     = 1 - F(|ψ_query⟩, ρ_date)     = 0.87
Distance to ρ_human    = 1 - F(|ψ_query⟩, ρ_human)    = 0.79
Distance to ρ_reason   = 1 - F(|ψ_query⟩, ρ_reason)   = 0.83
```

Where `F(|ψ⟩, ρ) = ⟨ψ|ρ|ψ⟩` is the quantum state fidelity.

#### Step 4e: Quantum Classifier (Block 6)
Softmax is applied over the **negative** distances scaled by a learnable temperature `β`:

```
logits = β × [-0.08, -0.91, -0.87, -0.79, -0.83]

Softmax → probabilities = [0.88, 0.02, 0.03, 0.04, 0.03]

Predicted class: 0 (Animal) ✓
```

#### Step 4f: Loss & Gradient (Block 8)
CrossEntropyLoss is computed over all 75 query predictions in the episode:

```
Loss = CrossEntropy(logits_matrix[75×5], true_labels[75])
     = 0.31   (example value for a well-trained model)
```

Gradients are computed via the **Parameter-Shift Rule**. For each of the 48 VQC parameters `θ_k`:

```
∂Loss/∂θ_k = 0.5 × [Loss(θ_k + π/2) - Loss(θ_k - π/2)]
```

This requires **2 × 48 = 96 additional full circuit evaluations** per training step. The Adam optimizer then updates all 48 angles.

### Training progress across 100 episodes:

```
Episode   1/100 — Loss: 3.91
Episode  10/100 — Loss: 2.87
Episode  25/100 — Loss: 1.94
Episode  50/100 — Loss: 1.22
Episode  75/100 — Loss: 0.89
Episode 100/100 — Loss: 0.71
=> Meta-Training Complete. Avg Loss: 1.63
```

---

## Stage 5 — Episodic Evaluation (Block 7)

**Files:** `quantum/eval/harness.py`, `quantum/evaluation/metrics.py`

After training, the model is evaluated on **300 test episodes** sampled from the **test classes** (disjoint from training classes). No gradient updates happen here — the VQC angles are frozen.

### Per episode (example):
```
Test Episode 1:
  Classes: "Numeric: Speed", "Entity: Food", "Human: Group", "Abbreviation: Abbreviation", "Location: Mountain"
  Query accuracy: 72%

Test Episode 2:
  Classes: "Entity: Color", "Numeric: Weight", ...
  Query accuracy: 68%

...
```

### Final aggregated result:
```
====================================
     EVALUATION RESULTS (5w1s)
====================================
  Episodes        : 300
  Mean Accuracy   : 71.4% ± 2.1%  (95% CI)
  Weighted F1     : 69.8% ± 2.3%
====================================
```

---

## Stage 6 — Visualization Output (Block 7)

**Files:** `quantum/evaluation/metrics.py`, `quantum/evaluation/visualizer.py`

Two publication-ready plots are generated:

### Confusion Matrix (`confusion_matrix.png`)
A heatmap of shape `(50×50)` (or N×N for N test classes) showing where the model is getting confused. High values on the diagonal = good classification. Off-diagonal blobs reveal systematic confusions (e.g., confusing "Location: City" with "Location: Country").

### t-SNE Embedding Plot (`tsne_plot.png`)
The 256-dimensional complex VQC output statevectors are flattened into 512-dimensional real vectors (real + imaginary parts concatenated) and projected to 2D using t-SNE.

**What you want to see:** Tight, well-separated clusters, one per class. This visually demonstrates that the VQC has learned to encode semantically different question types into different regions of the quantum state space — the key "quantum advantage" figure for the research paper.

---

## End-to-End Summary

```
Raw TREC-50 Questions (5,452 samples, 50 classes)
              ↓  [SBERT: all-MiniLM-L6-v2]
Dense Embeddings (5452, 384)
              ↓  [MinMax Normalization]
Normalized Embeddings (5452, 384) ∈ [0, 1]
              ↓  [Supervised LDA]
Discriminative Features (5452, 32)
              ↓  [QHBA — 10 agents, 30 iterations, Hybrid Fitness]
Optimal 8 Feature Indices (e.g., [1, 4, 7, 9, 13, 15, 22, 27])
              ↓  [Episodic Sampler — 5-way 1-shot]
100 Training Episodes (each: 5 support + 75 queries)
              ↓  [Angle Encoding → VQC(θ) → Density Matrix Prototypes → Infidelity → Softmax]
              ↓  [Parameter-Shift Gradients → Adam Optimizer]
Trained VQC (48 optimized angles θ*)
              ↓  [300 Test Episodes — disjoint classes]
              ↓  [Inference: no gradients, frozen θ*]
Research Results:
  ✓ Mean Accuracy ± 95% CI
  ✓ Weighted F1 Score
  ✓ Confusion Matrix (confusion_matrix.png)
  ✓ t-SNE Embedding Plot (tsne_plot.png)
```
