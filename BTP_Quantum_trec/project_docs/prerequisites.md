# Prerequisites — BTP_Quantum_trec

This document lists everything you need to learn and understand to have **full working knowledge** of this project. It is structured from foundational concepts (learn first) to advanced topics (learn last), with a pointer to the best resource for each.

---

## Level 1 — Foundations (Learn These First)

These are the bedrock concepts. Everything else builds on top of them.

### 1.1 Linear Algebra Basics
You need to be comfortable with vectors, matrices, and complex numbers because quantum states are just complex vectors.

| What to learn | Why it matters here |
|---|---|
| Vectors, matrices, matrix multiplication | Density matrices are matrices; all quantum ops are matrix multiplications |
| Complex numbers (magnitude, phase, conjugate) | Quantum state amplitudes are complex |
| Eigenvalues / eigenvectors | Used in understanding quantum measurement |
| Inner product / dot product | Fidelity is an inner product between states |

📖 **Resource:** [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

---

### 1.2 Classical Machine Learning
| What to learn | Why it matters here |
|---|---|
| Classification problem setup (features, labels) | The whole point of TREC-50 |
| k-Nearest Neighbours (kNN) | Used inside QHBA fitness function |
| SVM (Support Vector Machine) | One of the baselines; also the backbone of QSVC |
| Logistic Regression | Another baseline model |
| Cross-Entropy Loss | The loss function used in all episodic training loops |
| Softmax function | Converts distances to probabilities in the classifier block |

📖 **Resource:** [StatQuest Machine Learning Playlist (YouTube)](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)  
📖 **Resource:** [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

### 1.3 Deep Learning Basics (PyTorch)
The QPN model is a `torch.nn.Module`. You need to understand PyTorch to follow the training loop.

| What to learn | Why it matters here |
|---|---|
| Tensors vs NumPy arrays | All model inputs/outputs are tensors |
| `nn.Module`, `nn.Parameter` | VQC angles `theta` are stored as `nn.Parameter` |
| Forward pass & backpropagation | `model(support_x, support_y, query_x)` → `loss.backward()` |
| Adam optimizer | Used in all training loops (`meta_train_qpn`, `TrainedProtoNet`) |
| Learning rate schedulers (StepLR) | Used to decay the LR during training |

📖 **Resource:** [PyTorch Official Tutorial — 60-minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

---

## Level 2 — NLP & Text Processing

### 2.1 Sentence Embeddings
The project converts raw text to fixed-size vectors using a pretrained language model.

| What to learn | Why it matters here |
|---|---|
| What sentence embeddings are | Block 1 uses `all-MiniLM-L6-v2` to encode text to 384-dim vectors |
| How pretrained transformers work (high-level) | Why we use this instead of TF-IDF or bag-of-words |
| `sentence-transformers` library usage | Used directly in `data/preprocessor.py` |

📖 **Resource:** [SBERT.net Documentation](https://www.sbert.net/docs/pretrained_models.html)  
📖 **Resource:** [HuggingFace Course — Chapter 1](https://huggingface.co/learn/nlp-course/chapter1/1)

---

### 2.2 Dimensionality Reduction
After embedding, LDA reduces 384-dim vectors to 32 class-discriminative features.

| What to learn | Why it matters here |
|---|---|
| PCA (conceptual) | Contrast with LDA to understand why LDA is preferred |
| Linear Discriminant Analysis (LDA) | Used in `TRECPreprocessor` for supervised dim-reduction |
| Min-Max Normalization | Applied twice: after embedding and after LDA |

📖 **Resource:** [StatQuest — LDA (YouTube)](https://www.youtube.com/watch?v=azXCzI57Yfc)  
📖 **Resource:** [Scikit-Learn LDA docs](https://scikit-learn.org/stable/modules/lda_qda.html)

---

## Level 3 — Quantum Computing

### 3.1 Quantum Computing Fundamentals
| What to learn | Why it matters here |
|---|---|
| Qubits: ∣0⟩, ∣1⟩, superposition | Basis of all quantum circuits in this project |
| Bloch sphere | Visual intuition for single-qubit states |
| Quantum gates: X, H, RY, CNOT | RY gates are used in angle encoding; CNOT in entanglement |
| Measurement & collapse | How the oracle's probability distribution is obtained |
| Circuit depth & width | Reason why we limit to 4 qubits by default (exponential memory cost: 2^n) |
| Entanglement | Used in the VQC ansatz; connects qubits for expressibility |

📖 **Resource:** [IBM Quantum Learning — Basics of Quantum Information](https://learning.quantum.ibm.com/course/basics-of-quantum-information)  
📖 **Resource:** [Qiskit Textbook (free)](https://qiskit.org/learn)

---

### 3.2 Density Matrices & Mixed States
**This is the most important quantum concept for this project.** The prototype in a quantum ProtoNet is a density matrix, not a pure state.

| What to learn | Why it matters here |
|---|---|
| Pure state vs mixed state | A single sample = pure state (Statevector); a prototype = mixed state (DensityMatrix) |
| Density matrix formalism: ρ = ∣ψ⟩⟨ψ∣ | How `prototype_ops.py` computes prototypes |
| Partial trace | Conceptual background (not directly implemented but important) |
| Quantum Fidelity: F(ρ, σ) | The core distance metric in Block 5 |

📖 **Resource:** [Nielsen & Chuang — Quantum Computation and Quantum Information, Chapter 2](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview)  
📖 **Resource:** [Qiskit's DensityMatrix documentation](https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.DensityMatrix)

---

### 3.3 Variational Quantum Circuits (VQC)
| What to learn | Why it matters here |
|---|---|
| Parameterized quantum circuits | The VQC ansatz has trainable rotation angles (theta) |
| Ansatz families: EfficientSU2, RealAmplitudes | The options available in `vqc_extractor.py` |
| Expressibility and entanglement capability | How to choose `reps` and `entanglement` topology |
| Parameter-Shift Rule (PSR) | **Critical:** This is how VQC gradients are computed classically. Currently NOT implemented — this is the biggest open issue. |

📖 **Resource:** [Pennylane Codebook — Variational Circuits](https://codebook.xanadu.ai/)  
📖 **Resource:** [Mitarai et al. (2018) — Parameter-Shift Rule paper](https://arxiv.org/abs/1803.00745)

---

### 3.4 Qiskit Framework
| What to learn | Why it matters here |
|---|---|
| Building circuits: `QuantumCircuit`, gates | Used throughout `angle_encoding.py`, `quantum_oracle.py` |
| `Statevector` and `DensityMatrix` classes | Core quantum info objects in Blocks 3–5 |
| `state_fidelity()` function | Used in `qpn_model.py` and `fidelity_ops.py` |
| Qiskit Aer: `AerSimulator` | Runs circuit simulations (CPU or GPU) |
| `ParameterVector` | Used in `vqc_extractor.py` for trainable circuits |
| Transpilation | `transpile()` is called before every `backend.run()` |

📖 **Resource:** [Qiskit Documentation](https://docs.quantum.ibm.com/)  
📖 **Resource:** [Qiskit Aer Documentation](https://qiskit.github.io/qiskit-aer/)

---

## Level 4 — Quantum Machine Learning

### 4.1 Quantum Kernel Methods
Needed to understand the `QuantumKernelBaseline` (QSVC).

| What to learn | Why it matters here |
|---|---|
| Kernel methods and the kernel trick | How SVM can work in high-dimensional feature spaces |
| Fidelity Quantum Kernel: K(x, x') = ∣⟨ψ(x)∣ψ(x')⟩∣² | Used in `baselines.py` for QSVC |
| `FidelityQuantumKernel` in Qiskit ML | The specific class used |

📖 **Resource:** [Schuld & Killoran (2019) — Quantum Feature Spaces](https://arxiv.org/abs/1803.07128)  
📖 **Resource:** [Qiskit ML: Quantum Kernels tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/03_quantum_kernel.html)

---

### 4.2 Quantum Honey Badger Algorithm (QHBA)
The outer-loop optimizer for feature selection.

| What to learn | Why it matters here |
|---|---|
| Swarm optimization (conceptual) | QHBA is a population-based optimizer |
| Honey Badger Algorithm (HBA) | The classical base; Honey Phase + Badger Phase |
| V-shaped transfer function for binarization | How continuous positions become binary feature masks |

📖 **Resource:** [QHBA original paper (cite from project bibliography)]  
📖 **Resource:** [Honey Badger Algorithm paper — Hashim et al. (2022)](https://www.sciencedirect.com/science/article/pii/S0957417421016602)

---

## Level 5 — Few-Shot Learning

### 5.1 Prototypical Networks
**The theoretical backbone of the entire model.**

| What to learn | Why it matters here |
|---|---|
| Few-shot learning problem setup (N-way K-shot) | The episode structure in `episode_sampler.py` |
| Episodic training methodology | Why we train on episodes, not batches of individual samples |
| Class prototype = mean of support embeddings | Classical ProtoNet; QPN replaces this with a density matrix |
| Euclidean distance → cross-entropy loss | Classical ProtoNet loss; QPN uses quantum infidelity instead |

📖 **Resource:** [Snell et al. (2017) — Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175) *(read this paper carefully — it is the direct blueprint for the entire model)*

---

### 5.2 Quantum Prototypical Networks
The specific QML adaptation of the above.

| What to learn | Why it matters here |
|---|---|
| Replacing Euclidean centroids with density matrix prototypes | The core of `prototype_ops.py` |
| Replacing Euclidean distance with quantum infidelity | The core of `fidelity_ops.py` and Block 5 |
| Training QPN with the Parameter-Shift Rule | **Critical open issue in this project** |

📖 **Resource:** [Nguyen et al. — Quantum Prototypical Networks (search arXiv for latest)](https://arxiv.org/search/?searchtype=all&query=quantum+prototypical+network)

---

## Level 6 — Statistics & Evaluation

### 6.1 Confidence Intervals
Used in `harness.py` to report `mean ± 95% CI`.

| What to learn | Why it matters here |
|---|---|
| Standard deviation vs standard error | `ci95 = 1.96 × (std / sqrt(n_episodes))` |
| 95% confidence interval formula | Directly used in `evaluate()` |

📖 **Resource:** [StatQuest — Confidence Intervals (YouTube)](https://www.youtube.com/watch?v=TqOeMYtOc1w)

---

### 6.2 Wilcoxon Signed-Rank Test
Used in `stats.py` to determine if QPN is statistically significantly better than baselines.

| What to learn | Why it matters here |
|---|---|
| Hypothesis testing (null hypothesis, p-value) | The paired test computes p-value for QPN vs each baseline |
| Wilcoxon signed-rank test (non-parametric) | Used because accuracy distributions are not guaranteed to be normal |
| Effect size | Reported alongside p-value in `paired_test()` |

📖 **Resource:** [StatQuest — Wilcoxon Signed-Rank Test (YouTube)](https://www.youtube.com/watch?v=VHx8HoqVHOs)

---

## Quick Reading Priority

If you're new and want to get productive fast, study in this order:

1. ⭐ **Snell et al. (2017)** — Prototypical Networks paper  
2. ⭐ **IBM Quantum Basics course** — Qubits, gates, circuits  
3. ⭐ **Density matrices** (Nielsen & Chuang Ch. 2)  
4. ⭐ **Qiskit Textbook** — Build and run your first circuit  
5. ⭐ **Parameter-Shift Rule** (Mitarai et al.) — Critical unresolved issue  
6. **PyTorch 60-minute Blitz** — If you don't know PyTorch  
7. **StatQuest ML Playlist** — If classical ML is fuzzy  
