# Episodic Meta-Training in the Quantum Prototypical Network

## Overview

This document describes how **episodic meta-training** is implemented in this project, why it is essential for a true Prototypical Network, and how it mirrors the approach used in `BTP_Quantum_few_rel`.

---

## 1. What Is Episodic Meta-Training?

Episodic meta-training is the core training strategy introduced in **Snell et al. (2017), "Prototypical Networks for Few-shot Learning" (NeurIPS)**. The key insight is:

> *"Train the model the same way you will test it."*

Instead of showing the model the full dataset in one big batch (as a standard classifier does), you train it by repeatedly solving thousands of small, randomly constructed **few-shot tasks** called **episodes**. Each episode forces the model to learn how to generalize from very few examples — which is exactly what it must do at test time.

---

## 2. What Is an Episode?

An episode is defined by two parameters:
- **N-way**: How many classes to distinguish (e.g., 5 classes at a time).
- **K-shot**: How many labeled examples are given per class (e.g., 1 or 5).

Each episode consists of:

| Component | Shape | Description |
|---|---|---|
| **Support Set** | `(N × K, D)` | The K labeled examples per class given as context. |
| **Query Set** | `(N × Q, D)` | The Q unlabeled examples to classify. |

The model must use the support set to build its understanding of the N classes, then correctly classify the query examples — all within that single episode. The episode is then discarded, and a new random one is sampled.

### Concrete Example (5-way 1-shot for TREC-50):
1. Randomly pick 5 out of 50 TREC classes (e.g., "Location: City", "Numeric: Date", "Entity: Animal", "Human: Individual", "Description: Reason").
2. Sample 1 support example per class → 5 examples total.
3. Sample 15 query examples per class → 75 examples total.
4. Train on this mini-task.
5. Repeat with 5 *different* random classes next time.

---

## 3. The Prototypical Network Forward Pass (Per Episode)

For each episode, the full quantum pipeline runs as:

```
Support Set (N×K samples)
        ↓
[Angle Encoding + VQC] ← parameterized by θ (the thing we train)
        ↓
N×K Quantum States (Statevectors)
        ↓
[Prototype Calculation (Block 4)]
Average Statevectors per class → N Density Matrix Prototypes ρ_1, ..., ρ_N
        ↓
Query Set (N×Q samples)
        ↓
[Angle Encoding + VQC] ← same θ
        ↓
N×Q Quantum States
        ↓
[Distance Measurement (Block 5)]
Quantum Infidelity: d(query_j, ρ_k) = 1 - F(|ψ_j⟩, ρ_k)
        ↓
[Quantum Classifier (Block 6)]
Softmax over negative distances → Probability distribution
        ↓
CrossEntropyLoss against ground-truth query labels
        ↓
Parameter-Shift Gradients → Adam Optimizer updates θ
```

---

## 4. The Full Meta-Training Loop

```python
# Pseudocode for meta_train_qpn()

model = QuantumProtoNet(n_qubits=8, ...)
optimizer = Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

for episode_idx in range(n_train_episodes):
    # 1. Sample a new N-way K-shot episode from the TRAINING data pool
    episode = sampler.sample()

    # 2. Encode features and convert to tensors
    s_x = torch.tensor(episode.support_x)   # (N*K, n_qubits)
    q_x = torch.tensor(episode.query_x)     # (N*Q, n_qubits)
    s_y = torch.tensor(episode.support_y)   # (N*K,)  — labels 0 to N-1
    q_y = torch.tensor(episode.query_y)     # (N*Q,)  — labels 0 to N-1

    # 3. Forward pass (uses real quantum prototypes + fidelity distances)
    optimizer.zero_grad()
    logits = model(s_x, s_y, q_x)           # (N*Q, N)

    # 4. Compute loss and backpropagate (Parameter-Shift Rule fires here)
    loss = CrossEntropyLoss(logits, q_y)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

---

## 5. The Full Evaluation Loop

Evaluation uses the same episodic structure but on **test data** (classes disjoint from training), with no gradient updates:

```python
# Pseudocode for evaluate()

model.eval()
accuracies = []

for episode_idx in range(n_test_episodes):
    episode = test_sampler.sample()
    
    # Predict class for each query example
    preds = model.predict(episode)     # (N*Q,) — predicted labels
    
    acc = accuracy_score(episode.query_y, preds)
    accuracies.append(acc)

mean_acc = np.mean(accuracies)
ci_95    = 1.96 * np.std(accuracies) / np.sqrt(n_test_episodes)
print(f"Accuracy: {mean_acc:.4f} ± {ci_95:.4f}")
```

---

## 6. How This Compares to `BTP_Quantum_few_rel`

Both implementations follow the same episodic meta-training protocol from Snell et al., ensuring valid cross-dataset comparison for the research paper.

| Aspect | `BTP_Quantum_few_rel` | `BTP_Quantum_trec` |
|---|---|---|
| **Dataset** | FewRel (relation classification) | TREC-50 (question classification) |
| **Preprocessing** | SBERT embeddings → QCHBA | SBERT → LDA → QCHBA |
| **Encoding** | Amplitude or ZZ Feature Map | Angle Encoding |
| **Feature Selection** | QCHBA (fits once on train pool) | QCHBA (outer loop, per feature search) |
| **Training Loop** | `meta_train_qpn()` in `qpn/eval/train.py` | `meta_train_qpn()` in `quantum/eval/train.py` |
| **Prototype Type** | Density Matrix centroid | Density Matrix centroid |
| **Distance** | Global/Local Fidelity (configurable) | Quantum Infidelity (1 - Fidelity) |
| **Evaluation** | `evaluate()` in `qpn/eval/harness.py` | `evaluate()` in `quantum/eval/harness.py` |
| **Settings** | 5w1s, 5w5s, 10w1s, 10w5s | 5w1s, 5w5s, 10w1s, 10w5s |

---

## 7. New Files Created

| File | Purpose |
|---|---|
| [`data/episode_sampler.py`](../data/episode_sampler.py) | `Episode` dataclass + `EpisodeSampler` for TREC-50 |
| [`quantum/eval/train.py`](../quantum/eval/train.py) | `meta_train_qpn()` — episodic training loop |
| [`quantum/eval/harness.py`](../quantum/eval/harness.py) | `evaluate()` — episodic evaluation with CI reporting |
| [`experiments/run_experiments.py`](../experiments/run_experiments.py) | Full research entry point: QHBA → Train → Evaluate |

---

## 8. Key Design Decisions

### Why are training and test classes disjoint?
This is the hallmark of proper few-shot evaluation. The model must generalize to *unseen* classes, not just recall memorized patterns. TREC-50 has 50 fine-grained classes. We use a fixed train/test class split to ensure fair evaluation.

### Why N-way K-shot and not full-dataset training?
The research goal is to test whether quantum prototypical networks can classify questions with only 1 or 5 examples per class. Full-dataset training defeats this purpose. The episodic structure ensures the model is evaluated under realistic few-shot constraints.

### Why StepLR scheduler?
The learning rate is decayed by `gamma=0.5` every `step_size=15` episodes. This allows aggressive early learning followed by fine-grained convergence — a common practice in prototypical network training.

### Why Parameter-Shift Rule?
Standard backpropagation cannot be applied directly to quantum circuits because quantum gates are not differentiable in the classical sense. The Parameter-Shift Rule provides exact gradient estimates by running the circuit twice with slightly shifted parameters: `∂f/∂θ_k = 0.5 * [f(θ_k + π/2) - f(θ_k - π/2)]`.

---

## 9. Running Episodic Training

```bash
# Quick smoke test (few episodes, small config)
python experiments/run_experiments.py --config config/config.yaml --fast

# Full research run (as configured in config.yaml)
python experiments/run_experiments.py --config config/config.yaml
```

Key config parameters (in `config/config.yaml`):

```yaml
training:
  n_way: 5
  k_shot: 1
  n_query: 15
  n_train_episodes: 100
  learning_rate: 0.01
  lr_step_size: 15
  lr_gamma: 0.5

evaluation:
  n_way: 5
  k_shot: 1
  n_query: 15
  n_test_episodes: 300
```
