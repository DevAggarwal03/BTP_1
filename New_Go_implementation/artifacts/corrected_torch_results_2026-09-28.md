# Corrected PyTorch Statevector Results — 2026-09-28

**Result label:** `corrected-torch-statevector-2026-09-28`  
**Implementation:** historical `New_Go_Implementation` PyTorch statevector model, after the final data-re-upload schedule fix  
**Simulator:** exact PyTorch statevector simulation (`complex128` amplitudes), no shots  
**Episode protocol:** 5-way, 1-shot, 15 queries per class (80 examples per episode)  
**Embedding data:** frozen, unnormalized 384-dimensional `all-MiniLM-L6-v2` cache; 64 train relations and 16 disjoint validation relations, 700 examples per relation  
**Run date:** 2026-09-28

These are results from the corrected PyTorch implementation. They are not Qiskit or quantum-hardware results. The circuit schedule change means the quantum results should be interpreted as a corrected model version, not as a direct continuation of the archived circuit results.

## Main model and classical baseline runs

Each row is one seed-specific model run trained for 600 episodes and evaluated on 600 validation episodes. Values are percentages. The `±` values are 95% confidence intervals across that run's validation episodes; they are not confidence intervals across seeds.

| Run | Seed | Training / validation episodes | Accuracy ± 95% CI | Weighted F1 ± 95% CI |
|---|---:|---:|---:|---:|
| Quantum, 4 qubits | 7 | 600 / 600 | 46.38% ± 0.87% | 44.25% ± 0.86% |
| Quantum, 4 qubits | 17 | 600 / 600 | 39.90% ± 0.82% | 38.29% ± 0.81% |
| Quantum, 4 qubits | 27 | 600 / 600 | 41.25% ± 0.80% | 39.51% ± 0.79% |
| Quantum, 8 qubits | 7 | 600 / 600 | 60.00% ± 1.02% | 56.93% ± 1.08% |
| Quantum, 8 qubits | 17 | 600 / 600 | 60.82% ± 1.01% | 58.02% ± 1.08% |
| Quantum, 8 qubits | 27 | 600 / 600 | 58.06% ± 0.99% | 55.39% ± 1.03% |
| Classical Euclidean ProtoNet, 8D | 7 | 600 / 600 | 59.56% ± 0.99% | 57.04% ± 1.06% |
| Classical Euclidean ProtoNet, 8D | 17 | 600 / 600 | 60.58% ± 1.00% | 58.06% ± 1.07% |
| Classical Euclidean ProtoNet, 8D | 27 | 600 / 600 | 57.72% ± 0.97% | 55.13% ± 1.02% |
| Classical cosine ProtoNet, 8D | 7 | 600 / 600 | 60.39% ± 1.04% | 57.13% ± 1.11% |
| Classical cosine ProtoNet, 8D | 17 | 600 / 600 | 58.82% ± 1.01% | 55.80% ± 1.05% |
| Classical cosine ProtoNet, 8D | 27 | 600 / 600 | 59.02% ± 1.00% | 55.68% ± 1.06% |

## Eight-qubit training runs

Each row is one seed-7 run evaluated on the same 600-episode validation set. The 2,000-episode runs compare learning rates; the separate 5,000-episode run uses the selected `0.001` rate.

| Training episodes | Learning rate | Seed | Validation episodes | Accuracy ± 95% CI | Weighted F1 ± 95% CI | Runtime |
|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 0.0003 | 7 | 600 | 61.73% ± 0.97% | 58.81% ± 1.05% | 7.6 min |
| 2,000 | 0.001 | 7 | 600 | **63.78% ± 1.00%** | **60.92% ± 1.07%** | 9.2 min |
| 2,000 | 0.003 | 7 | 600 | 58.90% ± 0.99% | 56.14% ± 1.05% | 7.3 min |
| 5,000 | 0.001 | 7 | 600 | 62.81% ± 1.02% | 60.10% ± 1.09% | 17.5 min |

## How to read the results

- Random guessing in a 5-way episode is 20% accuracy.
- The 8-qubit runs are close to the classical baselines; this experiment does not establish a meaningful advantage for the quantum model.
- The 4-qubit runs are substantially weaker than the 8-qubit runs in this corrected implementation.
- The best single sweep result was 2,000 training episodes at learning rate `0.001`. The 5,000-episode run did not improve on it.
- The learning-rate sweep uses one seed, so it is a useful setting comparison, not evidence that `0.001` is optimal across seeds.

## Source result files

- [Main 4/8-qubit runs, seeds 7/17/27](../results/bug_fix_rerun_20260928/multi_seed_600_600.json)
- [Classical baselines, seeds 7/17/27](../results/bug_fix_rerun_20260928/classical_baselines_600_600.json)
- [2,000-episode learning-rate sweep](../results/bug_fix_rerun_20260928/training_sweep_2000.json)
- [5,000-episode confirmation](../results/bug_fix_rerun_20260928/training_sweep_5000.json)
- [Bug-fix and experiment log](../BUG_FIX_AND_EXPERIMENT_LOG.md)
