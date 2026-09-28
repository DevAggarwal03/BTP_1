# Bug Fix and Experiment Rerun Log

Date: 2026-09-28  
Scope: the explicitly requested historical PyTorch statevector implementation in `New_GO_implementation/`. This project remains separate from the Qiskit implementation described by the broader specification. Results below are from exact PyTorch statevector simulation and are not Qiskit or hardware results.

## Findings and corrections

| Finding | Correction | Files | Expected impact |
|---|---|---|---|
| The last trainable RY/RZ and CNOT block followed the final input encoding. It was therefore a common final unitary and canceled from every support/query fidelity. With two repetitions, the last block's `2 × n_qubits` parameters could not affect the classification objective. | Changed the circuit to initial input encoding followed by trainable rotations, circular entanglement, and input re-uploading for every repetition, including after the final trainable block. Reject zero repetitions. | `qpn_hybrid/quantum_encoder.py` | All configured trainable blocks are now input-conditioned in the overlap objective. This fixes an identifiability/training defect; accuracy impact will be measured below. |
| YAML files described the four/eight-qubit and episode settings, but launchers ignored them, allowing reported runs to silently diverge from those files. | Added safe YAML loading, validation for unknown keys, and dataclass construction. The smoke, multi-seed, baseline, and training-sweep launchers now load their configured architecture/protocol; command-line benchmark episode counts remain explicit overrides for the documented long runs. | `qpn_hybrid/config.py`, `experiments/train_main.py`, `experiments/run_multi_seed.py`, `experiments/run_baselines.py`, `experiments/run_training_sweep.py` | Each result includes the effective model, episode, simulator, optimizer, data-source, and split-size metadata. |
| The training-sweep default crossed both episode counts with all learning rates, which did not match the documented scheme (a 2,000-episode three-rate sweep, then a 5,000-episode run at the selected `0.001` rate). | Defaulted the sweep to 2,000 episodes; run the 5,000/`0.001` setting as a separate explicit command. | `experiments/run_training_sweep.py` | Avoids adding undocumented 5,000-episode settings and preserves comparability with the historical protocol. |
| Runners wrote checkpoints to fixed archived result paths; re-running the documented commands could overwrite the previous checkpoints/results. | Added configurable checkpoint directories and moved defaults under `results/reruns/`. This rerun uses its own dated results directory. | `experiments/train_main.py`, `experiments/run_multi_seed.py`, `experiments/run_baselines.py`, `experiments/run_training_sweep.py` | Existing archived results remain intact and the corrected run is separately reviewable. |
| NPZ loading used `zip(embeddings, labels)` without checking lengths, silently dropping trailing examples on a malformed cache. Non-finite and inconsistent-dimension vectors were not rejected. | Validate array ranks, sample counts, finite values, and dimensions before building relation pools; add compact relation/example-count summaries. | `qpn_hybrid/data.py` | Corrupt or incomplete caches fail before training rather than producing silently altered episode pools. |
| Classical baselines always selected nonlinear compression even if a config requested the linear option. | Pass the selected compression mode through to the baseline model. | `qpn_hybrid/baselines.py`, `experiments/run_baselines.py` | Baseline comparisons honor the same configured compression family. |
| Multi-seed config lookup assumed filenames such as `4_qubit.yaml`, while the repository uses the spelled-out names `four_qubit.yaml` and `eight_qubit.yaml`. | Added an explicit supported-qubit-to-config mapping and a clear error for unsupported sizes. | `experiments/run_multi_seed.py` | The first attempted launch exposed this mismatch; the corrected launch loaded both YAMLs and completed. |

## Rerun protocol

All runs use the cached 384-dimensional, frozen `all-MiniLM-L6-v2` embeddings, disjoint FewRel train/validation relations, and 5-way/1-shot/15-query episodes (80 examples per episode). Validation episode lists are fixed by seed and reused across architectures at the same seed. The quantum runs use the exact PyTorch statevector simulator with complex128 amplitudes and no shots.

Planned runs:

1. Main model: 4 and 8 qubits; seeds 7, 17, 27; 600 training and 600 validation episodes per setting.
2. Classical Euclidean and cosine ProtoNets: output dimension 8; seeds 7, 17, 27; 600 training and 600 validation episodes per setting.
3. Eight-qubit training sweep: seed 7; 2,000 training episodes at learning rates `0.0003`, `0.001`, and `0.003`; 600 validation episodes.
4. Eight-qubit longer run: seed 7; 5,000 training episodes at learning rate `0.001`; 600 validation episodes.

## Measured results

All outputs are under [`results/bug_fix_rerun_20260928/`](results/bug_fix_rerun_20260928/). The effective protocol was 5-way/1-shot/15-query, with the 64-relation/44,800-example train split and 16-relation/11,200-example validation split (700 examples per relation, 384 dimensions). Cached embeddings were frozen, unnormalized `all-MiniLM-L6-v2` vectors. The main and baseline runs used 600 training and 600 validation episodes per seed.

### Main model and matched classical baselines

Mean and standard deviation are across seeds 7, 17, and 27.

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Corrected quantum, 4 qubits | 42.51% ± 3.42% | 40.68% ± 3.15% |
| Corrected quantum, 8 qubits | 59.63% ± 1.42% | 56.78% ± 1.32% |
| Classical Euclidean ProtoNet, 8D | 59.29% ± 1.45% | 56.75% ± 1.49% |
| Classical cosine ProtoNet, 8D | 59.41% ± 0.85% | 56.20% ± 0.80% |

The 8-qubit result is close to its archived 59.34% accuracy; the 4-qubit result falls from the archived 53.71%. The corrected 8-qubit model is only 0.21 percentage points above cosine and 0.34 points above Euclidean across three seeds, which is not evidence of a meaningful advantage. The 4-qubit configuration remains substantially weaker. These corrected-circuit results replace the old circuit's results for future interpretation; they are not directly comparable as if only a training hyperparameter had changed.

Files: [`multi_seed_600_600.json`](results/bug_fix_rerun_20260928/multi_seed_600_600.json), [`classical_baselines_600_600.json`](results/bug_fix_rerun_20260928/classical_baselines_600_600.json). Each model checkpoint is in the corresponding `*_checkpoints/` directory.

### Eight-qubit training sweep (seed 7)

| Training episodes | Learning rate | Accuracy (600 validation episodes) | Weighted F1 | Runtime |
|---:|---:|---:|---:|---:|
| 2,000 | 0.0003 | 61.73% | 58.81% | 7.6 min |
| 2,000 | 0.001 | **63.78%** | **60.92%** | 9.2 min |
| 2,000 | 0.003 | 58.90% | 56.14% | 7.3 min |
| 5,000 | 0.001 | 62.81% | 60.10% | 17.5 min |

The best tested rate remains `0.001`; training for 5,000 episodes did not beat the 2,000-episode result on this fixed validation set. The `0.003` rate again performed worse. These are single-seed sweep results, not multi-seed evidence that `0.001` is globally optimal.

Files: [`training_sweep_2000.json`](results/bug_fix_rerun_20260928/training_sweep_2000.json), [`training_sweep_5000.json`](results/bug_fix_rerun_20260928/training_sweep_5000.json), with checkpoints in `training_sweep_checkpoints/`.

### Code checks and remaining verification gap

`compileall` completed successfully for `qpn_hybrid/` and `experiments/`, and `git diff --check` passed. The pytest suite was not run. Review found that the existing `test_quantum_encoder.py` uses the squared norm of normalized states as its gradient objective; that objective is identically one and cannot establish that trainable parameters affect fidelity. The finite-difference test checks a state-amplitude objective rather than the episodic classification loss required by the specification. This test-coverage weakness remains documented and the experiment results above should not be read as a substitute for that episode-level gradient check.

## Interpretation limits

This repair changes the implemented circuit schedule, so corrected-run numbers are not a direct continuation of the archived results. Comparisons with those historical Torch-simulator metrics are descriptive only. No quantum advantage or speedup is inferred from these simulator experiments.
