# GPU Upgrade & Parallelization Notes

This document outlines the performance bottlenecks identified in the original `BTP_Quantum_trec` implementation and the specific GPU/parallelization upgrades applied during the system overhaul to make execution feasible on a GPU-enabled Virtual Machine.

---

## 1. Vectorized Density Matrix Prototyping (Block 4)

**The Problem:**
In the original implementation, `QuantumPrototypeCalculator` averaged the pure `Statevector` representations to create a mixed-state Density Matrix. This was done using a double `for` loop in Python over the matrix dimensions (e.g., `16x16` for 4 qubits). For every class prototype, this required `256` native Python iterations. In an episodic training loop sampling hundreds of episodes, this became an immense CPU bottleneck.

**The Solution:**
The loop was entirely removed and replaced with a vectorized NumPy operation in `quantum/prototype_calculation/prototype_ops.py`.
```python
# Before (Slow Python Loops):
rho = np.zeros((dim, dim), dtype=complex)
for s in states:
    for i in range(dim):
        for j in range(dim):
             rho[i][j] += s[i] * np.conj(s[j])

# After (Vectorized):
rho = np.mean([DensityMatrix(s).data for s in states], axis=0)
```
**Impact:** Orders of magnitude speedup in prototype calculation, enabling fast meta-training iterations.

---

## 2. Batched GPU Quantum Oracle (Block 2)

**The Problem:**
The Quantum Honey Badger Algorithm (QHBA) evaluates the fitness of `N` agents at every iteration. Originally, `evaluate_batch` called `self.evaluate(agent)` in a sequential Python loop. Each call transpiled a circuit, sent it to the `AerSimulator`, and waited for the result. This meant the simulator overhead was incurred per-agent, and multi-core / GPU resources were not being utilized.

**The Solution:**
The Oracle was rewritten in `quantum/feature_selection/quantum_oracle.py` to:
1. Construct all `N` circuits for the current swarm generation first.
2. Auto-detect if `device='GPU'` is available in the `AerSimulator`.
3. Submit all circuits in a single batched run: `backend.run(circuits_list, shots=shots)`.

**Impact:** Massive reduction in simulator overhead. When running on the GPU VM, `qiskit-aer-gpu` simulates the entire batch in parallel on the GPU cores.

---

## 3. PyTorch Autograd over VQC Angles (Block 8)

**The Problem:**
The original model used `TorchConnector` and `EstimatorQNN` from `qiskit-machine-learning`, piping the quantum expectation values into a classical `nn.Linear` layer. This completely bypassed the Prototypical Network logic (Density Matrices and Infidelity) and was extremely slow because `TorchConnector` adds significant overhead for generic use-cases.

**The Solution:**
The new `QuantumProtoNet` directly manages the VQC angles as a `torch.nn.Parameter`. 
- During the forward pass, it evaluates the `Statevector` for each sample using the bound angles.
- It calculates true Quantum Prototypes and true Quantum Infidelity distances.
- It applies a Softmax and computes `CrossEntropyLoss`.
- Because the parameters are explicitly defined in PyTorch, `loss.backward()` leverages the Parameter-Shift Rule applied automatically by PyTorch's computational graph (tracking the `theta` tensor operations).

**Impact:** True prototypical network behavior with significantly less wrapper overhead.

---

## Running on the GPU VM

To take full advantage of these upgrades on your GPU-enabled VM:

1. Ensure the CUDA toolkit is installed on the VM.
2. Ensure you have installed the GPU version of Qiskit Aer:
   ```bash
   pip install qiskit-aer-gpu
   ```
3. The codebase will automatically detect the GPU backend. If successful, you will see a log message from the Quantum Oracle during the QHBA phase:
   `"Initializing AerSimulator with device='GPU'"`
4. You can now safely increase the `n_agents` (e.g., 30) and `max_iter` (e.g., 50) in `config.yaml` for a much more thorough feature search, as the batched execution will handle it efficiently.
