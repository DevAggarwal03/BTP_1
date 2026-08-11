# Unexplored Ideas & Future Optimizations

This document lists all the unexplored paths, alternative techniques, and scale-up strategies discovered during the implementation of the Quantum Prototypical Network pipeline. These can be used as references for future testing, ablation studies, or extending the research paper.

## 1. Prototype Calculation without Explicit Density Matrices
- **Current Approach**: We simulate pure `Statevector`s on classical hardware and explicitly calculate the $2^N \times 2^N$ `DensityMatrix` for the class prototypes. This is computationally feasible because we cap the qubits at $N=8$.
- **Unexplored Alternative**: For $N > 8$ qubits, storing a $2^N \times 2^N$ matrix becomes memory-prohibitive. We could avoid calculating the explicit density matrix entirely by computing the **Fidelity** (distance) between a query state and the mixed state prototype directly on the quantum circuit using a **SWAP Test** or **Compute-Uncompute** method.

## 2. Advanced VQC Ansatz Exploration
- **Current Approach**: The pipeline uses standard Qiskit circuit templates (`EfficientSU2`, `RealAmplitudes`, etc.) with basic entanglement topologies.
- **Unexplored Alternative**: 
    - Testing **Hardware-Efficient Ansatze** tailored to specific physical QPU topologies (e.g., IBM heavy-hex).
    - Testing data-reuploading circuits where the quantum features are encoded repeatedly between the parameterized layers to increase expressivity.

## 3. Alternative Data Encoding
- **Current Approach**: We select quantum features using QCHBA and encode them.
- **Unexplored Alternative**: Implementing **Amplitude Encoding** (which allows encoding $2^N$ classical features into $N$ qubits, massively increasing the feature space) instead of simple Angle Encoding.

## 4. Scaling the Dataset
- **Current Approach**: TREC-50 text dataset.
- **Unexplored Alternative**: Testing the pipeline on the `few-rel` dataset (as the other teammate is doing) to observe if the Quantum advantage holds across different NLP tasks, or expanding to larger, more complex text domains.

## 5. Noise and Error Mitigation
- **Current Approach**: Pure state simulations (AerSimulator without noise models).
- **Unexplored Alternative**: Injecting realistic quantum noise (depolarizing, thermal relaxation) and testing Error Mitigation strategies (like Zero Noise Extrapolation) to see how the Quantum Prototypical Network degrades on actual near-term hardware.

## 6. Physical Distance Measurement Circuits
- **Current Approach**: Because we simulate up to 8 qubits, we compute Quantum Infidelity directly via Qiskit's `state_fidelity` mathematical function.
- **Unexplored Alternative**: If we run this on actual QPU hardware, we cannot directly inspect the matrices. We must implement a **SWAP Test Circuit** or a **Compute-Uncompute (Inversion) Circuit**. Testing these physical circuits in simulation (and seeing how shot-noise affects the fidelity estimation) is a critical unexplored step before physical deployment.

## 7. Trace Distance vs Infidelity
- **Current Approach**: We use Infidelity ($1 - F$) as our Bregman divergence analog.
- **Unexplored Alternative**: Testing **Trace Distance**, which is another valid metric for distinguishability between quantum states. It is harder to measure on hardware but provides different theoretical guarantees.

## 8. Softmax Temperature Tuning
- **Current Approach**: We use a standard Softmax with a default Temperature $T=1.0$ over the quantum infidelities.
- **Unexplored Alternative**: Treating the Temperature $T$ as a learnable parameter during the training loop, or setting up a grid search to find the optimal temperature for smoothing the quantum gradients.

## 9. Classical vs Quantum Visual Overlays
- **Current Approach**: We use t-SNE to project and visualize the quantum embeddings in 2D space.
- **Unexplored Alternative**: Generating a side-by-side or overlaid t-SNE plot showing the classical LDA features (Block 1) versus the Quantum VQC features (Block 3) for the exact same samples. This is a very powerful visual argument for research papers to prove that the quantum circuit achieves *better* class separation than the classical baseline.

## 10. Barren Plateau Mitigation (Training)
- **Current Approach**: Using standard PyTorch Adam optimizer over the global Cross-Entropy loss.
- **Unexplored Alternative**: As VQC depth increases, gradients vanish (Barren Plateaus). Testing "Local Cost Functions" instead of global fidelity, or using "Layer-wise Training" (training the VQC one layer at a time) to avoid barren plateaus in deeper ansatze.

## 11. Early-Stopping and Transfer Learning in the Master Loop
- **Current Approach**: The QCHBA outer-loop evaluates fitness by training the VQC for a very low number of epochs (e.g., 2-3) starting from random weights every time.
- **Unexplored Alternative**: Implementing "Transfer Learning" between QHBA iterations. If the new feature mask is similar to the previous best feature mask, initialize the PyTorch inner-loop using the *previously trained VQC angles* rather than randomizing them. This could drastically speed up convergence during the outer-loop search.

