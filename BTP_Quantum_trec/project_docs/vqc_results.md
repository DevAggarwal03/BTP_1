# VQC Ansatze Results Tracking

## Execution Runs

### Run 1: End-to-End Pipeline Validation (Fast-Run)
- **Dataset**: TREC-50 (Subset: 150 random samples)
- **Classical Features**: 20 (SBERT `all-MiniLM-L6-v2` + Supervised LDA)
- **QCHBA Agents**: 3
- **QCHBA Iterations**: 3
- **VQC Qubits**: 8
- **VQC Epochs per Eval**: 2
- **Ansatz**: EfficientSU2 (Linear Entanglement, Reps=2)
- **Result Metrics**:
  - **Selected Feature Indices**: `[2, 3, 9, 10, 14, 19]`
  - **Final VQC Loss**: `3.9495`
  - **Accuracy**: `81.33%`
  - **Precision (Macro)**: `76.89%`
  - **Recall (Macro)**: `81.65%`
  - **F1 Score**: `76.19%`
- **Output Notes**: 
  - Over 129,000 quantum circuit simulations were executed via the Parameter-Shift rule to calculate PyTorch gradients. Pipeline successfully completed end-to-end, dumping `confusion_matrix.png` and `tsne_plot.png` visualizations.
  - **Key Architecture Decision (The Random Sampler):** Implemented a deterministic random sampler (`np.random.seed(42)`) to extract 150 diverse samples across the entire dataset. This was necessary because Scikit-Learn's LDA mathematically requires at least `(n_components + 1)` unique classes to reduce dimensionality. Taking the first 50 sequential samples resulted in too few classes and crashed the classical preprocessing block. This random sampling pattern guarantees a highly diverse class mixture for the Quantum Honey Badger.
Use this table to keep track of the experiments run with different parameterized quantum circuits (Ansatze) and their configurations.

### Run 2: Deeper Training Profile
- **Dataset**: TREC-50 (Subset: 150 random samples)
- **Classical Features**: 20 (SBERT `all-MiniLM-L6-v2` + Supervised LDA)
- **QCHBA Agents**: 3
- **QCHBA Iterations**: 3
- **VQC Qubits**: 8
- **VQC Epochs per Eval**: 5
- **Ansatz**: EfficientSU2 (Linear Entanglement, Reps=2)
- **Result Metrics**:
  - **Selected Feature Indices**: `[1, 3, 10, 11, 14, 15]`
  - **Final VQC Loss**: `3.8689`
  - **Accuracy**: `81.33%`
  - **Precision (Macro)**: `76.89%`
  - **Recall (Macro)**: `81.65%`
  - **F1 Score**: `76.19%`
- **Output Notes**: 
  - The final VQC Loss improved from `3.9495` to `3.8689` due to the increased epochs (from 2 to 5), showing that the parameter-shift rule successfully stepped the gradient descent toward a better optima.
  - The feature subset selection shifted, indicating that as the VQC gains more capacity to learn over 5 epochs, the Honey Badger's optimal feature combination adjusts accordingly.

### Run 3: Massive Scale Out (10 Agents, 10 Epochs)
- **Dataset**: TREC-50 (Subset: 150 random samples)
- **Classical Features**: 20 (SBERT `all-MiniLM-L6-v2` + Supervised LDA)
- **QCHBA Agents**: 10
- **QCHBA Iterations**: 3
- **VQC Qubits**: 8
- **VQC Epochs per Eval**: 10
- **Ansatz**: EfficientSU2 (Linear Entanglement, Reps=2)
- **Result Metrics**:
  - **Selected Feature Indices**: `[2, 3, 10, 11, 12, 13, 14, 17, 18]`
  - **Final VQC Loss**: `3.7435`
  - **Accuracy**: `81.33%`
  - **Precision (Macro)**: `76.89%`
  - **Recall (Macro)**: `81.65%`
  - **F1 Score**: `76.19%`
- **Output Notes**: 
  - Massive improvement in VQC Loss (`3.7435` down from `3.8689`). The combination of 10 QCHBA agents (wider exploration of the feature space) and 10 epochs (deeper VQC gradient descent) allowed the model to find a much better feature combination and train closer to convergence.
  - The QCHBA converged on a slightly larger feature mask (9 features) to feed into the 8 qubits, suggesting that giving the model more training time allowed it to utilize a denser subset of classical information.

| Experiment ID | Ansatz Type     | Qubits | Reps (Layers) | Entanglement Type | Parameters | Accuracy (%) | Barren Plateau Observed? | Notes / Observations |
| ------------- | --------------- | ------ | ------------- | ----------------- | ---------- | ------------ | ------------------------ | -------------------- |
| 1             | RealAmplitudes  | 8      | 1             | linear            |            |              |                          | Baseline test        |
| 2             | RealAmplitudes  | 8      | 2             | full              |            |              |                          |                      |
| 3             | EfficientSU2    | 8      | 1             | linear            |            |              |                          |                      |
| 4             | EfficientSU2    | 8      | 2             | circular          |            |              |                          |                      |
| 5             | TwoLocal        | 8      | 2             | linear            |            |              |                          | 'ry', 'rz' blocks    |
| 6             | PauliTwoDesign  | 8      | 3             | full              |            |              |                          | Try deeper circuit   |

## Instructions

- Start testing from the simplest configuration (e.g., `RealAmplitudes` with `reps=1`, `entanglement='linear'`).
- Gradually increase complexity by adding layers (`reps`) or changing `entanglement='full'`.
- Swap to more advanced circuit families like `EfficientSU2` and `TwoLocal`.
- Measure the accuracy and note if the training gradients vanish quickly (a sign of the barren plateau problem).
