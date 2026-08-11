### Currently Implemented(11-07-2026):

- Block 1 - Input Data & Preprocessing
    - File Reference: [data/trec_loader.py](./data/trec_loader.py), [data/preprocessor.py](./data/preprocessor.py), [config/config.yaml](./config/config.yaml)
    - Using the TREC-50 dataset as planned. It has 50 subcategories of questions.
    - Using SentenceTransformer for text embedding
    - Using Min-Max normalization to scale the features to [0, 1]
    - Using LDA for dimensionality reduction

- Block 2 - Quantum Feature Selection (QCHBA)
    - File Reference: [quantum/feature_selection/honey_badger_ops.py](./quantum/feature_selection/honey_badger_ops.py), [quantum/feature_selection/quantum_oracle.py](./quantum/feature_selection/quantum_oracle.py), [quantum/feature_selection/qhba.py](./quantum/feature_selection/qhba.py), [config/config.yaml](./config/config.yaml)
    - Using Hybrid approach, for feature selection and classification

- Block 3 - Variational Quantum Circuits (VQC) / Quantum Feature Extractor
    - File Reference: [quantum/vqc/vqc_extractor.py](../quantum/vqc/vqc_extractor.py), [project_docs/vqc_block_description.md](./vqc_block_description.md), [project_docs/vqc_results.md](./vqc_results.md)
    - Using highly customizable parameterized quantum circuits (Ansatze) leveraging Qiskit's circuit library (`RealAmplitudes`, `EfficientSU2`, `TwoLocal`, etc.).
    - Facilitates flexible experimentation with variable depth and entanglement configurations to find optimal embeddings and avoid barren plateaus.

- Block 4 - Quantum Prototype Calculation
    - File Reference: [quantum/prototype_calculation/prototype_ops.py](../quantum/prototype_calculation/prototype_ops.py), [project_docs/prototype_block_description.md](./prototype_block_description.md)
    - Calculates the mixed-state Density Matrix prototypes by averaging pure states from the VQC support set.
    - Operates under the current 8-qubit limit, allowing for direct density matrix calculation in classical simulation.

- Block 5 - Quantum Distance Measurement
    - File Reference: [quantum/distance_measurement/fidelity_ops.py](../quantum/distance_measurement/fidelity_ops.py), [project_docs/distance_block_description.md](./distance_block_description.md)
    - Replaces classical Euclidean distance by calculating Quantum Infidelity (1 - Fidelity) between query states and prototypes.
    - Uses mathematical fidelity calculation `state_fidelity` for efficient classical simulation, bypassing the need for a simulated SWAP test circuit.

- Block 6 - Quantum Classifier
    - File Reference: [quantum/classifier/quantum_classifier.py](../quantum/classifier/quantum_classifier.py), [project_docs/classifier_block_description.md](./classifier_block_description.md)
    - Assigns queries to the closest prototype using a Softmax function over the negative quantum distances (infidelities).
    - Includes a tunable Temperature scaling parameter for adjusting probability sharpness.

- Block 7 - Output & Evaluation Suite
    - File Reference: [quantum/evaluation/metrics.py](../quantum/evaluation/metrics.py), [quantum/evaluation/visualizer.py](../quantum/evaluation/visualizer.py), [project_docs/output_block_description.md](./output_block_description.md)
    - Replaces simple text output with a research-grade evaluation suite generating Precision, Recall, F1, Accuracy, and Confusion Matrices.
    - Includes a t-SNE visualizer to project complex quantum state vectors into 2D plots for publication-ready visual proof of class separation.

- Block 8 - Model Training & Optimization (Inner Loop)
    - File Reference: [quantum/training/qpn_model.py](../quantum/training/qpn_model.py), [quantum/training/trainer.py](../quantum/training/trainer.py), [project_docs/training_block_description.md](./training_block_description.md)
    - Overarching PyTorch module integrating the AngleEncoder and VQC using `qiskit_machine_learning`'s `TorchConnector`.
    - Handles the episodic meta-learning inner loop, taking Cross-Entropy loss and calculating Quantum Gradients to update VQC angles via Adam optimizer.

- Master Outer Loop (QCHBA Integration)
    - File Reference: [quantum/training/outer_loop.py](../quantum/training/outer_loop.py)
    - Replaces the mock KNN fitness function in the Quantum Honey Badger Algorithm with the actual PyTorch VQC inner loop.
    - Implements an Early-Stopping (low-epoch) fitness evaluator to make simulating the two-loop search computationally feasible.