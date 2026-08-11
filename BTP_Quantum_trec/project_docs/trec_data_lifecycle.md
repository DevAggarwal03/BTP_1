# The Lifecycle of the TREC Data

This document summarizes exactly what physically happens to the TREC dataset when the complete Quantum Prototypical Network pipeline is executed.

1. **Data Loading**: Text questions and their category labels (e.g., "What is the capital of France?" $\rightarrow$ LOCATION) are loaded from the HuggingFace TREC dataset.
2. **Classical Preprocessing (Block 1)**: The text is converted into numbers (TF-IDF vectors). These vectors are mathematically compressed using PCA/LDA into a dense matrix of features ($X$) alongside their labels ($y$).
3. **The Outer Loop Starts (Block 2)**: The Master Trainer hands the data to the Quantum Honey Badger Algorithm (QCHBA). The Honey Badger says: *"Let's try using features #3, #7, #12... up to 8 features"*. It filters the dataset down to just those 8 columns.
4. **The Inner Loop Starts (Block 8)**: The filtered dataset is passed to the PyTorch Worker. 
5. **Quantum Encoding & VQC (Blocks 3,4,5,6)**: The PyTorch Worker pushes the 8 classical data points into the Angle Encoder, turning them into quantum states. The data flows through the VQC gates, creates Prototypes, measures Quantum Infidelity distances, and applies a Softmax to guess the categories.
6. **Loss Calculation & Quantum Gradients**: PyTorch looks at the model's guesses, calculates the error (Loss), and uses the Parameter-Shift rule to slightly twist the VQC angles so it guesses better next time. It repeats this for a few epochs.
7. **Feedback to Honey Badger**: PyTorch hands the final Loss score back to the Honey Badger. The Honey Badger uses this score to intelligently pick a *different* set of 8 features. Steps 4-7 repeat for the specified number of iterations.
8. **Final Evaluation (Block 7)**: Once the best features and best VQC angles are found, the data is pushed through one last time. The Evaluation Suite generates F1-Scores, plots the Confusion Matrix, and generates the 2D t-SNE plot of the quantum space.
