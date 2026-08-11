import sys
import os
import numpy as np

# Ensure root directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.trec_loader import TRECLoader
from quantum.feature_selection.qhba import QHBAConfig
from quantum.training.outer_loop import QPNMasterTrainer
from quantum.evaluation.metrics import QuantumEvaluator
from quantum.evaluation.visualizer import QuantumSpaceVisualizer
from quantum.training.qpn_model import QuantumPrototypicalNetwork

def main():
    print("=== Phase 1: Data Loading & Classical Preprocessing ===")
    loader = TRECLoader()
    # 1. Load TREC dataset (subset of 50 samples for speed)
    print("Loading TREC dataset (50 samples)...")
    # 1. Load TREC dataset (subset of 150 random samples to ensure enough classes for LDA)
    print("Loading TREC dataset (150 samples)...")
    dataset = loader.load_split(split='train')
    
    np.random.seed(42)
    indices = np.random.choice(len(dataset['text']), size=150, replace=False)
    texts = [dataset['text'][i] for i in indices]
    labels = [dataset['label'][i] for i in indices]
    
    y = np.array(labels, dtype=int)
    unique_classes = np.unique(y)
    # Ensure class_names can be indexed by any label from 0 to 49
    class_names = [f"Class_{i}" for i in range(50)]
    
    # LDA components cannot exceed min(n_features, n_classes - 1)
    n_lda = min(20, len(unique_classes) - 1)
    
    from data.preprocessor import TRECPreprocessor
    print(f"Running Classical Feature Extraction (Block 1) into {n_lda} dimensions...")
    preprocessor = TRECPreprocessor(
        encoder_model="all-MiniLM-L6-v2",
        n_features_lda=n_lda,
    )
    X = preprocessor.fit_transform(texts, y)

    print("\n=== Phase 2: Quantum Honey Badger Outer Loop (Block 2 & 8) ===")
    # Configure Fast-Run for QCHBA
    qpn_trainer = QPNMasterTrainer(
        n_features=X.shape[1],
        epochs_per_eval=10,  # Inner loop epochs
        n_qubits=8
    )
    # Patch the QHBA config for speed
    qpn_trainer.qhba.cfg = QHBAConfig(n_agents=10, max_iter=3, use_quantum_oracle=False)
    
    # Run the Two-Loop Optimization
    result = qpn_trainer.fit(X, y)
    
    print("\n=== Phase 3: Final Output & Evaluation (Block 7) ===")
    print(f"Best Feature Mask found by QCHBA: {result.selected_indices}")
    print(f"Best Fitness (Loss) achieved: {result.best_fitness:.4f}")
    
    # Simulate final evaluation with a mock prediction to test the metrics engine
    print("\nGenerating Research Metrics...")
    evaluator = QuantumEvaluator(class_names=class_names)
    
    # Mock some predictions for the 50 samples based on the labels to test the evaluator
    y_pred = y.copy()
    # Randomly flip a few to simulate imperfect accuracy
    flip_indices = np.random.choice(len(y), size=int(0.2 * len(y)), replace=False)
    y_pred[flip_indices] = np.random.choice(unique_classes, size=len(flip_indices))
    
    metrics = evaluator.evaluate(y, y_pred)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.4f}")
        
    print("\nGenerating Visualizations (Saved to disk)...")
    evaluator.plot_confusion_matrix(y, y_pred, save_path="confusion_matrix.png")
    
    # Test the visualizer with dummy statevectors
    visualizer = QuantumSpaceVisualizer()
    # Create some dummy 2D real arrays representing statevectors for the t-SNE plot
    dummy_statevectors = [np.random.rand(4) + 1j * np.random.rand(4) for _ in range(len(y))]
    visualizer.plot_tsne(dummy_statevectors, y, class_names, save_path="tsne_plot.png")
    
    print("\n=== Pipeline Execution Complete ===")

if __name__ == "__main__":
    main()
