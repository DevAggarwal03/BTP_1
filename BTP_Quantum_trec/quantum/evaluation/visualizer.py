import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import List, Any

class QuantumSpaceVisualizer:
    """
    Projects high-dimensional quantum states down to 2D for visual proof
    of quantum advantage and class separation.
    """

    def __init__(self, random_state: int = 42):
        self.tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)

    def _statevector_to_real_array(self, statevectors: List[Any]) -> np.ndarray:
        """
        Converts Qiskit Statevectors (complex numbers) into a flat real array 
        by concatenating real and imaginary parts for t-SNE processing.
        """
        real_arrays = []
        for sv in statevectors:
            # Depending on if it's already an array or a Qiskit object
            data = np.asarray(sv)
            flat_real = np.concatenate([np.real(data), np.imag(data)])
            real_arrays.append(flat_real)
        return np.array(real_arrays)

    def plot_tsne(self, statevectors: List[Any], labels: List[int], class_names: List[str], save_path: str = None):
        """
        Reduces quantum embeddings to 2D using t-SNE and plots them.
        
        Args:
            statevectors: List of Qiskit Statevector objects from the VQC.
            labels: Integer labels for each state.
            class_names: String names for the classes.
            save_path: Optional path to save the plot.
        """
        # Convert complex quantum states to flat real arrays
        X = self._statevector_to_real_array(statevectors)
        
        # We need enough samples for t-SNE to work (perplexity requirement)
        if len(X) < self.tsne.perplexity:
            print(f"Warning: t-SNE requires more samples than perplexity. Adjusting perplexity to {max(1, len(X)-1)}.")
            self.tsne.set_params(perplexity=max(1, len(X)-1))
            
        # Fit and transform
        X_2d = self.tsne.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot for each class
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = np.where(labels == label)[0]
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=class_names[label], alpha=0.7, s=100)
            
        plt.title('t-SNE Visualization of Quantum VQC Embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"t-SNE plot saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
