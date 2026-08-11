import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List, Dict, Any

class QuantumEvaluator:
    """
    Calculates comprehensive classification metrics for research papers.
    """

    def __init__(self, class_names: List[str]):
        """
        Args:
            class_names: List of string names for the classes (e.g., ['LOCATION', 'SPORTS', ...])
        """
        self.class_names = class_names

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        """
        Computes Accuracy, Precision, Recall, and F1-Score.
        Uses 'macro' averaging to treat all classes equally, which is important for few-shot.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0)
        }
        return metrics

    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], save_path: str = None):
        """
        Plots and optionally saves a heatmap of the confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        
        plt.title('Quantum Prototypical Network Confusion Matrix')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
