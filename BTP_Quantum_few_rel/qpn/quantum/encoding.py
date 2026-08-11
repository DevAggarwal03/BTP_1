# This module prepares classical features for the quantum pipeline by generating feature maps and scaling inputs.
# It wraps the Qiskit circuit construction needed for kernel-based learning.
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif

class QuantumFeaturePreprocessor:
    """
    Handles feature scaling to [-pi, pi] (for angle/zz) or padding/normalization (for amplitude).
    Must be fit ONLY on the support set or training pool to prevent data leakage.
    """
    def __init__(self, n_qubits: int, fm_kind: str = "zz"):
        self.n_qubits = min(n_qubits, 12) # Cap at 12 to keep statevector tractable
        self.fm_kind = fm_kind
        self.scaler = MinMaxScaler(feature_range=(-np.pi, np.pi)) if fm_kind != "amplitude" else None
        self.selected_indices = None
        self.pad_dim = 2 ** self.n_qubits if fm_kind == "amplitude" else self.n_qubits
        
    def fit(self, X: np.ndarray, y: np.ndarray, qchba_indices: np.ndarray = None):
        """
        Fits the funnel and scaler on the training data.
        """
        if qchba_indices is not None:
            X_subset = X[:, qchba_indices]
            
            if len(qchba_indices) == self.pad_dim:
                self.selected_indices = qchba_indices
            elif len(qchba_indices) > self.pad_dim:
                # Funnel down using F-test
                f_vals, _ = f_classif(X_subset, y)
                f_vals = np.nan_to_num(f_vals)
                top_k = np.argsort(f_vals)[-self.pad_dim:]
                self.selected_indices = qchba_indices[top_k]
            else:
                if self.fm_kind == "amplitude":
                    # For amplitude encoding, we just take all QCHBA indices and pad with zeros later.
                    self.selected_indices = qchba_indices
                else:
                    # QCHBA selected fewer features than n_qubits. Pad with top ANOVA features.
                    f_vals, _ = f_classif(X, y)
                    f_vals = np.nan_to_num(f_vals)
                    sorted_all = np.argsort(f_vals)[::-1]
                    remaining = [idx for idx in sorted_all if idx not in qchba_indices]
                    needed = self.pad_dim - len(qchba_indices)
                    additional_indices = np.array(remaining[:needed])
                    self.selected_indices = np.concatenate([qchba_indices, additional_indices])
        else:
            # Funnel from raw D features to pad_dim using F-test
            f_vals, _ = f_classif(X, y)
            f_vals = np.nan_to_num(f_vals)
            self.selected_indices = np.argsort(f_vals)[-self.pad_dim:]
            
        if self.fm_kind != "amplitude":
            X_selected = X[:, self.selected_indices]
            self.scaler.fit(X_selected)
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices is None:
            raise ValueError("Preprocessor not fitted.")
        X_selected = X[:, self.selected_indices]
        
        if self.fm_kind == "amplitude":
            padded = np.zeros((X_selected.shape[0], self.pad_dim))
            padded[:, :X_selected.shape[1]] = X_selected
            norms = np.linalg.norm(padded, axis=1, keepdims=True)
            norms[norms == 0] = 1.0 # prevent division by zero
            normalized = padded / norms
            return normalized
        else:
            return self.scaler.transform(X_selected)
        
    def fit_transform(self, X: np.ndarray, y: np.ndarray, qchba_indices: np.ndarray = None) -> np.ndarray:
        self.fit(X, y, qchba_indices)
        return self.transform(X)

def get_feature_map(kind: str, n_qubits: int, reps: int = 1) -> QuantumCircuit:
    """
    Returns a Qiskit QuantumCircuit for encoding classical data into quantum states.
    kind: 'angle', 'zz', or 'amplitude'
    """
    if kind == "angle":
        # One feature -> one RY rotation
        qc = QuantumCircuit(n_qubits)
        params = ParameterVector('x', n_qubits)
        for i in range(n_qubits):
            qc.ry(params[i], i)
        return qc
    elif kind == "zz":
        return ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement='linear')
    elif kind == "amplitude":
        # Amplitude encoding requires StatePreparation which binds dynamically.
        return None
    else:
        raise ValueError(f"Unknown feature map kind: {kind}")
