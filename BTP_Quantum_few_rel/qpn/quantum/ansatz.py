# This module builds the parameterized quantum circuit ansatz used by the quantum encoders.
# It exposes helper functions for creating reusable circuit templates with configurable depth and initialization.
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import EfficientSU2

def create_ansatz(n_qubits: int, reps: int, init_type: str = "random") -> tuple[QuantumCircuit, np.ndarray]:
    """
    Creates a hardware-efficient ansatz (layers of RY/RZ + ring of CX).
    Returns the QuantumCircuit and the initial parameter values.
    
    init_type:
    - 'random': Small angle random initialization
    - 'identity_block': Grant et al. 2019 style initialization.
    - 'zeros': All zeros
    """
    ansatz = EfficientSU2(num_qubits=n_qubits, su2_gates=['ry', 'rz'], entanglement='circular', reps=reps)
    num_params = ansatz.num_parameters
    
    if init_type == "random":
        # Small angles around 0
        init_params = np.random.uniform(-0.1, 0.1, num_params)
    elif init_type == "identity_block":
        # If we set RY/RZ to 0, they are identity. 
        # But for Grant et al, the entangling blocks need to cancel out. 
        # We can approximate identity by setting parameters to 0. 
        init_params = np.zeros(num_params)
    elif init_type == "zeros":
        init_params = np.zeros(num_params)
    else:
        init_params = np.random.uniform(-np.pi, np.pi, num_params)
        
    return ansatz, init_params

class LayerwiseAnsatzManager:
    """
    Helper to manage layerwise training (Skolik et al. 2021).
    Allows freezing and unfreezing subsets of parameters corresponding to layers.
    """
    def __init__(self, ansatz: QuantumCircuit, reps: int):
        self.ansatz = ansatz
        self.reps = reps
        self.num_qubits = ansatz.num_qubits
        # In EfficientSU2 with ['ry', 'rz'], each layer has 2 parameters per qubit
        self.params_per_layer = self.num_qubits * 2 
        # Total layers = reps + 1 (the initial layer before the first entanglement)
        self.total_layers = reps + 1
        
    def get_layer_mask(self, active_layers: list[int]) -> np.ndarray:
        """
        Returns a boolean mask of which parameters are active (True) and frozen (False).
        active_layers: List of 0-indexed layers to train.
        """
        mask = np.zeros(self.ansatz.num_parameters, dtype=bool)
        for layer in active_layers:
            start_idx = layer * self.params_per_layer
            end_idx = start_idx + self.params_per_layer
            mask[start_idx:end_idx] = True
        return mask
