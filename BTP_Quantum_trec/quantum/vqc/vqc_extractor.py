from typing import Optional, Dict, Any
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, TwoLocal, PauliTwoDesign

class VQCFeatureExtractor:
    """
    Quantum Feature Extractor utilizing Variational Quantum Circuits (VQC) / QNN Embeddings.
    This module creates highly customizable parameterized quantum circuits (Ansatze)
    to allow for easy testing and comparison of different configurations.
    """

    def __init__(self, 
                 num_qubits: int, 
                 ansatz_type: str = "EfficientSU2", 
                 reps: int = 2, 
                 entanglement: str = "linear",
                 **kwargs):
        """
        Initialize the VQC Feature Extractor.
        
        Args:
            num_qubits (int): Number of qubits.
            ansatz_type (str): Type of the ansatz. Supported: 'EfficientSU2', 'RealAmplitudes', 'TwoLocal', 'PauliTwoDesign'.
            reps (int): Number of parameterized layers (repetitions).
            entanglement (str): Entanglement topology ('linear', 'full', 'circular', 'sca', etc.)
            kwargs: Additional arguments to pass to the specific Ansatz.
        """
        self.num_qubits = num_qubits
        self.ansatz_type = ansatz_type
        self.reps = reps
        self.entanglement = entanglement
        self.kwargs = kwargs
        self.circuit = self._build_ansatz()

    def _build_ansatz(self) -> QuantumCircuit:
        """Builds the selected parameterized quantum circuit."""
        if self.ansatz_type == "EfficientSU2":
            return EfficientSU2(num_qubits=self.num_qubits, reps=self.reps, entanglement=self.entanglement, **self.kwargs)
        elif self.ansatz_type == "RealAmplitudes":
            return RealAmplitudes(num_qubits=self.num_qubits, reps=self.reps, entanglement=self.entanglement, **self.kwargs)
        elif self.ansatz_type == "TwoLocal":
            # Provide sensible defaults for TwoLocal if not provided in kwargs
            rotation_blocks = self.kwargs.pop("rotation_blocks", ['ry', 'rz'])
            entanglement_blocks = self.kwargs.pop("entanglement_blocks", 'cz')
            return TwoLocal(num_qubits=self.num_qubits, rotation_blocks=rotation_blocks, 
                            entanglement_blocks=entanglement_blocks, reps=self.reps, 
                            entanglement=self.entanglement, **self.kwargs)
        elif self.ansatz_type == "PauliTwoDesign":
            return PauliTwoDesign(num_qubits=self.num_qubits, reps=self.reps, entanglement=self.entanglement, **self.kwargs)
        else:
            raise ValueError(f"Ansatz type '{self.ansatz_type}' is not supported yet.")

    def get_circuit(self) -> QuantumCircuit:
        """Returns the constructed Qiskit QuantumCircuit."""
        return self.circuit
        
    def get_num_parameters(self) -> int:
        """Returns the number of trainable parameters in the circuit."""
        return self.circuit.num_parameters
