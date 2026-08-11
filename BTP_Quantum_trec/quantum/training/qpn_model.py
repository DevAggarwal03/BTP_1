import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from quantum.encoding.angle_encoding import AngleEncoder
from quantum.vqc.vqc_extractor import VQCFeatureExtractor

class QuantumPrototypicalNetwork(nn.Module):
    """
    Overarching PyTorch Module for the Quantum Prototypical Network.
    Combines the Angle Encoder (Block 1) and VQC (Block 3) into a single 
    trainable neural network layer using TorchConnector.
    """

    def __init__(self, num_qubits: int, ansatz_type: str = "EfficientSU2", reps: int = 2, num_classes: int = 50):
        super().__init__()
        self.num_qubits = num_qubits
        
        # 1. Initialize the Encoder
        self.encoder = AngleEncoder(n_qubits=num_qubits)
        encoding_circuit, input_params = self.encoder.build_parameterized()
        
        # 2. Initialize the VQC Ansatz
        self.vqc = VQCFeatureExtractor(num_qubits=num_qubits, ansatz_type=ansatz_type, reps=reps)
        ansatz_circuit = self.vqc.get_circuit()
        weight_params = ansatz_circuit.parameters
        
        # 3. Compose them (Snap the legos together)
        self.full_circuit = encoding_circuit.compose(ansatz_circuit)
        
        # 4. Create the QNN (Quantum Neural Network)
        # The EstimatorQNN calculates the expectation value, but for prototypical networks, 
        # we often need the raw state to calculate fidelity (Block 4/5). 
        # For simplicity in this PyTorch wrapper example, we wrap it as a standard QNN layer.
        # Note: In a full rigorous implementation of Block 4/5, you might bypass EstimatorQNN 
        # and use custom PyTorch autograd functions to calculate state fidelity directly.
        self.qnn = EstimatorQNN(
            circuit=self.full_circuit,
            input_params=input_params,
            weight_params=weight_params
        )
        
        # 5. Connect to PyTorch
        self.quantum_layer = TorchConnector(self.qnn)
        
        # 6. Map to classes (Structural shim for PyTorch CrossEntropyLoss)
        self.linear = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): The classical feature inputs from QCHBA.
        Returns:
            torch.Tensor: The output mapped to num_classes.
        """
        q_out = self.quantum_layer(x)
        return self.linear(q_out)
