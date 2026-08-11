from typing import List, Union
from qiskit.quantum_info import Statevector, DensityMatrix

class QuantumPrototypeCalculator:
    """
    Quantum Prototype Calculator (Few-shot class prototypes).
    This class handles the conversion of pure states (from VQC embeddings)
    into mixed state Density Matrices, which serve as the prototypes for classes.
    """

    def __init__(self, num_qubits: int):
        """
        Initialize the calculator.
        Args:
            num_qubits (int): Number of qubits used in the VQC.
                              Currently limited to 8 to avoid exponential memory blowup.
        """
        self.num_qubits = num_qubits
        if self.num_qubits > 8:
            # Simple warning/guard as per discussions (can be removed if tested)
            print(f"Warning: Calculating explicit {2**num_qubits}x{2**num_qubits} density matrix "
                  f"for {num_qubits} qubits might be computationally expensive.")

    def calculate_class_prototype(self, support_states: List[Union[Statevector, DensityMatrix]]) -> DensityMatrix:
        """
        Calculates the mixed state (Density Matrix) representing the class prototype
        by averaging the states of all samples in the support set.

        Args:
            support_states: A list of Qiskit Statevector or DensityMatrix objects 
                            representing the quantum embeddings of the support set for a specific class.

        Returns:
            DensityMatrix: The averaged mixed state `ρ_k = (1/|S_k|) Σ |ψ_i⟩⟨ψ_i|`.
        """
        if not support_states:
            raise ValueError("The support set cannot be empty.")

        # Initialize an empty density matrix of the correct dimension
        dim = 2 ** self.num_qubits
        mixed_state_data = [[0.0j for _ in range(dim)] for _ in range(dim)]
        
        # Accumulate the density matrices
        for state in support_states:
            # If the input is a pure Statevector, DensityMatrix(state) calculates |ψ⟩⟨ψ|
            dm = DensityMatrix(state)
            
            # Add element-wise
            for i in range(dim):
                for j in range(dim):
                    mixed_state_data[i][j] += dm.data[i][j]

        # Average the accumulated matrix
        num_samples = len(support_states)
        for i in range(dim):
            for j in range(dim):
                mixed_state_data[i][j] /= num_samples

        return DensityMatrix(mixed_state_data)
