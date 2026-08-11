from typing import Union
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity

class QuantumDistanceCalculator:
    """
    Quantum Distance Calculator (Fidelity Measurement).
    This class handles the measurement of 'distance' between a query's 
    pure quantum state and a class's mixed state prototype.
    """

    def __init__(self):
        pass

    def calculate_fidelity(
        self, 
        query_state: Union[Statevector, DensityMatrix], 
        prototype_state: Union[Statevector, DensityMatrix]
    ) -> float:
        """
        Calculates the Quantum Fidelity between a query state and a prototype state.
        
        Mathematically: F = ⟨ψ|ρ|ψ⟩ (if query is pure and prototype is mixed)
        or F(ρ1, ρ2) = (Tr(√(√ρ1 ρ2 √ρ1)))^2
        
        Args:
            query_state: The Qiskit Statevector or DensityMatrix of the query.
            prototype_state: The Qiskit DensityMatrix of the class prototype.
            
        Returns:
            float: The fidelity value between 0.0 and 1.0.
        """
        # Qiskit's state_fidelity handles pure-pure, pure-mixed, and mixed-mixed implicitly
        return state_fidelity(query_state, prototype_state)

    def calculate_distance(
        self, 
        query_state: Union[Statevector, DensityMatrix], 
        prototype_state: Union[Statevector, DensityMatrix]
    ) -> float:
        """
        Calculates the Quantum Distance (Infidelity) between a query state and a prototype state.
        Distance = 1 - Fidelity.
        
        Args:
            query_state: The Qiskit Statevector or DensityMatrix of the query.
            prototype_state: The Qiskit DensityMatrix of the class prototype.
            
        Returns:
            float: The distance value between 0.0 and 1.0.
        """
        fidelity = self.calculate_fidelity(query_state, prototype_state)
        return 1.0 - fidelity
