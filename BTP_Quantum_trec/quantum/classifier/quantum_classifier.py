import numpy as np
from typing import List, Tuple

class QuantumPrototypicalClassifier:
    """
    Quantum Classifier block.
    Converts a list of quantum distances (infidelities) into a strict probability distribution
    over the classes using the Softmax function, and assigns the final prediction.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize the Quantum Prototypical Classifier.
        
        Args:
            temperature (float): Controls the 'sharpness' of the probability distribution.
                                 T=1.0 is standard softmax. 
                                 T < 1.0 makes it sharper (more confident).
                                 T > 1.0 makes it softer (flatter distribution).
        """
        self.temperature = temperature

    def classify(self, distances: List[float]) -> Tuple[int, np.ndarray]:
        """
        Takes the calculated distances (infidelities) to each class prototype
        and returns the predicted class ID along with the probability distribution.
        
        In Prototypical Networks, we apply softmax over the *negative* distances,
        so smaller distance = higher probability.
        
        Args:
            distances: A list or array of distance (infidelity) floats.
                       e.g., [dist_to_class_0, dist_to_class_1, ...]
                       
        Returns:
            Tuple[int, np.ndarray]: (predicted_class_id, array_of_probabilities)
        """
        distances = np.array(distances)
        
        # Softmax over negative distances scaled by temperature
        logits = -distances / self.temperature
        
        # Subtract max for numerical stability (avoids overflow during exp)
        logits_stable = logits - np.max(logits)
        
        exp_logits = np.exp(logits_stable)
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Assign to the closest prototype (highest probability)
        predicted_class_id = int(np.argmax(probabilities))
        
        return predicted_class_id, probabilities
