# Quantum Classifier Block

## What does this block do?

After **Block 5**, we know the "Quantum Infidelity" (distance) between our query and every single class prototype. 
For example, if we have 3 classes, we might have distances like:
- Distance to Class 0: `0.1`
- Distance to Class 1: `0.8`
- Distance to Class 2: `0.9`

The **Quantum Classifier** (Block 6) takes these raw distances and turns them into clean **probabilities** (percentages that add up to 100%).

It does this using a math function called **Softmax**. 
Because we want the *smallest* distance to have the *highest* probability, we apply Softmax to the *negative* distances.

### The Temperature Parameter ($T$)
We have included a customizable "Temperature" parameter. 
- A standard temperature ($T=1.0$) gives normal probabilities.
- A **low temperature** (e.g., $T=0.1$) makes the classifier act extremely confident. It will heavily boost the probability of the closest class, making it look almost like a 100% certainty.
- A **high temperature** (e.g., $T=5.0$) makes the classifier very unsure. It flattens the probabilities, making them all closer to each other.

### The Output
The classifier outputs two things:
1. **The Predicted Class**: The ID of the class with the highest probability.
2. **The Probabilities Array**: The full list of percentages. (This is extremely important because we will feed these probabilities into the "Loss Function" in the final Training Block to tell the VQC how to update its gates!).
