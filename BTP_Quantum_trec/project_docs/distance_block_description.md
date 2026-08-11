# Quantum Distance Measurement Block

## What does this block do?

After we have encoded our query text into a **Query Quantum State** (using Blocks 1-3) and we have calculated our **Quantum Prototypes** for each category (using Block 4), we need a way to figure out which category the query belongs to. 

In classical AI, you measure the physical "Euclidean Distance" (like a ruler) between the query vector and the category center. The shortest distance wins.

In Quantum AI, we can't easily measure Euclidean distance. Instead, we measure **Quantum Fidelity**. 
- **Fidelity** is a measure of "overlap" or similarity between two quantum states. 
- It is a number between `0.0` (completely different) and `1.0` (identical).

Because machine learning algorithms usually want to *minimize a distance* rather than *maximize a similarity*, we simply flip it! We use **Infidelity (1 - Fidelity)** as our quantum distance metric.

- **Fidelity** = `1.0` means **Distance** = `0.0` (Perfect match)
- **Fidelity** = `0.0` means **Distance** = `1.0` (Complete mismatch)

### How is it calculated?
In physical quantum hardware, this requires building complex extra circuitry (like the SWAP Test) to physically interfere the query state with the prototype state and measure the result. 

Because we are simulating the quantum computer on classical hardware (up to 8 qubits), we can directly compute the math formula: $F = ⟨ψ|ρ|ψ⟩$. This makes our training simulations drastically faster while remaining mathematically identical to a perfect quantum computer.
