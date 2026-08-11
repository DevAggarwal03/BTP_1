### why are we doing this?
- This is for our Btech final year project
- Our goal in to write and publish 2 research papers in a good coferance/journal in the upcoming semester.
- We are working in the research area of quantum machine learning.
- Under our professor from our college
- We are a team of 2 students, Implementing the pipeline in our different environment 
- I am using the TREC50 dataset, and my teammate is using the few-rel dataset
- Below is the paper idea (theorical part)

### Contributions: 

*We propose Quantum Prototypical Networks (QProtoNet), a framework for few-shot text classification that represents class prototypes as quantum mixed states and uses quantum state fidelity as the classification metric, providing a theoretically grounded quantum advantage in low-data regimes.*

A quantum implementation of prototypical networks specifically for **text/NLP tasks**, with **mixed-state prototypes** and **fidelity-based inference**, framed as **episodic meta-learning**. That combination has no existing paper.

Existing quantum few-shot learning work is either non-NLP (image-domain) or non-prototypical (DisCoCat/VQC fine-tuning). Nobody has proposed mixed-state prototypes + quantum fidelity as an episodic meta-learning framework for text classification.

- Table
    
    Here is the precise mapping from classical ProtoNet → QProtoNet:
    
    | Classical Component | Quantum Replacement |
    | --- | --- |
    | Encoder `fθ(x)` → vector `z ∈ ℝⁿ` | PQC `U(θ) |
    | Class prototype `cₖ = mean(zᵢ)` | Mixed state `ρₖ = (1/|Sₖ|) Σ |ψᵢ⟩⟨ψᵢ|` |
    | Distance `||z - cₖ||²` | Infidelity `1 - F(|ψ(x)⟩, ρₖ)` |
    | Softmax over distances | Softmax over infidelities |
    | Backprop through `fθ` | Parameter-shift rule through PQC |
    | Euclidean metric (Bregman div.) | Quantum fidelity (your Bregman analogue claim) |
    
    The softmax classification rule stays structurally identical — you just swap the distance. The training loop stays structurally identical — you just train via parameter-shift. This structural cleanliness is what makes the paper theoretically elegant.