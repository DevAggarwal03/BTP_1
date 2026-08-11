# This module measures how well the quantum kernel components can be trained on few-shot tasks.
# It provides a lightweight training loop for testing and comparing kernel-based learners.
import torch
import numpy as np
from tqdm import tqdm
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_aer.primitives import Sampler as AerSampler

from qpn.quantum.protonet import QuantumProtoNet
from qpn.quantum.encoding import get_feature_map

def estimate_gradient_variance(n_qubits: int, depth: int, cost_type: str, n_samples: int = 50, seed: int = 42) -> float:
    """
    Estimates Var[∂cost/∂θ_k] for a random parameter θ_k.
    We take a fixed random input, and sample many random initializations of θ.
    
    n_qubits: Number of qubits.
    depth: Number of layers in the ansatz.
    cost_type: "global" or "local".
    n_samples: Number of random parameter initializations to sample over.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Use a dummy problem: 2 support samples (1 per class), 1 query sample
    # Random normalized features scaled to [-pi, pi]
    s_x = torch.FloatTensor(np.random.uniform(-np.pi, np.pi, size=(2, n_qubits)))
    q_x = torch.FloatTensor(np.random.uniform(-np.pi, np.pi, size=(1, n_qubits)))
    s_y = torch.tensor([0, 1])
    
    model = QuantumProtoNet(n_qubits=n_qubits, fm_kind="angle", ansatz_reps=depth, cost_type=cost_type, init_type="random")
    model.train()
    
    gradients = []
    
    # We will track the gradient of the first parameter to observe the barren plateau
    # (Any parameter works, but first layer is typical)
    
    for _ in range(n_samples):
        # Re-initialize to random angles
        # The ansatz typically has parameters in [-pi, pi]
        new_theta = torch.FloatTensor(np.random.uniform(-np.pi, np.pi, size=model.theta.shape))
        model.theta.data = new_theta
        model.theta.requires_grad_(True)
        
        # Zero gradients
        if model.theta.grad is not None:
            model.theta.grad.zero_()
            
        # Forward and backward
        logits = model(s_x, q_x, s_y)
        # Using a dummy cross-entropy-like target (e.g. query belongs to class 0)
        target = torch.tensor([0])
        # Cross entropy loss
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        
        grad_val = model.theta.grad[0].item()
        gradients.append(grad_val)
        
    variance = np.var(gradients)
    return variance

def estimate_kernel_concentration(n_qubits: int, n_samples: int = 50, seed: int = 42) -> float:
    """
    Estimates the variance of the off-diagonal kernel elements to measure kernel concentration.
    An exponentially decaying variance implies the kernel matrix becomes the identity matrix, 
    making learning impossible.
    """
    np.random.seed(seed)
    
    # Generate random data samples
    # We measure pairwise fidelity between random data points
    X = np.random.uniform(-np.pi, np.pi, size=(n_samples, n_qubits))
    
    feature_map = get_feature_map("zz", n_qubits, reps=1)
    sampler = AerSampler(run_options={"shots": None})
    fidelity = ComputeUncompute(sampler=sampler)
    qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    
    K = qkernel.evaluate(x_vec=X)
    
    # Extract off-diagonal elements
    off_diagonal_elements = K[np.triu_indices(n_samples, k=1)]
    
    variance = np.var(off_diagonal_elements)
    return variance
