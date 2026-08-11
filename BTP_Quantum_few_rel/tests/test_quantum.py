# This test module covers the quantum components of the repository.
# It validates circuit construction and fidelity-related behavior for the proposed quantum pipeline.
import pytest
import torch
import torch.nn as nn
import numpy as np
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity

from qpn.quantum.encoding import get_feature_map, QuantumFeaturePreprocessor
from qpn.quantum.ansatz import create_ansatz
from qpn.quantum.protonet import QuantumProtoNet, FidelityParamShift

def test_encoding_valid_states():
    qc = get_feature_map("angle", 2)
    # 2 features for 2 qubits
    features = np.array([0.5, -0.5])
    qc_bound = qc.assign_parameters(features)
    sv = Statevector(qc_bound)
    
    assert np.isclose(np.linalg.norm(sv.data), 1.0)
    
def test_fidelity_bounds():
    qc1 = get_feature_map("angle", 2).assign_parameters([0.5, 0.5])
    qc2 = get_feature_map("angle", 2).assign_parameters([-0.5, -0.5])
    
    sv1 = Statevector(qc1)
    sv2 = Statevector(qc2)
    
    fid = state_fidelity(sv1, sv2)
    assert 0.0 <= fid <= 1.0
    
    self_fid = state_fidelity(sv1, sv1)
    assert np.isclose(self_fid, 1.0)

def test_density_matrix_trace():
    qc1 = get_feature_map("angle", 2).assign_parameters([0.5, 0.5])
    qc2 = get_feature_map("angle", 2).assign_parameters([-0.5, -0.5])
    
    dm = 0.5 * DensityMatrix(qc1) + 0.5 * DensityMatrix(qc2)
    
    assert np.isclose(np.trace(dm.data), 1.0)
    
def test_gradient_finite_difference():
    model = QuantumProtoNet(n_qubits=2, fm_kind="angle", ansatz_reps=1, cost_type="global")
    model.train()
    
    # 2 support samples, 1 query sample
    s_x = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    q_x = torch.tensor([[0.5, -0.1]])
    s_y = torch.tensor([0, 0])
    
    # Enable gradients
    model.theta.requires_grad_(True)
    
    # Forward pass
    logits = model(s_x, q_x, s_y)
    loss = logits.sum()
    loss.backward()
    
    param_shift_grad = model.theta.grad.clone()
    assert param_shift_grad is not None
    assert torch.any(param_shift_grad != 0.0) # Should be non-zero
    
    # Finite difference
    epsilon = 1e-4
    fd_grad = torch.zeros_like(param_shift_grad)
    
    # Need to preserve the original theta
    orig_theta = model.theta.detach().clone()
    
    for k in range(len(orig_theta)):
        theta_plus = orig_theta.clone()
        theta_plus[k] += epsilon
        model.theta.data = theta_plus
        logits_plus = model(s_x, q_x, s_y)
        loss_plus = logits_plus.sum().item()
        
        theta_minus = orig_theta.clone()
        theta_minus[k] -= epsilon
        model.theta.data = theta_minus
        logits_minus = model(s_x, q_x, s_y)
        loss_minus = logits_minus.sum().item()
        
        fd_grad[k] = (loss_plus - loss_minus) / (2 * epsilon)
        
    model.theta.data = orig_theta
    
    # Check if param-shift gradient matches finite difference
    assert torch.allclose(param_shift_grad, fd_grad, atol=1e-3, rtol=1e-2)

def test_local_gradient_finite_difference():
    model = QuantumProtoNet(n_qubits=2, fm_kind="angle", ansatz_reps=1, cost_type="local")
    model.train()
    
    s_x = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    q_x = torch.tensor([[0.5, -0.1]])
    s_y = torch.tensor([0, 0])
    
    model.theta.requires_grad_(True)
    logits = model(s_x, q_x, s_y)
    loss = logits.sum()
    loss.backward()
    
    param_shift_grad = model.theta.grad.clone()
    assert torch.any(param_shift_grad != 0.0)
    
    epsilon = 1e-4
    fd_grad = torch.zeros_like(param_shift_grad)
    orig_theta = model.theta.detach().clone()
    
    for k in range(len(orig_theta)):
        theta_plus = orig_theta.clone()
        theta_plus[k] += epsilon
        model.theta.data = theta_plus
        logits_plus = model(s_x, q_x, s_y)
        loss_plus = logits_plus.sum().item()
        
        theta_minus = orig_theta.clone()
        theta_minus[k] -= epsilon
        model.theta.data = theta_minus
        logits_minus = model(s_x, q_x, s_y)
        loss_minus = logits_minus.sum().item()
        
        fd_grad[k] = (loss_plus - loss_minus) / (2 * epsilon)
        
    model.theta.data = orig_theta
    assert torch.allclose(param_shift_grad, fd_grad, atol=1e-3, rtol=1e-2)
