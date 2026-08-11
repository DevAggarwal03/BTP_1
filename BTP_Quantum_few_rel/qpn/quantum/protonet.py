# This module defines the quantum prototype network model and its training loop.
# It combines quantum feature encoding with a prototypical network objective for few-shot classification.
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity, partial_trace
from qiskit.circuit.library import StatePreparation
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import Sampler

from qpn.episodes import Episode
from qpn.quantum.encoding import get_feature_map, QuantumFeaturePreprocessor
from qpn.quantum.ansatz import create_ansatz

def _normalize_for_sp(tensor_vec):
    v = tensor_vec.numpy().astype(np.float64)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v

class FidelityParamShift(torch.autograd.Function):
    """
    Manual parameter-shift implementation for fidelity gradients.
    Computes F(s, q; theta) = |<psi(s; theta)|psi(q; theta)>|^2
    """
    @staticmethod
    def forward(ctx, theta, s_x, q_x, precomputed_circuits, fidelity_primitive, cost_type, n_qubits):
        """
        theta: (num_params,)
        s_x: (S, D) - Support set features
        q_x: (Q, D) - Query set features
        """
        ctx.save_for_backward(theta, s_x, q_x)
        ctx.precomputed_circuits = precomputed_circuits
        ctx.fidelity_primitive = fidelity_primitive
        ctx.cost_type = cost_type
        ctx.n_qubits = n_qubits
        
        # Construct parameter bindings
        # We need pairwise fidelities between all support and query
        S = s_x.shape[0]
        Q = q_x.shape[0]
        
        # For simplicity, we just use the primitive sequentially or batched.
        # However, for training we typically compute the mean over support for each class.
        # We will compute the full S x Q matrix.
        fidelities = torch.zeros(S, Q)
        
        # We will use the Statevector simulator directly for the forward pass 
        # because ComputeUncompute requires binding both feature map and ansatz.
        # Actually, Statevector is much faster for state simulation.
        
        theta_np = theta.detach().numpy()
        
        # Evaluate all support states
        s_states = []
        for i in range(S):
            if precomputed_circuits[0] is not None:
                bound_circ = precomputed_circuits[0].assign_parameters(np.concatenate([s_x[i].numpy(), theta_np]))
            else:
                # Amplitude encoding: dynamic circuit construction
                sp = StatePreparation(_normalize_for_sp(s_x[i]))
                qc = QuantumCircuit(n_qubits)
                qc.append(sp, range(n_qubits))
                qc = qc.compose(precomputed_circuits[1]) # precomputed_circuits[1] holds the ansatz
                bound_circ = qc.assign_parameters(theta_np)
            s_states.append(Statevector(bound_circ))
            
        # Evaluate all query states
        q_states = []
        for j in range(Q):
            if precomputed_circuits[0] is not None:
                bound_circ = precomputed_circuits[0].assign_parameters(np.concatenate([q_x[j].numpy(), theta_np]))
            else:
                sp = StatePreparation(_normalize_for_sp(q_x[j]))
                qc = QuantumCircuit(n_qubits)
                qc.append(sp, range(n_qubits))
                qc = qc.compose(precomputed_circuits[1])
                bound_circ = qc.assign_parameters(theta_np)
            q_states.append(Statevector(bound_circ))
            
        for i in range(S):
            for j in range(Q):
                if cost_type == "global":
                    fidelities[i, j] = state_fidelity(s_states[i], q_states[j])
                else: # local
                    trace_qubits = list(range(1, n_qubits))
                    rho_s_local = partial_trace(s_states[i], trace_qubits)
                    rho_q_local = partial_trace(q_states[j], trace_qubits)
                    fidelities[i, j] = state_fidelity(rho_s_local, rho_q_local)
                
        ctx.s_states = s_states
        ctx.q_states = q_states
        return fidelities

    @staticmethod
    def backward(ctx, grad_output):
        theta, s_x, q_x = ctx.saved_tensors
        precomputed_circuits = ctx.precomputed_circuits
        cost_type = ctx.cost_type
        n_qubits = ctx.n_qubits
        
        S = s_x.shape[0]
        Q = q_x.shape[0]
        num_params = theta.shape[0]
        
        grad_theta = torch.zeros_like(theta)
        
        # Parameter shift rule for fidelity
        # F_k = 0.5 * (F(theta + pi/2) - F(theta - pi/2))
        shift = np.pi / 2.0
        
        theta_np = theta.detach().numpy()
        
        for k in range(num_params):
            theta_plus = theta_np.copy()
            theta_plus[k] += shift
            
            theta_minus = theta_np.copy()
            theta_minus[k] -= shift
            
            # Recompute F(theta_plus) and F(theta_minus)
            # To optimize, we can compute only the ones needed.
            # But we must sum over all (i, j) weighted by grad_output[i, j]
            s_states_plus = []
            q_states_plus = []
            s_states_minus = []
            q_states_minus = []
            
            for i in range(S):
                if precomputed_circuits[0] is not None:
                    s_states_plus.append(Statevector(precomputed_circuits[0].assign_parameters(np.concatenate([s_x[i].numpy(), theta_plus]))))
                    s_states_minus.append(Statevector(precomputed_circuits[0].assign_parameters(np.concatenate([s_x[i].numpy(), theta_minus]))))
                else:
                    sp = StatePreparation(_normalize_for_sp(s_x[i]))
                    qc = QuantumCircuit(n_qubits)
                    qc.append(sp, range(n_qubits))
                    qc = qc.compose(precomputed_circuits[1])
                    s_states_plus.append(Statevector(qc.assign_parameters(theta_plus)))
                    s_states_minus.append(Statevector(qc.assign_parameters(theta_minus)))
                    
            for j in range(Q):
                if precomputed_circuits[0] is not None:
                    q_states_plus.append(Statevector(precomputed_circuits[0].assign_parameters(np.concatenate([q_x[j].numpy(), theta_plus]))))
                    q_states_minus.append(Statevector(precomputed_circuits[0].assign_parameters(np.concatenate([q_x[j].numpy(), theta_minus]))))
                else:
                    sp = StatePreparation(_normalize_for_sp(q_x[j]))
                    qc = QuantumCircuit(n_qubits)
                    qc.append(sp, range(n_qubits))
                    qc = qc.compose(precomputed_circuits[1])
                    q_states_plus.append(Statevector(qc.assign_parameters(theta_plus)))
                    q_states_minus.append(Statevector(qc.assign_parameters(theta_minus)))
            
            grad_k = 0.0
            for i in range(S):
                for j in range(Q):
                    if grad_output[i, j] != 0:
                        if cost_type == "global":
                            f_plus = state_fidelity(s_states_plus[i], q_states_plus[j])
                            f_minus = state_fidelity(s_states_minus[i], q_states_minus[j])
                        else:
                            trace_qubits = list(range(1, n_qubits))
                            rho_s_plus = partial_trace(s_states_plus[i], trace_qubits)
                            rho_q_plus = partial_trace(q_states_plus[j], trace_qubits)
                            f_plus = state_fidelity(rho_s_plus, rho_q_plus)
                            
                            rho_s_minus = partial_trace(s_states_minus[i], trace_qubits)
                            rho_q_minus = partial_trace(q_states_minus[j], trace_qubits)
                            f_minus = state_fidelity(rho_s_minus, rho_q_minus)
                            
                        grad_k += grad_output[i, j].item() * 0.5 * (f_plus - f_minus)
                        
            grad_theta[k] = grad_k
            
        return grad_theta, None, None, None, None, None, None

class QuantumProtoNet(nn.Module):
    def __init__(self, n_qubits: int, fm_kind: str = "zz", ansatz_reps: int = 2, init_type: str = "identity_block", cost_type: str = "global"):
        super().__init__()
        self.n_qubits = min(n_qubits, 12)
        self.fm_kind = fm_kind
        self.cost_type = cost_type
        
        # Build circuits
        self.feature_map = get_feature_map(fm_kind, self.n_qubits)
        self.ansatz, init_params = create_ansatz(self.n_qubits, ansatz_reps, init_type)
        
        if self.feature_map is not None:
            self.full_circuit = self.feature_map.compose(self.ansatz)
        else:
            self.full_circuit = None
        
        # Learnable parameters
        self.theta = nn.Parameter(torch.tensor(init_params, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(10.0, dtype=torch.float32)) # Learnable inverse-temperature
        
        self.sampler = Sampler()
        self.fidelity_primitive = ComputeUncompute(sampler=self.sampler)
        
    def _compute_local_fidelity(self, state1: Statevector, state2: Statevector) -> float:
        """
        Computes local fidelity (overlap on the first qubit).
        Traces out all qubits except qubit 0.
        """
        if self.n_qubits == 1:
            return state_fidelity(state1, state2)
            
        trace_qubits = list(range(1, self.n_qubits))
        rho1 = partial_trace(state1, trace_qubits)
        rho2 = partial_trace(state2, trace_qubits)
        
        return state_fidelity(rho1, rho2)

    def _trainable_forward(self, s_x, q_x, s_y):
        """
        Trainable path using parameter-shift.
        s_x: (S, D), q_x: (Q, D)
        """
        fidelities = FidelityParamShift.apply(self.theta, s_x, q_x, [self.full_circuit, self.ansatz], self.fidelity_primitive, self.cost_type, self.n_qubits)
        
        # fidelities is (S, Q)
        # We want prototype fidelity: mean over support for each class
        classes = torch.unique(s_y)
        num_classes = len(classes)
        Q = q_x.shape[0]
        
        c_fidelities = []
        for c_idx, c in enumerate(classes):
            mask = (s_y == c)
            # Mean over support instances
            c_fidelities.append(fidelities[mask, :].mean(dim=0))
            
        proto_fidelities = torch.stack(c_fidelities, dim=1)
        logits = self.beta * proto_fidelities
        return logits

    def _eval_forward(self, s_x, q_x, s_y):
        """
        Eval-only path: Construct DensityMatrix prototypes for exact, fast inference.
        """
        theta_np = self.theta.detach().numpy()
        S = s_x.shape[0]
        Q = q_x.shape[0]
        
        s_states = []
        for i in range(S):
            if self.full_circuit is not None:
                bound = self.full_circuit.assign_parameters(np.concatenate([s_x[i].numpy(), theta_np]))
            else:
                sp = StatePreparation(_normalize_for_sp(s_x[i]))
                qc = QuantumCircuit(self.n_qubits)
                qc.append(sp, range(self.n_qubits))
                qc = qc.compose(self.ansatz)
                bound = qc.assign_parameters(theta_np)
            s_states.append(Statevector(bound))
            
        q_states = []
        for j in range(Q):
            if self.full_circuit is not None:
                bound = self.full_circuit.assign_parameters(np.concatenate([q_x[j].numpy(), theta_np]))
            else:
                sp = StatePreparation(_normalize_for_sp(q_x[j]))
                qc = QuantumCircuit(self.n_qubits)
                qc.append(sp, range(self.n_qubits))
                qc = qc.compose(self.ansatz)
                bound = qc.assign_parameters(theta_np)
            q_states.append(Statevector(bound))
            
        classes = torch.unique(s_y)
        num_classes = len(classes)
        proto_fidelities = torch.zeros(Q, num_classes)
        
        for c_idx, c in enumerate(classes):
            mask = (s_y == c)
            class_states = [s_states[i] for i in range(S) if mask[i]]
            
            # Density matrix centroid computed via raw arrays
            rho_data = DensityMatrix(class_states[0]).data * (1.0 / len(class_states))
            for i in range(1, len(class_states)):
                rho_data += DensityMatrix(class_states[i]).data * (1.0 / len(class_states))
            rho_c = DensityMatrix(rho_data)
                
            for j in range(Q):
                if self.cost_type == "global":
                    f = state_fidelity(q_states[j], rho_c)
                else: # Local cost
                    trace_qubits = list(range(1, self.n_qubits))
                    rho_q_local = partial_trace(q_states[j], trace_qubits)
                    rho_c_local = partial_trace(rho_c, trace_qubits)
                    f = state_fidelity(rho_q_local, rho_c_local)
                
                proto_fidelities[j, c_idx] = f
                
        logits = self.beta * proto_fidelities
        return logits

    def forward(self, s_x, q_x, s_y):
        if self.training:
            return self._trainable_forward(s_x, q_x, s_y)
        else:
            return self._eval_forward(s_x, q_x, s_y)
            
    def predict(self, episode: Episode) -> np.ndarray:
        self.eval()
        
        # We assume the episode has already been preprocessed to the correct n_qubits
        # and scaled to [-pi, pi] by the global benchmark harness.
        s_x_t = torch.tensor(episode.support_x, dtype=torch.float32)
        q_x_t = torch.tensor(episode.query_x, dtype=torch.float32)
        s_y_t = torch.tensor(episode.support_y, dtype=torch.long)
        
        with torch.no_grad():
            logits = self.forward(s_x_t, q_x_t, s_y_t)
            preds = torch.argmax(logits, dim=1).numpy()
            
        classes = np.unique(episode.support_y)
        return classes[preds]
