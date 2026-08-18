"""
qpn_model.py
Quantum Prototypical Network — Core PyTorch Module.

Architecture aligned with BTP_Quantum_few_rel/qpn/quantum/protonet.py.

Key design decisions (matching few_rel):
    - Amplitude Encoding via Qiskit StatePreparation (dynamic per-sample).
    - EfficientSU2 ansatz with circular entanglement.
    - FidelityParamShift: custom autograd.Function that computes PAIRWISE
      state fidelity F(s_i, q_j; theta) and differentiates via PSR.
    - Prototypical logits = beta x mean_support_fidelity_per_class (not infidelity).
    - cost_type='global' (full state fidelity) or 'local' (first qubit trace).

Gradient computation (Parameter-Shift Rule)
-------------------------------------------
    dF/dtheta_k = 0.5 x [F(theta_k + pi/2) - F(theta_k - pi/2)]

References
----------
- Snell et al. (2017) "Prototypical Networks for Few-shot Learning"
- Mitarai et al. (2018) "Quantum Circuit Learning"
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity, partial_trace
from qiskit.circuit.library import StatePreparation, EfficientSU2

from data.episode_sampler import Episode


# ==============================================================================
# Internal helper
# ==============================================================================

def _normalize_for_sp(tensor_vec: torch.Tensor) -> np.ndarray:
    """L2-normalize a 1-D float tensor into a unit numpy vector for StatePreparation."""
    v = tensor_vec.numpy().astype(np.float64)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


# ==============================================================================
# Custom autograd Function — Fidelity via Parameter-Shift Rule
# ==============================================================================

class FidelityParamShift(torch.autograd.Function):
    """
    Custom autograd.Function for differentiating through Qiskit quantum circuits
    using the Parameter-Shift Rule.

    Forward:  Computes the full S x Q pairwise fidelity matrix F[i,j].
    Backward: For each VQC angle theta_k, runs two shifted circuits (+/-pi/2)
              dF[i,j]/dtheta_k = 0.5 * [F(theta_k+pi/2) - F(theta_k-pi/2)].

    Mirrors few_rel's FidelityParamShift in qpn/quantum/protonet.py.
    """

    @staticmethod
    def forward(ctx, theta, s_x, q_x, ansatz, cost_type, n_qubits):
        ctx.save_for_backward(theta, s_x, q_x)
        ctx.ansatz = ansatz
        ctx.cost_type = cost_type
        ctx.n_qubits = n_qubits

        theta_np = theta.detach().numpy()
        s_states = FidelityParamShift._encode_batch(s_x, ansatz, theta_np, n_qubits)
        q_states = FidelityParamShift._encode_batch(q_x, ansatz, theta_np, n_qubits)

        S, Q = len(s_states), len(q_states)
        fidelities = torch.zeros(S, Q)
        for i in range(S):
            for j in range(Q):
                fidelities[i, j] = FidelityParamShift._fidelity(
                    s_states[i], q_states[j], cost_type, n_qubits
                )
        ctx.s_states = s_states
        ctx.q_states = q_states
        return fidelities

    @staticmethod
    def backward(ctx, grad_output):
        theta, s_x, q_x = ctx.saved_tensors
        ansatz = ctx.ansatz
        cost_type = ctx.cost_type
        n_qubits = ctx.n_qubits

        theta_np = theta.detach().numpy()
        S, Q = s_x.shape[0], q_x.shape[0]
        num_params = theta_np.shape[0]
        shift = np.pi / 2.0
        grad_theta = torch.zeros_like(theta)

        for k in range(num_params):
            t_plus = theta_np.copy(); t_plus[k] += shift
            t_minus = theta_np.copy(); t_minus[k] -= shift

            s_plus  = FidelityParamShift._encode_batch(s_x, ansatz, t_plus,  n_qubits)
            q_plus  = FidelityParamShift._encode_batch(q_x, ansatz, t_plus,  n_qubits)
            s_minus = FidelityParamShift._encode_batch(s_x, ansatz, t_minus, n_qubits)
            q_minus = FidelityParamShift._encode_batch(q_x, ansatz, t_minus, n_qubits)

            grad_k = 0.0
            for i in range(S):
                for j in range(Q):
                    if grad_output[i, j] != 0:
                        f_p = FidelityParamShift._fidelity(s_plus[i],  q_plus[j],  cost_type, n_qubits)
                        f_m = FidelityParamShift._fidelity(s_minus[i], q_minus[j], cost_type, n_qubits)
                        grad_k += grad_output[i, j].item() * 0.5 * (f_p - f_m)
            grad_theta[k] = grad_k

        return grad_theta, None, None, None, None, None

    @staticmethod
    def _encode_batch(X: torch.Tensor, ansatz: QuantumCircuit,
                      theta_np: np.ndarray, n_qubits: int) -> list:
        """Amplitude-encode each row of X, compose with VQC(theta), return Statevectors."""
        bound_ansatz = ansatz.assign_parameters(theta_np)
        states = []
        for i in range(X.shape[0]):
            sp = StatePreparation(_normalize_for_sp(X[i]))
            qc = QuantumCircuit(n_qubits)
            qc.append(sp, range(n_qubits))
            qc = qc.compose(bound_ansatz)
            states.append(Statevector(qc))
        return states

    @staticmethod
    def _fidelity(s, q, cost_type: str, n_qubits: int) -> float:
        """Compute fidelity between two states (global or local cost)."""
        if cost_type == "global":
            return float(state_fidelity(s, q))
        else:
            trace_qubits = list(range(1, n_qubits))
            rho_s = partial_trace(s, trace_qubits)
            rho_q = partial_trace(q, trace_qubits)
            return float(state_fidelity(rho_s, rho_q))


# ==============================================================================
# QuantumProtoNet — nn.Module
# ==============================================================================

class QuantumProtoNet(nn.Module):
    """
    Quantum Prototypical Network for few-shot text classification.

    Architecture aligned with BTP_Quantum_few_rel's QuantumProtoNet:
      - Amplitude Encoding (StatePreparation) instead of Angle Encoding.
      - EfficientSU2 ansatz with circular entanglement (matching few_rel).
      - FidelityParamShift autograd for real gradients via PSR.
      - Prototypical logits = beta x mean support fidelity per class.
      - Global and local cost function support.

    Args:
        n_qubits:    Number of qubits. State dimension = 2^n_qubits.
        ansatz_reps: Number of EfficientSU2 repetition layers.
        init_type:   'identity_block' (zeros), 'random' (small), 'uniform' (full range).
        cost_type:   'global' or 'local'.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        ansatz_reps: int = 2,
        init_type: str = "identity_block",
        cost_type: str = "global",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.cost_type = cost_type

        # EfficientSU2 with circular entanglement (matching few_rel)
        self._ansatz = EfficientSU2(
            num_qubits=n_qubits,
            su2_gates=["ry", "rz"],
            entanglement="circular",
            reps=ansatz_reps,
        )
        num_params = self._ansatz.num_parameters

        if init_type == "identity_block":
            init_vals = np.zeros(num_params)
        elif init_type == "random":
            init_vals = np.random.uniform(-0.1, 0.1, num_params)
        else:
            init_vals = np.random.uniform(-np.pi, np.pi, num_params)

        self.theta = nn.Parameter(torch.tensor(init_vals, dtype=torch.float32))
        # Learnable temperature (few_rel initialises to 10.0)
        self.beta = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, s_x: torch.Tensor, s_y: torch.Tensor,
                q_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_x: (S, state_dim) float  -- amplitude-encoded support.
            s_y: (S,)           long   -- support labels 0..N-1.
            q_x: (Q, state_dim) float  -- amplitude-encoded query.

        Returns:
            logits: (Q, N) float tensor -- beta x per-class mean fidelity.
        """
        if self.training:
            return self._trainable_forward(s_x, q_x, s_y)
        else:
            return self._eval_forward(s_x, q_x, s_y)

    def _trainable_forward(self, s_x, q_x, s_y) -> torch.Tensor:
        """PSR-differentiable training path."""
        fidelities = FidelityParamShift.apply(
            self.theta, s_x, q_x, self._ansatz, self.cost_type, self.n_qubits
        )  # (S, Q)

        classes = torch.unique(s_y)
        proto_fidelities = []
        for c in classes:
            mask = (s_y == c)
            proto_fidelities.append(fidelities[mask, :].mean(dim=0))  # (Q,)

        proto_fidelities = torch.stack(proto_fidelities, dim=1)  # (Q, N)
        return self.beta * proto_fidelities

    def _eval_forward(self, s_x, q_x, s_y) -> torch.Tensor:
        """DensityMatrix-prototype inference path (no grad required)."""
        theta_np = self.theta.detach().numpy()
        s_states = FidelityParamShift._encode_batch(
            s_x, self._ansatz, theta_np, self.n_qubits
        )
        q_states = FidelityParamShift._encode_batch(
            q_x, self._ansatz, theta_np, self.n_qubits
        )

        classes = torch.unique(s_y)
        Q = q_x.shape[0]
        proto_fidelities = torch.zeros(Q, len(classes))

        for c_idx, c in enumerate(classes):
            mask = (s_y == c)
            class_states = [s_states[i] for i in range(len(s_states)) if mask[i]]

            rho_data = DensityMatrix(class_states[0]).data / len(class_states)
            for sv in class_states[1:]:
                rho_data += DensityMatrix(sv).data / len(class_states)
            rho_c = DensityMatrix(rho_data)

            for j, q_sv in enumerate(q_states):
                if self.cost_type == "global":
                    f = float(state_fidelity(q_sv, rho_c))
                else:
                    trace_qubits = list(range(1, self.n_qubits))
                    rho_q_local = partial_trace(q_sv, trace_qubits)
                    rho_c_local = partial_trace(rho_c, trace_qubits)
                    f = float(state_fidelity(rho_q_local, rho_c_local))
                proto_fidelities[j, c_idx] = f

        return self.beta * proto_fidelities

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, episode: Episode) -> np.ndarray:
        """
        Run inference on an amplitude-preprocessed Episode.

        Returns:
            np.ndarray of predicted class indices (episode-local 0..N-1).
        """
        self.eval()
        with torch.no_grad():
            s_x = torch.tensor(episode.support_x, dtype=torch.float32)
            q_x = torch.tensor(episode.query_x,   dtype=torch.float32)
            s_y = torch.tensor(episode.support_y,  dtype=torch.long)
            logits = self.forward(s_x, s_y, q_x)
            return logits.argmax(dim=1).numpy()
