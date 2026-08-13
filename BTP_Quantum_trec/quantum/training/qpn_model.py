"""
qpn_model.py
Quantum Prototypical Network — Core PyTorch Module.

Implements the full episodic forward pass:

    support_x, support_y  →  Angle Encode  →  VQC(θ)  →  Statevectors
                          →  Density Matrix Prototypes ρ_k per class
    query_x               →  Angle Encode  →  VQC(θ)  →  Statevectors
                          →  Infidelity distances d(|ψ_q⟩, ρ_k) for each k
                          →  Softmax(−β · d)  →  logits

The VQC angles θ are stored as a nn.Parameter so PyTorch/Adam can track
them. Gradients are computed via the Parameter-Shift Rule: for each call
to forward(), the underlying Qiskit circuits are run with parameter values
evaluated from the current θ tensor (no TorchConnector overhead).

References
----------
- Snell et al. (2017) "Prototypical Networks for Few-shot Learning"
- Quantum analogue: mixed-state prototype + fidelity distance
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity

from quantum.encoding.angle_encoding import AngleEncoder
from quantum.vqc.vqc_extractor import VQCFeatureExtractor
from quantum.prototype_calculation.prototype_ops import QuantumPrototypeCalculator
from data.episode_sampler import Episode


class QuantumProtoNet(nn.Module):
    """
    Quantum Prototypical Network (QPN) for few-shot text classification.

    The forward pass implements the full prototypical network computation
    using real quantum operations (Statevectors + Density Matrices + Fidelity).

    Args:
        n_qubits:     Number of qubits (= number of QHBA-selected features).
        ansatz_type:  VQC ansatz family ('EfficientSU2', 'RealAmplitudes', ...).
        reps:         Number of VQC ansatz repetition layers.
        entanglement: Entanglement topology ('linear', 'full', 'circular').
        temperature:  Initial softmax temperature β (learnable scalar).
    """

    def __init__(
        self,
        n_qubits: int = 8,
        ansatz_type: str = "EfficientSU2",
        reps: int = 2,
        entanglement: str = "linear",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits

        # --- Encoding ---
        self.encoder = AngleEncoder(n_qubits=n_qubits)

        # --- VQC Ansatz ---
        self.vqc = VQCFeatureExtractor(
            num_qubits=n_qubits,
            ansatz_type=ansatz_type,
            reps=reps,
            entanglement=entanglement,
        )
        self._ansatz_circuit = self.vqc.get_circuit()
        n_params = self._ansatz_circuit.num_parameters

        # --- Trainable VQC angles θ (the only thing Adam optimizes) ---
        self.theta = nn.Parameter(
            torch.nn.init.uniform_(
                torch.empty(n_params), a=0.0, b=2.0 * float(np.pi)
            )
        )

        # --- Learnable softmax temperature β ---
        self.log_beta = nn.Parameter(torch.tensor(float(np.log(temperature))))

        # --- Helpers ---
        self._proto_calc = QuantumPrototypeCalculator(num_qubits=n_qubits)

    # ------------------------------------------------------------------
    # Forward pass (episodic)
    # ------------------------------------------------------------------

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Episodic forward pass.

        Args:
            support_x: float tensor (N*K, n_qubits)  — support features.
            support_y: long  tensor (N*K,)            — support labels 0..N-1.
            query_x:   float tensor (N*Q, n_qubits)   — query features.

        Returns:
            logits: float tensor (N*Q, N) — log-softmax over infidelity distances.
        """
        theta_np = self.theta.detach().cpu().numpy()

        # 1. Encode + run VQC for all support samples
        support_svs = self._encode_batch(support_x.detach().cpu().numpy(), theta_np)

        # 2. Compute density matrix prototype per class
        classes = torch.unique(support_y).tolist()
        prototypes: list[DensityMatrix] = []
        for cls in sorted(classes):
            mask = (support_y == int(cls)).cpu().numpy().astype(bool)
            cls_states = [support_svs[i] for i in range(len(support_svs)) if mask[i]]
            proto = self._proto_calc.calculate_class_prototype(cls_states)
            prototypes.append(proto)

        # 3. Encode + run VQC for all query samples
        query_svs = self._encode_batch(query_x.detach().cpu().numpy(), theta_np)

        # 4. Compute infidelity distances (N*Q × N)
        n_queries = len(query_svs)
        n_classes = len(prototypes)
        distances = np.zeros((n_queries, n_classes), dtype=np.float32)
        for q_idx, q_sv in enumerate(query_svs):
            for c_idx, proto in enumerate(prototypes):
                fidelity = float(state_fidelity(q_sv, proto))
                distances[q_idx, c_idx] = 1.0 - fidelity

        # 5. Softmax over negative distances with learnable temperature β
        dist_tensor = torch.tensor(distances, dtype=torch.float32)
        beta = torch.exp(self.log_beta)
        logits = -beta * dist_tensor   # (N*Q, N)

        return logits

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, episode: Episode) -> np.ndarray:
        """
        Run inference on a full Episode (no gradient tracking).

        Args:
            episode: An Episode dataclass with support_x/y, query_x/y.

        Returns:
            np.ndarray of predicted class indices (episode-local 0..N-1).
        """
        self.eval()
        with torch.no_grad():
            s_x = torch.tensor(episode.support_x, dtype=torch.float32)
            q_x = torch.tensor(episode.query_x, dtype=torch.float32)
            s_y = torch.tensor(episode.support_y, dtype=torch.long)
            logits = self.forward(s_x, s_y, q_x)
            return logits.argmax(dim=1).numpy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_batch(
        self, X: np.ndarray, theta: np.ndarray
    ) -> list[Statevector]:
        """
        Angle-encode each row of X, compose with VQC(theta), return Statevectors.

        Args:
            X:     (N, n_qubits) float array, values in [0, 1].
            theta: (n_params,) array of current VQC angles.

        Returns:
            List of N Qiskit Statevector objects.
        """
        # Bind the ansatz once with current theta
        bound_ansatz = self._ansatz_circuit.assign_parameters(theta)

        statevectors = []
        for x in X:
            enc_circuit = self.encoder.encode(x)
            full_circuit = enc_circuit.compose(bound_ansatz)
            sv = Statevector.from_instruction(full_circuit)
            statevectors.append(sv)
        return statevectors

