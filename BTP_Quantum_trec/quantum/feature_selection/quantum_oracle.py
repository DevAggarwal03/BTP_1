"""
quantum_oracle.py
Quantum fitness oracle for QHBA using Qiskit Aer simulation.

Role in the pipeline
--------------------
The quantum oracle evaluates a candidate feature-mask (represented as a
continuous position vector) by:

    1. Angle-encoding the position onto n_qubits.
    2. Applying a fixed shallow variational ansatz (RealAmplitudes, reps=1)
       as a "quantum mixer" — this scrambles the encoded state in a way that
       depends non-trivially on the input values.
    3. Measuring all qubits over `shots` runs.
    4. Returning a fitness proxy derived from the measurement distribution.

Fitness proxy
-------------
    quantum_fitness = 1 − P(|0...0⟩)

    Intuition: a feature subset that produces a highly concentrated
    measurement distribution (high P(|0⟩)) is preferred — it suggests
    the encoded angles are aligned with the ansatz's ground state,
    indicating a "meaningful" feature configuration.

This is a *heuristic* quantum enhancement. It is combined 50/50 with a
classical KNN fitness in ``qhba.py`` for the final evaluation.

GPU Support
-----------
The oracle auto-detects a CUDA-capable GPU via ``qiskit-aer-gpu`` and
falls back to CPU simulation if unavailable. ``evaluate_batch`` submits all
agent circuits as a *single* ``backend.run()`` call, removing the per-agent
transpile-wait loop and enabling GPU parallelism.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import real_amplitudes
from qiskit_aer import AerSimulator


def _make_aer_backend(seed: int) -> AerSimulator:
    """Return a GPU-accelerated AerSimulator, falling back to CPU."""
    try:
        backend = AerSimulator(device="GPU", seed_simulator=seed)
        # Probe the backend — raises if no GPU available
        backend.configuration()
        print("[QuantumOracle] GPU backend initialised (qiskit-aer-gpu).")
        return backend
    except Exception:
        print("[QuantumOracle] GPU unavailable — using CPU simulator.")
        return AerSimulator(seed_simulator=seed)


class QuantumOracle:
    """
    Qiskit-based quantum fitness oracle.

    Args:
        n_qubits: Number of qubits. Must match the feature vector length
                  passed to ``evaluate()``.
        shots:    Number of measurement shots per evaluation.
        seed:     Random seed for the fixed ansatz weights and simulator.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        shots: int = 1024,
        seed: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.shots = shots

        # GPU-aware backend with CPU fallback
        self.backend = _make_aer_backend(seed)

        # Fix random ansatz weights once (not trained — used as a deterministic mixer)
        rng = np.random.default_rng(seed=seed)
        ansatz_template = real_amplitudes(n_qubits, reps=1)
        self._ansatz_weights = rng.uniform(
            0.0, 2.0 * np.pi, size=ansatz_template.num_parameters
        )
        # Pre-bind the ansatz so we don't rebind on every call
        self._bound_ansatz: QuantumCircuit = ansatz_template.assign_parameters(
            self._ansatz_weights
        )
        self._all_zeros_key = "0" * n_qubits

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluate the quantum fitness of a continuous position vector.

        Args:
            position: 1-D array of length ≥ ``n_qubits`` in ``[0, 1]``.
                      Only the first ``n_qubits`` elements are used.

        Returns:
            Scalar quantum fitness in ``[0, 1]`` (lower = better).
        """
        x = np.clip(np.asarray(position, dtype=float)[: self.n_qubits], 0.0, 1.0)
        qc = self._build_circuit(x)

        transpiled = transpile(qc, self.backend, optimization_level=0)
        job = self.backend.run(transpiled, shots=self.shots)
        counts: dict[str, int] = job.result().get_counts()

        p_zero = counts.get(self._all_zeros_key, 0) / self.shots
        return float(1.0 - p_zero)

    def evaluate_batch(self, positions: np.ndarray) -> np.ndarray:
        """
        Evaluate quantum fitness for a batch of position vectors.

        Builds all circuits first, then submits them as a **single**
        ``backend.run()`` call. This removes the per-agent sequential
        wait and allows GPU parallelism over the circuit batch.

        Args:
            positions: 2-D array of shape ``(n_agents, n_features)``.

        Returns:
            1-D array of fitness values, shape ``(n_agents,)``.
        """
        circuits = []
        for pos in positions:
            x = np.clip(np.asarray(pos, dtype=float)[: self.n_qubits], 0.0, 1.0)
            circuits.append(self._build_circuit(x))

        # Single batched transpile + run
        transpiled = transpile(circuits, self.backend, optimization_level=0)
        job = self.backend.run(transpiled, shots=self.shots)
        result = job.result()

        fitness_values = []
        for i in range(len(circuits)):
            counts = result.get_counts(i)
            p_zero = counts.get(self._all_zeros_key, 0) / self.shots
            fitness_values.append(float(1.0 - p_zero))

        return np.array(fitness_values)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """
        Build: [angle encoding] + [variational ansatz] + [measure all].

        Args:
            x: 1-D array of length ``n_qubits`` in ``[0, 1]``.
        """
        qc = QuantumCircuit(self.n_qubits, name="QuantumOracle")

        # --- Angle encoding ---
        for i, val in enumerate(x):
            qc.ry(float(np.pi * val), i)

        # --- Variational ansatz (fixed weights) ---
        qc.compose(self._bound_ansatz, inplace=True)

        # --- Measurement ---
        qc.measure_all()
        return qc

    def __repr__(self) -> str:
        return f"QuantumOracle(n_qubits={self.n_qubits}, shots={self.shots})"
