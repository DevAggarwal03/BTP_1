# Historical Torch-simulator implementation

This folder preserves the earlier implementation, which simulated quantum
gates directly with PyTorch. Its code and results are retained as a reference;
they are not the active Qiskit implementation.

The current Qiskit specification is in
[`PROJECT_IMPLEMENTATION_SPEC.md`](PROJECT_IMPLEMENTATION_SPEC.md), and its
implementation lives in
[`../New_Plus_Qiskit_Implementation/`](../New_Plus_Qiskit_Implementation/).
Use that folder for Qiskit training, evaluation, and new results. Checkpoints
from this Torch-simulator project are not compatible with the Qiskit model.
