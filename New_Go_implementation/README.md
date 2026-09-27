# New hybrid classical--quantum FewRel implementation

This folder contains the clean implementation specified in
`PROJECT_IMPLEMENTATION_SPEC.md`. It does not overwrite the exploratory
implementation in `../BTP_Quantum_few_rel`.

## Architecture

1. Frozen 384-dimensional `all-MiniLM-L6-v2` sentence embeddings.
2. Trainable `LayerNorm -> Linear(384, 64) -> GELU -> Linear(64, n_qubits) -> pi*tanh` compression.
3. Exact differentiable statevector simulation with angle encoding, trainable RY/RZ gates, and data re-uploading.
4. Density-matrix prototypes for each relation in the support set.
5. Fidelity scores converted to logits with a positive learnable temperature.
6. Cross-entropy loss over query examples in each 5-way/1-shot episode.

## Quick start

From this directory, install the packages in `requirements.txt`, then run:

```text
python -m pytest -q
python experiments/train_main.py --train-episodes 20 --eval-episodes 20
python experiments/evaluate_main.py --checkpoint results/smoke_checkpoint.pt
```

The default data path points to the existing project's `data/` directory.
Use `--data-dir` if the embeddings are stored elsewhere.

The initial smoke run uses 20 training and 20 validation episodes. A single
5w1s episode contains 5 support examples, 75 query examples, and 80 examples
in total.

The quantum block is an exact statevector simulation, not a shot-based
hardware approximation. This is appropriate for the small four-qubit smoke
experiment and keeps gradients available for both trainable parts.
