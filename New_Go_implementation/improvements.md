# Improvement TODO

This file tracks the planned improvements for the hybrid classical--quantum
FewRel model. The current 8-qubit model averages about 59% accuracy across
three seeds, compared with about 54% for the 4-qubit model. Random guessing in
a 5-way episode is 20%.

## 1. Establish classical baselines

- [x] Run a trained classical Euclidean ProtoNet.
- [x] Run a trained classical cosine ProtoNet.
- [x] Use the same 5-way/1-shot episode definition as the quantum model.
- [x] Use 600 training and 600 validation episodes per seed.
- [x] Use seeds 7, 17, and 27.
- [x] Compare the classical results against the 4-qubit and 8-qubit results.

The baseline models use the same 384-dimensional Sentence-BERT input and the
same 64-unit nonlinear compression shape, ending in an 8-dimensional vector.
They differ only in the classical distance used to compare query vectors with
class prototypes. This is a trained classical comparison, not an untrained
nearest-centroid result.

### Baseline result

| Model | Mean accuracy | Accuracy standard deviation | Mean weighted F1 |
|---|---:|---:|---:|
| Classical Euclidean ProtoNet | 59.29% | 1.45% | 56.75% |
| Classical cosine ProtoNet | 59.41% | 0.85% | 56.20% |
| Quantum, 8 qubits | 59.34% | 0.37% | 56.50% |
| Quantum, 4 qubits | 53.71% | 2.73% | 51.04% |

The current quantum model is therefore competitive, but it does not yet show
a clear accuracy advantage over the simpler classical baselines. The next
improvements should focus on the input representation, compression layer, and
training settings rather than assuming that adding more quantum structure will
automatically improve accuracy.

## 2. Improve training

- [x] Try 2,000 training episodes.
- [x] Try 5,000 training episodes with the strongest learning rate from the
      2,000-episode sweep.
- [x] Compare learning rates such as `0.0003`, `0.001`, and `0.003`.
- [ ] Compare Adam and AdamW.
- [ ] Plot training loss and validation accuracy to detect under-training or
      overfitting.

### Training sweep result

These runs used the 8-qubit model, seed 7, and 600 validation episodes. The
2,000-episode sweep tested all three learning rates. The 5,000-episode run
used the best 2,000-episode rate, `0.001`.

| Training episodes | Learning rate | Accuracy | Weighted F1 | Runtime |
|---:|---:|---:|---:|---:|
| 2,000 | 0.0003 | 61.48% | 58.50% | 5.9 min |
| 2,000 | 0.001 | **61.58%** | **58.54%** | 6.4 min |
| 2,000 | 0.003 | 58.68% | 55.72% | 6.3 min |
| 5,000 | 0.001 | **61.74%** | **58.95%** | 14.0 min |

Training longer helped compared with the 600-episode, seed-7 8-qubit result
of 59.20%, but the improvement from 2,000 to 5,000 episodes was small. The
learning rate `0.003` was too aggressive in this test. The best setting still
needs to be checked across multiple seeds before it becomes the new default.

## 3. Improve the classical compression layer

- [ ] Compare the current `384 -> 64 -> qubits` network with
      `384 -> 128 -> 64 -> qubits`.
- [ ] Test L2-normalising Sentence-BERT embeddings before compression.
- [ ] Test a learnable angle scale instead of always using `pi * tanh`.
- [ ] Compare linear and nonlinear compression with the same episode list.

## 4. Improve quantum capacity carefully

- [ ] Use the 8-qubit model as the main candidate because it performed better
      in the initial multi-seed test.
- [ ] Test three and four data-reuploading repetitions.
- [ ] Test additional input rotation types and entanglement patterns.
- [ ] Keep exact statevector runtime and memory recorded for every setting.

## 5. Improve the text representation

- [ ] Compare the current entity-marked MiniLM embeddings with a stronger
      frozen sentence model.
- [ ] Test an entity-pair text format that explicitly names the head entity,
      tail entity, and surrounding context.
- [ ] Consider fine-tuning the text encoder only after the frozen-embedding
      baselines are understood.

## 6. Test the effect of support examples

- [ ] Run a separate 5-way/5-shot experiment.
- [ ] Keep 5-way/1-shot as the main target so the comparison remains fair.

## Rules for interpreting improvements

- Change one major factor at a time.
- Reuse the same validation episode list when comparing models.
- Keep the seed and configuration in every result file.
- Do not claim that a change helps until it improves the mean across multiple
  seeds.
