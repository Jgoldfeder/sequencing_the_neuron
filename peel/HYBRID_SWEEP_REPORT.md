# Hybrid recovery: 3072 → 256 → 256 → 256

Implemented in `hybrid_sweep.py`, dispatched by `kink_solve.recover_layer`
with `sampling="design"` (the `run.py --design-refine` path).

The last 256 is a linear output layer: this fixture has two hidden layers.

## How it works

For the first hidden layer, the good guess supplies a right inverse to generate
candidate crossings while keeping guessed sibling neurons away from zero.
Actual forward teacher queries locate and fingerprint the kink. Coordinate
finite differences on its two sides estimate the change in the output Jacobian.
The solver shrinks the probe size until that change is numerically rank one,
and requires matching estimates from two distinct anchors before accepting.
The weight direction comes from its leading singular vector; the bias comes
from the kink location. An optional low-norm bias probe improves accuracy when
it succeeds; its failure does not discard an otherwise consistent solution.

For later hidden layers, the existing informative kink-point solver works in
recovered-prefix feature coordinates. No teacher gradients or hidden states
are used. The guess is essential for candidate construction, identification,
and orientation; this is not extraction from scratch.

Automatic selection under `--design-refine`:

- Single hidden layer, input dimension >=1024 and hidden width <=input width:
  existing shallow affine sweep.
- Multiple hidden layers, first-layer input dimension >=1024 and >=4 times
  first hidden width: new verified affine sweep for layer 1.
- Other hidden layers: existing informative kink-point solver.

Failed neurons retain their guesses. The `run.py --fast` refinement loop now
stops attempting deeper hidden layers when design refinement leaves an
incomplete prefix. Output least squares can still fit the resulting model.

## Measured experiment

Random double-precision leaky-ReLU network, teacher seed 0, normalized weight
rows, biases drawn with standard deviation 0.1 before normalization. Guesses:
weight noise standard deviation `0.01/sqrt(input_dimension)`, bias noise 0.005;
output guess zero. Teacher parameters are used only to construct the fixture
and measure errors. Layer 2 uses the actually recovered layer 1.
Errors below compare weights and biases in unit-weight gauge.

| Hidden layer | Recovered | Teacher input evaluations | Time | Mean absolute parameter error | Maximum absolute parameter error |
|---|---:|---:|---:|---:|---:|
| 3072 → 256 | 256/256 | 8,937,688 | 136.7 s | 3.30e-14 | 2.20e-12 |
| 256 → 256 | 256/256 | 8,002,662 | 208.0 s | 6.35e-13 | 1.94e-10 |

First-layer costs combine the original 248/256 run and a retry of its eight
rejected neurons after fixing the optional bias-probe gate. This is **not** a
clean full-run benchmark of the final implementation. Raw evidence:
`hybrid_full.json`, `hybrid_completed.json`, and their `.log` files.
Saved teacher/reconstruction: `hybrid_completed.pt`.

Output least squares used 4096 additional teacher inputs. On 1024 fresh random
inputs, output absolute error was mean 2.07e-11, maximum 1.34e-10
(`hybrid_output.json`). Hidden refinement total: 16,940,350 input evaluations,
344.8 seconds. Queries count input examples, not batched API calls.

This succeeds on this one good-guess fixture; it is not a general convergence
or machine-precision guarantee. The deeper layer loses accuracy, and the
query cost is still substantial. Square 3072-wide hidden layers do not take
this fast dispatch. No full training-from-scratch `run.py` benchmark was run
for this hybrid change.

## Reproduce from code/

```bash
OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  python peel/test_hybrid_sweep.py --channels 256 --second \
  --output peel/hybrid_rerun.json
```

This uses the production dispatch, measures both hidden layers, then fits and
validates the output. Change `--seed` for another random teacher. For a bounded
first-layer probe use `--channels 4` without `--second`.

CPU regression: `OPENBLAS_NUM_THREADS=1 python peel/test_hybrid_smoke.py`.
It checks production dispatch, forward-only recovery, query accounting,
normalization, zero-budget/empty requests, and abstention for a constant oracle.
