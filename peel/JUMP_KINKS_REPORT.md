# Multi-region Jacobian-jump experiment

The prototype reduces forward-query counts, but does **not** achieve the existing
kink-point solver's accuracy with an exact prefix. It is not wired into production.

## Reproduce

From `code/` (choose an idle GPU):

```sh
OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python peel/test_jump_kinks.py --widths 200 400 --hidden-layers 7 --layer 3 --channels 4 --regions 8 --prefix-noise 0 1e-10 --output peel/jump_comparison_f64.json
```

Layer indices are zero based. These are four-neuron tests of hidden layer 3,
not full-layer or end-to-end timings. Teacher seed 0, float64, leaky slope .01,
input dimension equal to width, seven hidden layers, 100 outputs. Guesses have
weight noise .01/sqrt(width), bias noise .005. The perturbed prefix has Gaussian
weight noise 1e-10/sqrt(width), bias noise 1e-10 at each preceding layer; this is
controlled perturbation, not an actually recovered prefix. Downstream guess
weights/biases are zero. Both solvers receive only a forward callable teacher.

## Results

Errors compare unit-weight rows and corresponding biases to the teacher.
Means include every weight and bias for all four tested neurons. Jump errors
below use the dense diagnostic reference; the iterative results agree closely.
Wall time includes query generation, measurements and **both** dense and iterative
fits for the jump method. The baseline runs only its usual fit. These timings
therefore do not isolate the fastest possible jump implementation.

| Width | Prefix noise | Method | Teacher samples | Seconds | Mean absolute error | Max absolute error |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 200 | 0 | jump | 66,736 | 2.73 | 5.76e-11 | 1.78e-09 |
| 200 | 0 | points | 227,054 | 1.48 | 1.98e-15 | 7.86e-14 |
| 200 | 1e-10 | jump | 66,776 | 2.46 | 1.29e-10 | 3.31e-09 |
| 200 | 1e-10 | points | 227,354 | 1.36 | 4.81e-11 | 2.41e-09 |
| 400 | 0 | jump | 141,436 | 8.54 | 1.74e-10 | 5.06e-09 |
| 400 | 0 | points | 537,142 | 9.21 | 2.12e-15 | 8.58e-14 |
| 400 | 1e-10 | jump | 141,586 | 8.93 | 1.86e-10 | 7.30e-09 |
| 400 | 1e-10 | points | 525,566 | 9.25 | 4.04e-11 | 4.32e-09 |

All four baseline neurons passed its heuristic acceptance checks in each case.
The jump prototype reports estimates, not an acceptance certificate. All jump
neurons acquired eight regions. No 3,000-wide run has been performed.

## How it works

`jump_kinks.py` uses the noisy guess to propose target crossings. The existing
forward-only scan/fingerprint identifies teacher kinks. At each kink, coordinate
sweeps estimate the Jacobian on each side, at two finite-difference step sizes.
Disagreement and non-rank-one jumps trigger smaller steps. A singular vector of
the accepted jump estimates its input-space normal.

For region r, the measured normal n_r should be parallel to P_r^T w, where P_r
is the reconstructed prefix Jacobian. The joint solve imposes
`(I - n_r n_r^T) P_r^T w = 0` and `h_r^T w + b = 0`.
It pins one coordinate to fix scale and normalizes the resulting weight.
Prefix products use weights and activation masks, without inverse Jacobians.
LSMR uses a matrix-free operator; dense least squares is a diagnostic reference.
The analytic regression test checks both fits, fp64 masks, a known forward-only
jump and teacher sample accounting:

```sh
OPENBLAS_NUM_THREADS=1 python peel/test_jump_algebra.py
```

## Interpretation and limitations

Queries dropped approximately 3.4–3.8 times at these settings, but exact-prefix
parameter errors remained around 1e-9 instead of 1e-13. The analytic joint-system
test passes to better than 1e-10, and measured-data dense/iterative fits agree.
Together with observed finite-difference disagreement around 1e-9, this suggests
measurement precision, rather than iterative convergence alone, is the obstacle.
This diagnosis is evidence from this experiment, not a universal impossibility
result. With a perturbed prefix, the point solver also loses accuracy.

Each accepted region costs 8 * input_dimension teacher samples per attempted
step size, plus anchor scans. With a fixed region count and input dimension n,
this is O(n) queries per neuron and O(n^2) per full width-n layer. Dense teacher
forward cost remains O(depth*n^2) per sample, so full-layer arithmetic is not
linear in width. Matrix-free fitting avoids stacking a dense joint system, but
its iteration count can grow with conditioning; measured counts were hundreds
to thousands. This script deliberately also runs the dense diagnostic fit, so
it is not configured as a memory-bounded 3,000-width production implementation.

The rank-one and step-size checks are heuristic. There is no independent held-out
crossing acceptance gate yet. Accuracy is evaluated against synthetic teacher
parameters only in the benchmark. A useful next experiment is a hybrid in which
jump measurements guide the choice of kink-point constraints; these results do
not establish that such a hybrid will recover machine accuracy faster.

## Files

- `peel/jump_kinks.py`: experimental solver.
- `peel/test_jump_kinks.py`: configurable comparison script.
- `peel/test_jump_algebra.py`: analytic regression checks.
- `peel/jump_comparison_f64.json` and `.log`: final benchmark data.
- `peel/jump_comparison.json` and `.log`: preliminary run before an fp32 mask
  construction bug was fixed; do not use as final results.
- `peel/jump_smoke.json` and `.log`: initial width-40 smoke test, also before fix.
