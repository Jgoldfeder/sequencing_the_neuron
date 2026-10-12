# Local direct MLP disagreement queries

`--peel-direct` now uses local reachable hidden-input charts rather than the
unstable global inverse. Applies after a complete frozen prefix in leaky-ReLU
MLP mode. It does not change the design refiner or CNN sampling.

Each outer iteration chooses a fresh random raw-input anchor. The recovered
prefix gives its hidden activation and local Jacobian. An SVD right inverse
maps hidden displacements to raw displacements, dropping singular directions
below 1e-5 of the largest singular value. Displacements are restricted to remain
inside the prefix's activation region, with a 20% boundary margin, and bounded
by the anchor norm (at least 1). The student runs from the frontier on the mapped
hidden inputs. The full teacher sees bounded raw inputs. Before storing labels,
the actual recovered-prefix forward pass is checked against the chart, and the
actual hidden inputs are stored. An invalid roundtrip raises rather than silently
storing mismatched examples.

This is **local direct access**, not unrestricted independent hidden queries.
At depth, the resolvable rank and hidden displacement range can shrink greatly.
Changing anchors gives new regions, but no convergence improvement has yet been
established. Model-only chart construction is not a teacher query. Construction
requires a prefix Jacobian and SVD each outer round; timing on the full target
training job has not been benchmarked. Prefix estimation errors remain a source
of teacher/student mismatch and cannot be certified by self-roundtrip checks.

## Numerical evidence

`test_direct_chart.py` uses the same cached trained teacher as the previous
failed global-inverse diagnostic, with an exact prefix as a diagnostic control.
The production chart uses only recovered weights. For one anchor and 128 queries
at each depth, measured double-precision direct-tail equivalence:

| Frozen prefix layers | Retained rank / 200 | Relative output error | Relative query-gradient error |
|---|---:|---:|---:|
| 1 | 200 | 3.61e-16 | 8.77e-15 |
| 2 | 160 | 3.97e-16 | 1.58e-13 |
| 3 | 103 | 4.09e-16 | 7.25e-13 |
| 4 | 65 | 3.88e-16 | 1.47e-12 |
| 5 | 64 | 5.01e-16 | 3.36e-12 |

Maximum raw coordinates stayed below 4. Contrast the historical global inverse:
~7e14 raw coordinates and ~2 relative output error after four frozen layers.
These are different query distributions: the new test deliberately restricts
queries to safe reachable regions. Hidden spread at depth five was only ~1e-6
(mean coordinate standard deviation), so numerical correctness does not imply
informative sampling. Training still uses fp32, not the precision in this table.
Raw evidence: `direct_chart_results.json`.

## Scale and buffer fixes

Warm MLP freeze now compensates the scale change in each frozen row by inversely
rescaling its outgoing column before installing the target weights. Pure positive
scale changes preserve the function; changes in direction are intentional
refinement and cannot preserve it exactly. Tested across row scales 0.01–100.
The old MLP helper copied frozen values without compensating downstream weights.

Output least squares and disagreement weighting now respect the hidden-space
buffer when direct mode is active. Legacy inverse code is retained for other
callers; its inaccurate normalization claim was removed.

## Validation and scope

```bash
OPENBLAS_NUM_THREADS=1 python peel/test_direct_chart.py
OPENBLAS_NUM_THREADS=1 python peel/test_direct_chart_pipeline.py
OPENBLAS_NUM_THREADS=1 python peel/test_design_pipeline.py
```

All passed. The new small pipeline test runs query optimization and Adam training,
refinement, freeze, restart, fp64 export and incomplete-prefix deferral. The
numerical test checks values, query gradients, bounded inputs, nonconstant hidden
queries, and warm function preservation. No long seven-layer training job was
started. This fixes the demonstrated large-inverse query corruption, not general
stochastic training convergence. Restart Python to load the changes.
