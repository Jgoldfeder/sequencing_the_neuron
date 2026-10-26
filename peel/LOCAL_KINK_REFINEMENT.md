# Independent local-kink refinement (2026-10-26)

Experimental; not a completed deep-network recovery or a machine-precision certificate.

`local_kink_refine.py` uses only forward calls to its teacher argument (a plain
callable is supported). It projects independently drawn inputs onto the guess's
surface, chooses informative brackets, and uses `locate_light` (eight input
queries per bracket). It explores widths then favors productive widths. It
retains raw input kink coordinates so a corrected prefix can reuse the queries.
Fits use disjoint held-out points. Attempts are bounded; rejected rows stay
unsolved. A close guess is used for surface proposals and orientation, not as
a reason to waive numerical checks.

Driver opt-in:
```
CUDA_VISIBLE_DEVICES=0 python -u refine_snapshot_mlp.py \
 recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt \
 --member 2 --local-refine --local-rounds 24 --fallback-rounds 16 \
 --out recon/_refined_mlp9_local.pt --unit-report
```
L1 still uses hybrid sweeps. The local path replaces tracking at later layers;
design fallbacks are limited to the configured number of rounds per candidate.
Guess-free discovery has a separate existing line budget. This is not a strict
global query cap. Existing behavior remains selectable without `--local-refine`.

Driver correctness fixes:
- Candidate fallbacks reject duplicates against accepted rows.
- Guess-free exclusion now receives refined rows, rather than stale guesses
  marked as solved.
- Save partial rows before alternative guesses and full state after every layer.
- Do not fit the output layer after an incomplete hidden layer; record completion
  metadata rather than implying every checkpoint is a solved network.

Validation:
`python peel/test_local_kink_refine.py` checks a plain forward-only oracle,
counted/bounded queries, exact-prefix accuracy, point retention, and duplicate
rejection. `test_local_mlp9.py` evaluates selected rows on the real fixture;
`run_local_chain.py` tests all 200 rows from the imported L1 checkpoint. Its
reported new queries exclude the historical imported-prefix cost, explicitly
recorded in its checkpoint.

Initial real-fixture tests (before adaptive widths):
- L3 channels 0..7: 8 accepted, 102,144 queries, 8 seconds, aligned max error
  4.56e-9 using the saved tracking L1/L2 prefix.
- L5 channels 0..7: none accepted with the saved tracking L1..L4 prefix;
  113,664 queries. Seven fits have numerical residuals consistent with a
  damaged prefix; one is a poor fit. Checks were not loosened to accept them.

These tests do not establish complete recovery or solve prefix-error propagation.

## Full-row and archive-reuse measurements

The first full run (before the cheap retry was added) completed L2 and L3:
- L2: 200/200 after design fallback; 4,457,290 additional queries; max W error
  7.61e-11, max bias error 2.90e-9. This was not cheaper than tracking.
- L3: 200/200 after design fallback; 4,413,118 layer queries; max W error
  3.89e-9, max bias error 1.18e-9.

A subsequent replay of L3's 29 rejected rows reused their original kink points.
Stronger robust fitting proposed a new plane; fresh narrow brackets validated
and refitted it. This recovered 21/29 with only 96,768 NEW forward queries,
maximum aligned W error 2.57e-9. The original point-acquisition cost is not
included in that incremental count. Eight remain rejected. This retry is now
built into local refinement; it does not waive the validation threshold.

Fresh retry budget is at most eight batches per channel. Thus the local oracle
bound is 16 * batch * channels * (rounds + 8), including fresh retry validation.
Raw input archives are large (about 2.3 GB per 200-row layer at input width 3072).
They are retained intentionally for query-free refits after prefix corrections.

Actual CLI smoke tests: `python peel/test_refine_snapshot_driver.py` exercises a
small complete network and an intentionally stopped partial run. No claim that
the full 9-layer target has been solved or that prefix amplification is fixed.


## Optional float32-prior precision cleanup

This is an explicit additional assumption: the teacher stores float32 parameters
and evaluates their double-precision cast. It is not applicable to a generic
float64 teacher. `float32_lattice.py` searches possible positive scales of a
recovered row using only that row. It fixes coordinates whose float32 rounding
is unambiguous. `precision_prefix.py` solves the remaining uncertain coordinates
and bias from kink constraints, then checks separate held-out points. Corrected
row scales are compensated in the next layer's columns. Teacher weights enter
only the experiment's post-hoc diagnostic error report.

Measured full-prefix cleanup, starting from the saved local recovery and reusing
its raw kink archives:

| Layer | New cleanup queries for this layer | Max normalized W error | Max bias error |
|---|---:|---:|---:|
| L1 | 979,968 | 3.29e-15 | 0 |
| L2 | 189,568 | 5.55e-17 | 1.39e-17 |
| L3 | 112,096 | 8.33e-17 | 6.94e-18 |
| L4 | 162,312 | 8.33e-17 | 6.94e-18 |
| L5 | 665,512 | 1.11e-16 | 6.94e-18 |

All 200 rows per layer passed; 2,109,456 **additional** queries total. This excludes
all earlier committee and kink-archive acquisition queries. Checkpoints are
`recon/_precision_L1.pt` through `_precision_L5.pt`; logs are
`pool_logs/precision_prefix_full.log` and `pool_logs/precision_resume.log`.

`python peel/test_precision_cleanup.py` tests lattice recovery, sparse coefficient
correction, held-out rejection, and rejection of an arbitrary float64 row.
`resume_precision_prefix.py` retries only rows that failed and saves fresh raw
points. L6 and the complete network remain unfinished at this update; the older
canonical L6 checkpoint has 199/200 approximate recovered rows.
