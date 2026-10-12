# CNN solver and pinning audit — 2026-09-29

## What the supplied run actually shows

L2's accepted count increases 5 → 6 → 10 → 12 and stays at 12 across the last
restart. `cons 14/16` includes two unrefined candidates. A subsequent `cons 12/16`
does not mean two solved channels were lost. The stored fp64 errors remain near
5e-14. Ordinary committee copies are fp32, so differences around 1e-8 are expected.

The rejection messages were misleading: `_gate` imposed `min(angle_gate, 4)` but
printed 12 degrees. Thus 4–6 degree fits with 1e-14 residuals were rejected solely
by an undisclosed angle threshold. The previous independent-sampling retry was
working on these cases; the later angle cap prevented accepting its result.

`pin-check` compared raw rows in different input gauges and counted only partial
hooks, not freeze hooks installed by a retry. That explains its 8e-5 discrepancies
and -1 mask counts without establishing that pins were lost. There were also real
state-transfer defects that the audit reproduced separately.

## Changes

- CNN recovery honors the configured angle threshold. A fit now also needs
  held-out residual agreement, a unique identifiable normal, singular-value gap,
  conditioning-scaled residual checks, and closest-guess channel identity. This
  addresses weak fits and precisely fitted sibling surfaces without imposing a
  hidden universal 4-degree basin. Rejections print the additional checks.
- CNN tracked fits collect Din+80 points and reserve 8–32 interleaved points
  before fitting. Independent multiscale retries remain bounded to eight rounds.
  Rank-deficient/nonfinite fitting failures abstain. BLAS thread pools are scoped
  to one thread during recovery, as in the MLP design solver.
- Transfers from consensus into the fp64 record, and from that record into raw
  members, account for **both** source and destination prefix scales. The previous
  injection accounted only for destination scales. Warm freeze/rebuild also maps
  the record into each member's own gauge rather than copying raw values.
- **Correction, 2026-09-30:** the audit incorrectly changed partial peeling to
  require unanimity without authorization. This caused candidate starvation in
  the subsequent run. The original configured cluster quorum is restored:
  `max(2, round(cluster_quorum * p))`, default 5 of 8 members. A dissenting member
  does not veto a qualifying cluster; accepted rows are still pinned across the
  committee. The refiner and state-preservation fixes remain in place.
- Frozen masks are tracked, move with channel permutations, and participate in
  pin diagnostics alongside partial masks. Retry banks are retained in `frozen`.
  Diagnostics compare canonicalized rows, not incompatible raw input gauges.
- `--fast-peel-partial --peelrefresh` enables full-layer freeze/refresh without
  requiring an additional `--fast-peel`. Extended peel and resume do not advance
  refinement beyond an incompletely recovered prefix.
- Resume, partial and peel-try records use fp64. A local-import shadowing error
  that crashed CNN resume is fixed. Snapshots include fp64 solved state and masks;
  resume prefers `refined_state`; final delivery overlays solved rows in a common
  canonical gauge. Final selection uses the current population.
- CNN kink-refinement query counts are recorded separately from training queries.

MLP sampling/acceptance behavior is otherwise unchanged. No running training
process was stopped or updated in place; restart Python to load these changes.

## Validation

The cached trained LeNet has hidden widths 6, 16, 120 and 84. These tests construct
nearby guesses (relative weight noise 0.01, bias noise 0.005) and erase the guessed
downstream weights. Teacher parameters construct/score the fixtures; recovery
itself uses teacher forward queries and the guessed/recovered prefix.

| Trained leaky-ReLU LeNet layer | Exact-prefix test: accepted / total | Worst error | Recursive test: accepted / total | Worst error |
|---|---:|---:|---:|---:|
| Conv1 | 6/6 | 3.62e-14 | 6/6 | 3.62e-14 |
| Conv2 | 16/16 | 6.93e-13 | 16/16 | 9.25e-13 |
| Conv3 | 120/120 | 6.00e-11 | 120/120 | 2.36e-10 |
| FC84 | 84/84 | 1.32e-11 | 84/84 | 4.42e-9 |

Errors are maximum coordinates of augmented unit [w,b] rows. The exact-prefix
run used 3,230,508 queries and 128 seconds. The recursive run used 8,264,546
queries and 251 seconds. It uses actual recovered
prefix directions; it restores their teacher scale only to keep later-layer error
comparisons in the teacher gauge. Its frontier guesses are still deliberately
constructed good guesses, not a learned committee. The exact-prefix sweep ran
before the final closest-guess check; the recursive sweep includes that check.
Raw results: `cnn_audit_lenet_full.json`, `cnn_audit_lenet_recursive.json`.

A separate mutable saved-best checkpoint at iteration 20 had three learned L2
guesses 7.77, 6.32 and 5.62 degrees from their targets. With an aligned exact L1
prefix, all three recovered with errors at most 3.03e-14 (557,671 queries). Earlier
iteration-15 guesses at 10.05 and 8.94 degrees also recovered below 4.9e-14.
This diagnoses the hidden-cap issue; it does not replay the historical consensus.

Additional coverage:

- Non-square, two-channel input; convolution stride/padding, average pooling,
  convolution-to-FC transitions; three random seeds and both ReLU/leaky-ReLU.
  64/72 accepted, worst accepted error 5.56e-12. All leaky-ReLU cases completed;
  remaining abstentions were ReLU FC cases with deficient sample rank.
- Trained ReLU LeNet: all nine sampled convolution channels recovered; all three
  sampled FC channels abstained on rank-deficient samples. Tiny residuals alone
  do not identify their weights. This is a remaining coverage/recovery limitation.
- CPU state regressions: arbitrary positive gauges, warm/cold freeze, actual Adam
  updates, channel permutations, fp64 output, rank-deficient and contaminated fits.
- Actual production refiner in a small three-hidden-layer CNN: partial peel,
  warm refresh, cold retry retaining partial solutions, configured-quorum enforcement (including a dissenting member),
  incomplete-prefix deferral, and resume. Worst solved-row error across these
  integration cases was 5.55e-13. Checks include query accounting and snapshots.
- The user's exact flag combination is parsed and its configuration asserted,
  with teacher training and reconstruction intercepted before external artifacts
  are written. The long training command was not relaunched.

## Reproduce from code/

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python peel/test_cnn_state.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python peel/test_cnn_pipeline.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python peel/test_cnn_cli.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python peel/test_cnn_checkpoint.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python peel/test_cnn_audit.py --suite trained --act leaky_relu --all-channels --recursive --seeds 1 --device cuda:2 --output /tmp/cnn_recursive.json
```

The checkpoint test intentionally depends on the latest local snapshot and can
change as training updates it. The audit is evidence for these architectures,
seeds and good-guess regimes—not guaranteed recovery of arbitrary CNNs, dead
ReLU units, ambiguous channels, arbitrary prefix errors or poor guesses. Held-out
tracking points can be correlated, and coherent prefix errors can bias both fit
and validation. The numerical gates are heuristic, not accuracy certificates.


## Follow-up: real-run sampling failures — 2026-09-30

The latest log's repeated `fallback found too few kinks` was reproduced using
an iteration-20 saved partial record and its actual recovered prefix. Channels
0 and 15 failed together: 0/2 recovered, 1,061,304 queries, 69.8 seconds.
Their augmented row norms were 0.0222 and 0.1572. Sampling radii incorrectly
depended on these arbitrary positive scales, producing excessively wide searches.
CNN bracket seeding and tracking now use unit augmented normals internally;
the guessed model and returned weight gauge are untouched.

With this fix, the same saved rows recover 2/2 in 243,881 queries and 11.7 seconds,
with maximum coordinate errors below 6e-14. No exact teacher prefix was substituted.
Raw results: `cnn_sampling_scale_record.json`.

A second defect affected identity checking: unrequested rows in the persistent
partial record can be stale. They must not veto a fresh candidate merely because
they lie closer to its recovered normal. Closest-guess comparison now uses the
requested candidate set; production deduplication against solved rows remains.
Refreshing channels 0, 2 and 9 from the saved best member recovers 3/3 in 387,231
queries and 14.6 seconds, errors below 7.2e-14. These match three distinct teacher
filters, none already solved. Unsolved channel labels can be permuted (candidate
9 matches teacher filter 15). This is a saved-best test, not a historical-consensus
replay. Raw results: `cnn_sampling_scale_refreshed.json`.

`test_cnn_sampling_scale.py` checks three frontiers across a 2500x row-scale range,
model immutability, and a stale unrequested competitor. The configured partial
consensus quorum remains unchanged (5/8 with the user's configuration).

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python peel/test_cnn_sampling_scale.py
# Uses a local saved snapshot; provide the desired snapshot explicitly.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python peel/test_cnn_sampling_checkpoint.py --snapshot /tmp/cnn_sampling_snapshot.pt --channels 0,15
```

## Duplicate false-completion fix

The supplied log exposed a second acceptance path: partial peel rejected and
rerolled a duplicate, but full-layer peel reused that iteration's stale consensus
and accepted the same filter without deduplication. The full-layer kink loop now
checks every proposed row against existing exact rows and against the same batch.
Rejected duplicates remain unsolved and are rerolled. Partial-path rerolls clear
the affected consensus bits; a per-pass invalidation set also prevents stale
oracle-gated/full-layer triggers or extend-through from revisiting that layer.
Partial checks include solved rows retained in `frozen` after retries.

`test_cnn_duplicate_freeze.py` reproduces the bypass with controlled duplicate
solver results through production reconstruction and checks both partial+full
and full-only paths. It passes, as does the existing CNN pipeline suite. No
teacher-error threshold was added to acceptance, and the configured quorum is
unchanged. This prevents future false freezes; it does not repair already saved
bad prefixes or prove that rerolls will discover the missing channel. Resume
from a known-good earlier checkpoint or explicitly repair the corrupted record.

```bash
OPENBLAS_NUM_THREADS=1 python peel/test_cnn_duplicate_freeze.py
OPENBLAS_NUM_THREADS=1 python peel/test_cnn_pipeline.py
```
