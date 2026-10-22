# Independent audit of LeNet recovery, 2026-10-21

The failure cannot currently be attributed solely to an intrinsic committee-learning ceiling. There is a reproduced numerical failure in the residual sampler, and the supplied audit omits its deterioration during the best run. This does **not** establish that fixing that failure solves all 120 C5 rows, conv2's last row, or F6.

Scope: inspected the supplied audit, relevant recovery/training/seeding/balancing/oracle code, and the best residual and conv2-stall logs; ran small CPU diagnostics against the attributable 103-row record and tracked teacher. Did not use any of the three collided snapshot families to establish results. Did not launch training, modify recovery code, or change existing records. Existing uncommitted edits were preserved. The code inspected is the current working tree; historical logs need not reflect every current edit.

## 1. High priority: fp32 prefix contamination produces bogus residual directions

**Confirmed from logs and reproduced in isolation.** `method.py:7261` clones the fp32 live member and calls `.double()`. That promotes already-rounded numbers; it does not restore the exact fp64 prefix/pinned rows. `residual_seed.py:78` then treats this prefix as exact when mapping image finite differences into frontier coordinates. Its rank decision at line 86 uses only `S > 1e-4*S[0]`, without a numerical-noise estimate or an impossible-rank check.

In `pool_logs/lenet_c5_resseed_noreinit.log`:

- line 66: 103 pinned, rank 17; the leading singular value is 14.
- line 383: still 103 pinned, rank 20, already exceeding 17 missing rows.
- line 418: 104 pinned, rank 69.
- line 455: 104 pinned, rank 173.
- lines 540–764: 106 pinned, rank 176–179, although only **14** rows remain. Lines 653 and 709 explicitly say the sampler uses those 176/179 directions.

The large leading signal disappears as rows are pinned. A threshold relative only to that leading signal then admits the pre-existing numerical contamination. For example, at line 652 the leading singular value is 0.1, the cutoff is about 1e-5, and unwanted singular values around 8e-5 pass.

Controlled reproduction (`probe.py`, CPU, same 8 Jacobian points / 6,400 queries per comparison): align the teacher to the real 103-row record, diagnostically promote the most active missing row using its exact teacher row, then change **only** whether the attacker's model goes through fp32. With 104 pinned and 16 missing:

| Prefix/pinned coefficients | Reported rank | First unwanted singular value |
|---|---:|---:|
| fp64 record | 16 | 3.70e-8 |
| Same record rounded through fp32 | 65 | 1.06e-4 |

This establishes a sufficient mechanism for the rank explosion. It does not attribute every contaminated direction in historical runs to rounding; finite-difference prefix-region mistakes, conditioning, and stale gauges can contribute too.

**Repair before another long residual run:** construct the Jacobian prefix and pinned rows from the persistent fp64 record in one consistent gauge, rather than promoting live fp32 weights. Reject or explicitly diagnose residual ranks above the remaining width. Compare step sizes and independent point batches to estimate noise. A rank cap is a defensive check, not a substitute for correcting the prefix or evidence that the retained directions are right.

## 2. High priority: the audit's “reachability fixed, only learning is slow” account is contradicted by its own log

**Confirmed.** Early residual batches have 0–1 silent C5 channels, e.g. lines 68 and 173 of the same best-run log. Following rank contamination, lines 542, 561, 580, 654, 673, 710 and 746 report **7–9 silent channels**. The deterioration is absent from F12/F13 and the bottom-line account.

The supported account is: seeding initially improved coverage; after additional pins, its estimated span degraded and coverage partially regressed. The experiments do not isolate slow learning under *sustained correct coverage*. They therefore cannot settle whether data weighting, balancing, or committee assignment is the main remaining cause.

Track coverage separately for seeded and ordinary queries, and track both signs/near-boundary occupancy rather than just one positive example. Keep rank and seed-inversion error alongside those metrics.

## 3. Medium priority: the residual span is a span of projected rows, not full missing rows

**Confirmed mathematically and measured on this teacher.** With known-row projector `P_perp`, the ideal residual is

`J_h P_perp = A_unknown(x) W_unknown P_perp`.

Consequently its row span lies in `rowspan(W_unknown P_perp)`. It generally does not contain the full `W_unknown`. The introductory claims in `residual_seed.py` that the residual is spanned “EXACTLY by the unpinned rows” and that its rank always equals missing width are too strong. Equality needs observability, enough varied downstream coefficients, independent projected rows, and a full-row-rank local prefix Jacobian. A zero residual rank alone is not a universal completeness certificate.

On the actual 17 missing teacher rows, their cosine with their best projection into the exact projected residual span ranges from **0.647 to 0.828**. Thus the initialization in `reinit_unpinned_` cannot put a row close to its true full direction merely by drawing it inside this span. Training can subsequently leave the span, so this is an initialization limitation, not a permanent expressivity restriction. It does not explain the no-reinitialization arm by itself.

Use the projected span for exploration. If initializing full rows from it, allow or fit a component along the known-row span. For direct rank-one jump recovery, retain the full unprojected Jacobian jump once a target kink is isolated.

## 4. Medium priority: headline query counts omit substantial black-box work

**Confirmed accounting behavior.** `qtot` increases only by `cfg.q` at `method.py:7302`. Residual finite differences, recovery probes, and certificates are separate in `refinement_queries`; final results retain separate fields. The printed `q` column is not total oracle consumption.

The best-run log contains 17 residual-subspace rebuilds at 32,000 queries each: **544,000 additional queries**, before counting kink refinement and certificates. Sixteen rebuilds precede the iteration-51 headline of 1,020,000 sampler queries, so that point costs at least **1,532,000** combined queries, plus refiner/certificate calls. The next rebuild is after iteration 51. This matters when comparing sampling strategies by query efficiency.

Report sampler queries, finite-difference queries, refiner/certificate queries, and their sum. Keep the oracle compute/precision contract explicit: these finite differences query a double-precision copy of the teacher, while normal training labels use the original teacher.

## 5. Medium priority: the standalone kink script is oracle-assisted beyond scoring

**Confirmed code behavior.** In `c5_residual_kink.py`, `prefix()` and `prefix_preacts()` directly read the teacher's prefix. More significantly, the known set is chosen using teacher C5 activation fractions, and `Wk`/`bk` are copied from teacher C5 rows. Later, the accepted row's sign is selected using the teacher-based `score()` result. The header's “oracle used ONLY for scoring” statement is false as written.

Supplying an exact recovered prefix/known set can be a legitimate diagnostic assumption, but this script does not load those assumptions from an attacker record. It is an oracle-assisted prototype; it is not an end-to-end honest extraction result. Hyperplane orientation also needs an attacker-side rule rather than a scoring-derived sign when presenting a deployable leaky-ReLU recovery.

The production `residual_seed.py` forward-only use is distinct: its `_bb` accesses teacher outputs, not teacher row values. `--oracle-excite` only prints diagnostics in the inspected path.

## 6. Lower priority: initial balancing violates its documented raw-pin invariant, but is not proven to corrupt the function

**Confirmed invariant mismatch; ruled out as a standalone proof of lost pins.** The first-epoch branch at `method.py:7344` calls `_bal_canon` on all hidden layers, bypassing `_bal_eligible`, even on a resumed pinned prefix. Subsequent refreshes do consult eligibility. `cnn_balance.py` says pinned rows are never rescaled.

A controlled call on the record changes pinned raw weights by maxima of 0.55, 0.17 and 0.23 in the three conv layers. However, the operation preserves the function (max error 1.5e-15) and canonical weights (3.3e-16) in fp64. It is therefore a gauge/bookkeeping risk, not evidence that exact hyperplanes were destroyed. Any raw record overlays must respect those gauges; the current candidate injection does use `_cnn_transfer_rows_`.

Either honor eligibility from initialization onward or document and validate record/gauge transforms explicitly. Do not blame row-norm balancing for the entire stall without an ablation.

## 7. Conv2's last row remains undiagnosed

The stalled run logs `0 consensus candidates` at 15/16 repeatedly, e.g. lines 673–714. At those passes the refiner has no new candidate to reject. Raising its angle gate alone cannot fix an upstream absence of candidates.

The sampler frontier uses `frozen`; partial peeling uses `partial_mask | frozen`. These can differ in general, but I did **not** establish that this mismatch caused the reported conv2 stall: that run explicitly freezes conv1 at line 106 and therefore samples at conv2. Spatial boundary seeding defaults off, and residual seeding deliberately skips conv2's patch rows. “Run with residual seeds” does not mean the stalled conv2 row received that intervention.

A short diagnostic should identify the remaining teacher channel for scoring only; measure its teacher firing on mined inputs, every member's nearest-row distance/sign/bias, duplicate matches, gradient norms, and sampler coverage of its spatial kink. These measurements distinguish starvation, alternative linear representations on the sampled regions, and optimization failure. None is established yet.

## Claims independently supported and limits

- The attributable file named `_record_lenet_c1c2_c5_105.pt` actually contains **6, 16, 103** pinned rows.
- Aligning its canonical fp64 rows to the tracked teacher yields maxima **5.24e-14, 5.88e-13, 3.10e-10**, independently confirming its stored-row accuracy.
- The best log explicitly records **104, 105, then 106** pinned C5 rows, with the stored-row check at line 538 reporting 3.10e-10. This supports a historical logged result, not an attributable recoverable 106-row snapshot.
- F6 has been trained as part of the surrogate tail and has consensus diagnostics; it has not been reached as the exact extraction frontier. “Never started” needs that distinction. The head is also trained, while exact head recovery requires a solved prefix.
- I did not independently reproduce the pooling/stride multi-seed study, MNIST silent-channel counts, or the offline oracle-ascent study. Their referenced memory note/scratch scripts are not self-contained evidence in the supplied audit. F5's MNIST count also differs from the bottom-line count (10 versus 9).
- The shared snapshot collisions remain disqualifying for attribution. No collided snapshot was used here.

## Recommended next experiment

First repair the numerical prefix/rank issue and require the controlled 104-row reproduction to return rank 16 without oracle row substitution in the actual pipeline. Start from the attributable 103-row record; use a unique output path and immutable checkpoints. On a short run, require the estimated rank to remain at most the missing width and coverage to stay improved after every new pin. Only after that passes compare weighting or balancing settings under equal **total** query budgets. Continue to require global kink verification before recording rows as exact.

The new probe's teacher-based promotion and channel matching are explicit **audit diagnostics**, not honest recovery steps. Reproduce with `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python audit_lenet/probe.py` from the code directory. Numeric results are in `probe_results.json`; the original logs and records are untouched.
