# Precision continuation — 2026-10-26

Goal remains complete forward-only recovery from committee guesses. NOT yet a
solved output network. GPU0 used, no other jobs modified.

## Confirmed improvement

The explicit float32-parameter prior eliminates the measured precision cascade.
Teacher weights are used only for diagnostic scoring, not in lattice search,
kink proposals, fits, or acceptance. This assumes float32 stored parameters
queried after a double cast; it does not solve arbitrary float64 teachers.

| Layer | Accepted | Max normalized W error | Max bias error |
|---|---:|---:|---:|
| L1 | 200 | 3.29e-15 | 0 |
| L2 | 200 | 5.55e-17 | 1.39e-17 |
| L3 | 200 | 8.33e-17 | 6.94e-18 |
| L4 | 200 | 8.33e-17 | 6.94e-18 |
| L5 | 200 | 1.11e-16 | 6.94e-18 |
| L6 | 200 | 8.33e-17 | 1.73e-18 |

Best prefix: `recon/_precision_L6_complete.pt`. It has all first six layers
correct, but the remaining tail is still a guess. Do NOT regard its outputs as
solved. Earlier precision checkpoints `_precision_L1.pt` through `_precision_L5.pt`.

Cleanup queries: 2,109,456 for L1–L5; 2,622,016 for the 199 known L6 rows;
13,064 for the newly discovered L6 row. These are ADDITIONAL to the historical
committee/anchor acquisition and exclude missing-neuron discovery experiments.

## Missing L6 neuron: solved via an L7 guess

A naive gradient mapback scan, residual-fit optimization, and direct prefix
inversion did not solve it. Prefix inversion needed huge inputs and was rejected
without spending teacher queries on that route.

Successful mechanism:
1. With 199 exact L6 neurons, form augmented features [their activations, H5].
   On the missing neuron's usual negative branch, an L7 preactivation is affine
   in these 399 features. Use an actual close committee L7 row as the guess.
2. Collect L7 kink points: 46,312 queries, 438 points. One known L6 feature never
   changed sign, making the affine remainder unidentifiable. Pinning its outgoing
   coefficient to the guess yielded missing-normal cosine .966 (diagnostic).
3. Target points on the same L7 surface where that known L6 feature changes sign.
   11,784 queries removed the ambiguity and recovered the missing normal to
   about 3e-10 max coordinate error.
4. Search for bias with this normal. Early narrow and poorly covered searches
   failed. A covering scan yielded a three-point bias proposal; 2,560 fresh
   queries confirmed it on 40 independent kinks. Bias error 5.5e-12. The rare
   positive branch selects orientation; this is a stated heuristic, confirmed
   diagnostically on this fixture.
5. One-row local refinement plus float32 cleanup: 13,064 queries, exact row.

Discovery query accounting (do not report only the successful branch):
- initial augmented points 46,312
- targeted flat feature 11,784
- failed initial bias scan 56,430
- failed quantile grid 89,596
- failed light cluster grid 24,576
- wide covering bias scan 391,972
- independent bias confirmation 2,560
Total for this successful discovery route INCLUDING its bias-search failures:
623,230, plus 13,064 precision queries. Other earlier failed discovery experiments
are separate and remain in their logs (including the 118,520-query residual-seeded
projected guess-free experiment). Some experimental checkpoint `discovery_queries`
fields count only successful-branch calls and are NOT total expenditure.

Scripts/logs:
- `peel/augmented_missing_probe.py`, `pool_logs/augmented_missing_reuse.log`
- `peel/target_augmented_flat.py`, `pool_logs/augmented_target.log`
- `peel/recover_missing_bias_wide.py`, `pool_logs/missing_bias_wide.log`
- `peel/validate_missing_bias.py`, `pool_logs/validate_missing_bias.log`
- `peel/finish_precision_L6.py`, `pool_logs/finish_precision_L6.log`

## Current continuation

`peel/finish_deep_tail.py` aligns committee guesses against the recovered prefix,
selects actual rows by agreement, carries accepted rows, collects independent
kink points, and applies the prior. A small L7 test reached 7/8 at numerical
precision. Full L7/L8 run logging to `pool_logs/tail_precision_full.log`.

Some proposed duplicate transfer applies ONLY when a missing member row actually
matches another member row (cos > .98). In the latest full run this triggered for
member0 only. Do not attribute all improvement to that step; row-wise committee
selection and accumulated points changed too.

Current on-disk local sampler now accepts explicit width schedules. The running
full-tail process was started before this change, so it still starts with narrow
widths. Future calls should start (.01,.02,.005,.04,...) for committee-level guesses
and use narrow widths only for already-refined guesses. Current on-disk tail
script requests 240 points; the already-running process requested 320.

No final output least-squares solve yet. Need finish L7/L8, fit output, validate on
fresh forward queries, and consolidate experimental helpers into a clean runner.

Tests: `python peel/test_precision_cleanup.py` passes lattice recovery, sparse
correction, held-out rejection, unsupported float64, and duplicate frozen-row
rejection with zero oracle calls. Existing earlier forward-only/local-driver tests
also exist; rerun after final integration.
