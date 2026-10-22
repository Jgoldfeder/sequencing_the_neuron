# Controlled C5 committee experiments, 2026-10-21

## Correction to the causal interpretation

My initial claim that these tests identified downstream fitting as the explanation for the original C5 stall was unsupported and is withdrawn. Successful MLP and LeNet L1/L2 recovery did not require an exact suffix. The tests below used fixed oracle-targeted data instead of the adaptive committee sampler and lacked matched successful-layer controls. The measurements remain recorded, but they do not explain the layer-to-layer gap.

## Question and limits

Does making the missing teacher C5 units fire suffice for the committee to recover them? Are the remaining rows themselves unlearnable, or does learning the downstream layers at the same time obstruct recovery?

These are **oracle-assisted diagnostic experiments**, not an honest end-to-end recovery. Teacher C5 rows guide query construction and score the learned rows. In the fixed-suffix controls, F6/head weights are supplied exactly. No additional C5 rows were put into an honest recovery record, and no kink refinement or global certificate was run. Approximate coefficient agreement below must not be described as 120/120 exact extraction.

The experiment starts with the attributable `_record_lenet_c1c2_c5_105.pt`: conv1 6/6, conv2 16/16, C5 **103/120**. The persistent fp64 prefix is used directly; the known C5 rows remain pinned. It does not use the collided snapshot families.

## Implementation and validation

`committee_boundary_test.py` trains eight independently initialized 400→120→84→10 tails on teacher output logits. It caches the exact prefix at real image queries and vectorizes member training. Each member is optimized independently with Adam, batch 512, learning rate 0.001, L1 loss unless marked MSE. Learning rate drops tenfold at 60% and 85% of the configured epoch budget. All 17 missing C5 rows train together in these committees. Pinned rows are gradient-masked. The free arms train F6/head; the fixed-suffix arms use the exact canonical teacher F6/head and freeze them.

This isolates training from the production bookkeeping and sampler. It is **not a replay of adaptive disagreement sampling**: the sampler is deliberately replaced with a fixed diagnostic dataset. The row-norm arm preserves the pinned input gauge, canonicalizes F6 once, and projects unpinned hidden rows after each update. It is an isolated counterpart of the production heuristic, not a bit-for-bit replay of all historical rescalings.

`validate_tail_cache.py` independently compares this representation with the actual ConvNet. Cached features match exactly, saved teacher labels match exactly, full ConvNet versus cached-tail maximum output difference is 3.81e-6, and the exact-parameter oracle control differs from labels by at most 1.91e-6 (`tail_validation.json`).

“Accurate” means the maximum coordinate error of a signed unit-normalized augmented `[w|b]` row is below **0.02**, after one-to-one matching to the missing teacher rows. It is a proximity/refiner-basin diagnostic, not an exactness certificate. Ground-truth matching is used for measurement and never as a training loss. “Four accurate members” is the script's stopping rule; the successful joint-C5 control actually has at least five accurate members per row (counts below). Inf-norm agreement clusters are also measured separately: agreement alone could be wrong.

## First test and why it was strengthened

Run `a` used 256 targeted images per missing row, half at preactivation −0.5 and half at +0.5, plus the same number of Gaussian images: 8,704 images total. Three arms used identical initial populations: Gaussian+row norm, targeted+row norm, and targeted+free norms. After 600 epochs, none had any missing row within the 0.02 coefficient threshold.

Run `b` reused those queries for 4,000 epochs. Supplying the exact suffix recovered one missing row; whitening with a trainable suffix and MSE with a trainable suffix recovered none. Run `c` combined whitening, MSE, and the exact suffix and also recovered one.

These early negatives were insufficient to conclude that the rows are intrinsically hard: there were only **128 strongly positive examples per targeted row**, fewer than its 401 augmented coefficients. The constant positive target also confines those examples to one affine preactivation plane.

## Stronger dataset

Run `e` instead generates **1,024 images per missing row**, with 512 positive and 512 negative target preactivations of independently varied magnitude between 0.1 and 0.9. It adds an equal number of Gaussian images, for **34,816 real image queries**. Images are optimized through the exact recovered prefix and constrained to `[-3,3]`. The targets are used to generate queries, not as hidden-label training targets.

Every missing channel's positive-example augmented feature matrix has numerical rank **401**, using relative tolerance 1e-5. Thus each has enough varied, full-rank positive examples; the test is stronger than “one firing point” or an aggregate silent-channel count.

Runs `e` (both suffix settings), `g` (row norm), and `h` (Gaussian+exact suffix) use equal-sized datasets. The two arms within `e` have identical initial populations. Runs that reuse a saved dataset begin from the same fresh initialization seed as one another, but their population differs from `e`, which consumed random draws generating its images. This is a diagnostic on one trained teacher, not a multi-seed statistical comparison.

## Positive result: all 17 missing rows learn together when F6/head are exact

In run `e`, the exact-suffix eight-member committee meets the stopping criterion at **epoch 2,525**:

- **17/17** missing rows have an accurate member.
- **17/17** have at least four accurate members; in fact every row has at least **five**.
- Accurate-member counts across the 17 teacher-matched rows are `[8,8,8,6,7,8,8,8,8,7,7,8,8,5,8,8,8]`.
- Worst across rows of the best member's normalized coefficient error: **0.01474**.
- Worst across rows of the best member's cosine: **0.999328**.
- Best member training L1 output loss: **2.28e-4**.
- Pinned coefficients remain fixed in fp32; their difference from the fp64 record is at most **1.48e-8**.

Applying the **actual production `_cnn_consensus` routine**, with `eps=0.02` and quorum **5/8**, to the saved successful population gives **120/120 C5 in consensus**, including **17/17** previously missing rows (`production_consensus_check.json`). This validates the consensus result independently of the diagnostic metric.

This is a real positive committee-learning result for the missing rows under a strong oracle suffix control. It rules out treating these individual C5 rows as categorically unlearnable. It does not supply an honest route to the exact F6/head weights used by that control.

## Single-row isolation control

`single_row_control.py`, run `f`, independently learns each missing row eight times while supplying the other 119 C5 rows and the complete suffix exactly. It uses the same stronger dataset, output-logit MSE, whitened input coordinates, and no hidden-label loss. It meets its stricter stopping rule at epoch **2,750**: every missing row has at least four fits with coefficient error <0.02 **and** cosine >0.99. Best-row coefficient errors range from about 8e-7 to 0.00823, and the worst best-row cosine is 0.999788.

This corroborates that the rows can be learned individually. Its oracle assumptions are even stronger than the exact-suffix committee's, so it must not be presented as a practical extraction algorithm.

## Artifacts and reproduction

- `committee_boundary_test.py`: query construction and eight-member joint-C5 experiment.
- `single_row_control.py`: independent single-row isolation control.
- `boundary_test_20261021_{a,b,c,e,g,h}/events.jsonl`: per-epoch measurements, without shared run tags.
- `boundary_test_20261021_e/dataset.pt`: stronger dataset; includes image queries, cached features and teacher output labels.
- Each run has its own config and checkpoint paths. `boundary_test_20261021_d` was deliberately stopped during generation and replaced by `e` to vary the target margins; it is not a result.
- `single_row_20261021_f/events.jsonl`: independent-row measurements.
- `committee_test_summary.json`: consolidated arm summaries; regenerate with `python audit_lenet/summarize_committee_tests.py`.

From the code directory, for example:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python audit_lenet/committee_boundary_test.py --out audit_lenet/NEW_UNIQUE_PATH --device cuda:1 --epochs 4000 --per-channel 1024 --arms boundary_fixedtail,boundary_free
```

To reuse the saved stronger dataset, add `--dataset-from audit_lenet/boundary_test_20261021_e`. Choose a new output directory; the script refuses to overwrite an existing run directory.

## Final comparison on the stronger dataset

| Queries | F6/head | Row-norm projection | Epochs | Missing rows accurate in ≥5/8 members | Best training L1 |
|---|---|---|---:|---:|---:|
| Gaussian | teacher exact, fixed | No | 4000 | 2/17 | 2.6e-05 |
| Targeted, full-rank positive examples | teacher exact, fixed | No | 2525 | 17/17 | 0.000228 |
| Targeted, full-rank positive examples | random, trained | No | 4000 | 0/17 | 0.033 |
| Targeted, full-rank positive examples | random, trained | Yes | 4000 | 0/17 | 0.0303 |

The targeted-data, exact-suffix and free-suffix arms within run `e` share their initial C5 population and query dataset. Their difference is whether the suffix is supplied exactly and frozen or initialized randomly and trained. Supplying that suffix changes the outcome from zero to all 17 accurate rows under this test. The separate balancing run does not close that gap.

The Gaussian/exact-suffix control fits its outputs more closely than the successful targeted-data control, while recovering far fewer missing rows. Low output loss on a biased query distribution is therefore inadequate evidence of parameter recovery.

**Corrected interpretation:** these results show that this fixed-dataset training setup benefits from an exact oracle suffix. They do not establish that the committee method needs known downstream layers, or that downstream fitting explains the original C5 stall. The successful MLP experiments and LeNet L1/L2 recovery already demonstrate recovery with unknown downstream weights. This experiment did not reproduce those successes as positive controls, replaced adaptive disagreement sampling with a fixed oracle-targeted dataset, and changed the training implementation. Its negative arms therefore do not diagnose the production algorithm. Full rank of positive feature samples also does not certify that these finite output-only samples identify every internal row. The historical C5 failure remains unexplained beyond the separately reproduced residual-sampler numerical bug.

The appropriate next audit is to reproduce a known successful MLP run and LeNet L1/L2 with the unchanged adaptive committee pipeline and unknown suffix, then trace C5 against those positive controls. A direct 400→120→84→10 MLP representation of the same LeNet tail can help separate prefix/query geometry from CNN-specific training, alignment, and pinning behavior. No requirement for a known suffix follows from the experiments above. Existing recovery code and records remain untouched.
