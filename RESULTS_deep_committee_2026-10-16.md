# Committee sampling through a solved prefix: deep layers (2026-10-16, autonomous session)

Question (Judah): given layers 1..7 solved exactly, can the committee (population disagreement) solve layer 8 through the
prefix, in a practical way, implying the whole net can be peeled? Committee method only (no guess-free kink recovery).

## Setup
- 9-layer teacher 3072x200^8x100, leaky ReLU, CIFAR-trained 25 epochs, seed 0
  (teachers/teacher_3072x200x200x200x200x200x200x200x200x100_e25_s0.pt).
- "Tail framework": students = tail net (layers K+1..9) whose input is h_K; black box = tail teacher
  (teachers/teacher_<taildims>_e25_s8.pt = the full teacher's layers K+1..9); queries are produced by a map from raw x
  through the EXACT first K teacher layers (`--query-via-mlp-prefix "FULLDIMS:K"`). This is identical to the pipeline
  with K pinned layers, minus the committee's own pinning bookkeeping. 8 members, 20k queries/iteration unless noted,
  pins = kink refiner + global kink certificate (`--verify-cands`); pins checked permutation-free with SIGNED cosine
  (scratchpad/check_mlp_pins.py-style).
- 6-layer teacher 3072x200^5x100 (seed 0) for the end-to-end chain.

## Findings

1. Exact inversion of the prefix (`--mine-in-h`, mine in h-space, map back x = prefix^-1(h)) is exact and cheap for K=1
   (|x| ~ 700, round trip 1e-6) and made via-prefix == direct on the 6-layer net (L2 200/200 + L3 192 at it 15 vs
   x-mining 178/0). It is NOT usable deeper: every negative coordinate costs a factor 100 per layer; square layers make
   the pre-image unique, so for K=7 the minimum-norm input is |x| ~ 1e20 (fp64 round trip fails, fp32 teacher is noise).
   K=2 is already |x| ~ 1e6 with round-trip error 4.5e-3. `mine_in_h_max_k = 1` now limits it; deeper frontiers fall back
   to x-space mining with `--h-push`.
2. Reachability is not the problem: all 101 layer-8 kinks that random inputs almost never cross are reachable at
   |x| <= 30 (the natural norm) with margins 30-60. The problem is the DISTRIBUTION at the frontier: reachable h7 lives
   in a cone close to the nonnegative orthant (negatives compressed x0.01), 100/200 L8 units active <5% under random x,
   14-47 h7 coordinates positive <5%, coordinate std spread 65x. The direct attack's h7 ~ U(-1,1) cannot be produced.
3. What makes the committee work through the prefix (all legal, attacker-only information):
   - loosen the consensus gate: cluster_eps 0.02 -> 0.1 (rows at depth agree to ~5%, enough for the refiner's basin,
     and every pin is certified by the kink refiner + verify-cands);
   - push each raw query along a random direction of the frontier's input space through the known prefix before
     disagreement mining (`--h-push 30 --h-push-mode dir`, |x| capped at 1.1x the natural norm);
   - per-row hot Adam steps for rerolled rows (`--hot-reroll 10`), duplicate reroll (`--reroll-dups`), kink_solve for
     L1 (`--loc-refine-l1`), reinit lr for freeze-reinit (`--reinit-lr`).
4. ORIENTATION TRAP (real, new): with loose consensus and no pushes, 3/158 pinned L8 rows were SIGN-FLIPPED teacher rows
   (cos -1). They are genuine kink hyperplanes (verify_row = 1.00), so no local kink certificate can catch them:
   leaky(-z) = leaky(z) - 1.0101 z and the linear term is absorbed by always-on units when the data live in the
   nonnegative orthant. Orientation is identified only by data where the unit flips and the "linear bus" is broken.
   The push recipe had 0 flipped pins. TODO: global consistency check after a layer completes (fresh pushed queries,
   flip-and-fit), or a far-field slope test.
5. Pin precision at depth: the default "track" refiner degrades (L2 1e-9, L3 2e-6, L4 1e-4 eps); the design sampler
   (`--design-refine`) holds 2-3e-9 at every layer (L1..L4 on the 6-layer chain) at ~10M queries per layer (vs 1.5M).

## Layer 8 via the 7-layer prefix (tail framework, 20k q/it; consensus count / pinned exact)

| arm | sampling | consensus eps | it 5 | it 10 | it 15 | it 20 | it 25-30 | wrong pins |
|---|---|---|---|---|---|---|---|---|
| direct (h7 ~ U(-1,1), unreachable; idealised) | h-space | 0.02 | 199 / 199 | | | | | 0 |
| C: via prefix, x-space mining (pipeline as it was) | x | 0.02 | 0 | 0 | 4 / 1 | 17 | | 0 |
| D: reachable init, ascent in h (NOT legal) | h | 0.02 | | 35 / 28 | 128 / 122 | | | 0 |
| E: standardised h7 | x | 0.02 | 0 | 0 | 8 / 5 | | | |
| E2: standardised h7 | x | 0.1 | 13 | 114 / 89 | 132 / 110 | | | 0 (verify rejected 7) |
| G: DIRECT but h7 ~ |U| (nonneg orthant) | h | 0.02 | 1 | 108 / 106 | 121 / 120 | | | 1 flipped |
| F: pushes toward student units | x | 0.02 | 0 | 71 / 50 | 108 / 89 | | | 0 |
| H: pushes along random directions | x | 0.02 | 0 | 79 / 50 | 104 / 95 | | | 0 |
| O: pushes toward TEACHER units (oracle diag.) | x | 0.02 | 0 | | 98 / 78 | | | 0 |
| C2: x-mining | x | 0.1 | 0 | 151 | 163 / 158 | 165 / 161 | | 3 flipped |
| **J: random-direction pushes** | x | **0.1** | 25 / 5 | **171 / 169** | | 177 / 176 | 178 / 178 (it 30) | **0** |
| J48k: J with 48k q/it | x | 0.1 | | 178 / 176 | | 180 / 177 | 179 / 177 (it 30) | 1 flipped |

Other depths with the J recipe (9-layer net): K=6 (attack L7, tail 200-200-100): L7 195/200 pinned exact at it 15-30,
L8 174-177 in consensus. K=4 (attack L5, tail with 4 hidden layers): 17/200 at it 30 (the deep tail itself converges
slowly at 20k q/it; not yet studied).

## 6-layer chain end-to-end (committee -> peel one layer at a time)
- chain3 (`--mine-in-h --h-push 30 --cluster-eps 0.1 --hot-reroll 10 --reroll-dups --verify-cands --loc-refine-l1`),
  40 iterations: L1 200, L2 200, L3 200, L4 199 pinned exact (eps L3 2e-6, L4 1e-4 with the track refiner); L5 reached
  179 in consensus but was NEVER refined because the frontier only advances at 100% and the last L4 unit was not found
  (hot-reroll did not re-find it in 25 iterations).
- chain4 (same + `--design-refine`), 40 iterations: L1 200 (3e-10), L2 200 (2e-9), L3 200 (2.5e-9), L4 199 (2.9e-9);
  L5 190 in consensus, same 100%-gate stall at L4 199/200.
- mlp9_full_recipe2 (9-layer net from scratch, full recipe, 40 iterations, 1.9M queries): the committee itself never
  converged (loss 1.0-1.2, no L1 pins). This is the separate from-scratch deep-committee problem (see balance-flag
  notes), not the via-prefix question; the via-prefix evidence is the tail framework above.

## What is still open
1. The 100% frontier gate: both chains sat at L4 199/200 with L5 at 179-190 in consensus and never refined L5. Advance
   at >= 99% with stragglers trainable (needs care: the next layer's refine uses the record prefix, so the unsolved
   unit's column must be masked or the straggler solved first).
2. Orientation certificate (finding 4). 1 flipped pin in J48k, 3 in C2, 1 in G; 0 in J. Needs a global check.
3. Last 10% of a deep layer: J plateaus at 177-180/200 from it 20 to 30 (direct: 199 at it 5). Candidates: targeted
   pushes toward the students' rarely-flipping units once most of the layer is pinned, more members, longer runs.
4. From-scratch committee convergence on the 9-layer net (committee problem, not peel).

## Commands (copy-paste; run from code/)
# layer-8 via the exact 7-layer prefix, best legal recipe (J):
python run.py --variant mergedbest512 --arch 200,200,100 --teacher-seed 8 --teacher-seed-full 0 --query-via-mlp-prefix "3072,200,200,200,200,200,200,200,200,100:7" --h-push 30 --h-push-mode dir --cluster-eps 0.1 --device cuda --outer 30 --window 30 --q 20000 --disk-seg-gb 0.5 --fast-peel-partial --peelrefresh --loc-refine --loc-refine-l1 --verify-cands --pop-save-every 5 --tag deep9_tail_J
# direct (idealised) baseline:            same without --teacher-seed-full/--query-via-mlp-prefix/--h-push/--cluster-eps
# pipeline as it was (C):                 same without --h-push and --cluster-eps  (4/200 at it 15)
# 6-layer chain (resume from the h-mined committee; chain4 = + --design-refine instead of --loc-refine):
python run.py --variant mergedbest512 --arch 3072,200,200,200,200,200,100 --device cuda --outer 40 --window 30 --q 48000 --qg-chunk-gb 1 --disk-seg-gb 1 --resume-pop recon/_pop__mergedbest512_mlp6_chain3_eps01__3072x200x200x200x200x200x100__s0.pt --fast-peel-partial --peelrefresh --loc-refine --loc-refine-l1 --verify-cands --reroll-dups --hot-reroll 10 --mine-in-h --h-push 30 --h-push-mode dir --cluster-eps 0.1 --pop-save-every 5 --balance 1 --balance-refresh 5 --lr 1e-4 --tag mlp6_chain3_eps01
# tail teachers: teachers/teacher_200x200x100_e25_s8.pt (K=7), teacher_200x200x200x100_e25_s8.pt (K=6), teacher_200x200x200x200x200x100_e25_s8.pt (K=4) = layers K+1..9 of the 9-layer seed-0 teacher
# logs: pool_logs/deep9_tail_*.log, pool_logs/mlp6_chain*.log; snapshots: recon/_pop__mergedbest512_<tag>__*.pt

## Commands (copy-paste)
(see the end of this file; filled in at the end of the session)

## Addendum (09:45): end-to-end consensus and the orientation certificate
- `--orient-check` (orient.py): global orientation test; offline 4/4 flipped pins detected with 0 false positives
  (alpha -1.02..-1.18 vs correct rows in [-0.10, 0.25]); live it sign-corrected the 3 flipped candidates of the loose
  no-push arm (C2_orient: 161 pinned, all exact) and found none in the push arm (J_orient: 176 exact at it 15).
- `--masked-consensus`: deep rows compared only on input columns already in consensus. Plain 6-layer committee (no peel,
  no refiner, it 30 snapshot, eps 0.1 deep / 0.02 L1): consensus L1 200, L2 195, L3 195, L4 196, L5 181; validated
  against the teacher 200 / 195 / 191 / 196 / 178 correct (median 1-cos 1e-4..1e-3). The full-row metric had reported
  L3 0 and L4 0-2. End-to-end consensus is therefore already most of the way there for ReLU, as for tanh; the refiner is
  polish, and the deep-prefix sampling work above is about the LAST units of each layer.

## Addendum (10:30): which part of the consensus change matters
Same no-peel committee, quorum 5/8, counts per layer L1..L5:
- weight-chained alignment, full rows, eps 0.02 (old): 200, 194, 0, 0, 119
- weight-chained alignment, full rows, eps 0.1:        200, 194, 196, 196, 182
- weight-chained alignment, masked columns, eps 0.1:   200, 195, 195, 196, 181
- activation-matched alignment, full rows, eps 0.02:   200, 193, 0, 0, 18
- activation-matched alignment, full rows, eps 0.1:    200, 195, 170, 195, 174
=> the eps is the lever; masking changes little; activation-based alignment is slightly worse, not better. Deep rows are
~1e-2 accurate as whole rows (full-row cos>0.995 fails at L3-L5) and ~1e-3 on the columns of matched input units.
LeNet (saved best members, channel-masked metric, no rerun): conv1 6/6, conv2 16/16, C5 100-104/120 in the better runs
(old consensus read 87), F6 40-71/84 on the matched C5 columns.

## Addendum (18:50): LeNet residual committee
Freeze everything known (exact conv1/conv2/C5 pins, F6 and head at the resumed member's values) and train only the
unsolved C5 rows + their F6 columns (`--resume <snap> [--resume-record] --residual-warm`): C5 72 -> 100 pinned in 7
iterations (faster growth than the full committee's 61 -> 100 in 15), then the SAME ceiling: 101-102 pinned, consensus
105-107 of which every candidate beyond ~102 is junk (copies / F6 surfaces, rejected by dedupe, verify-cands, refiner).
At iteration 10 the best member's 19 free C5 rows are 0.13-0.56 cosine from any teacher row: the committee never
represents the one-sided channels. With hot x10 rerolls the free rows diverge. Conclusion: for LeNet's C5 the committee
method's limit is learning the rarely-firing channels, not consensus/alignment/pinning; the kink-isolation scan (not
allowed as the method) finds them in minutes.
Also added today for CNN runs: `--layer-consensus` prints the isolated two-sided ("iso") consensus (cnn_iso.py),
`--resume-record`, `--residual-warm`, `--reroll-dups --reroll-stale K --hot-reroll F`.

## 2026-10-18: why the 9-layer (3072-200x8-100) committee was stuck, and the fix
1. Gauge bug (fixed): with --fast-peel-partial on, the peel-mode balance re-canonicalised members on the MINED batch
   (|x| ~1.5x larger) at it 5 and held L1 rows at norm 0.27 instead of 2.0 -> Adam's relative step on L1 7x too large.
   _mb_iter_start/_bal_iter_start/rownorm refresh+init now canonicalise on random N(0, qg_init_std) inputs.
2. Not the cause by itself: with the fix, q 48k x window 30 committees still sat at loss 0.6-1.1, L1 median cos 0.87.
3. Root cause: DISTINCT-QUERY COUNT. Single student (Adam, batch 2048, 50k steps): fresh samples -> L1 199/200, loss
   0.31; fixed 3M buffer -> 197/200; fixed 1.4M buffer (= q48k x 30) -> 1/200, loss 0.60. The 6-layer teacher transitions
   at ~25M samples, the 9-layer needs ~100M visits over >= ~3M distinct inputs. Oracles: only "L2-9 given, train L1" is
   easy (0.031); any >=4 trainable hidden layers is slow; shallow students (1-5 hidden) do not recover L1 at all.
4. Committee at q 200k x window 20 (4M distinct): loss 0.70 @it15 -> 0.076 @it20 -> 0.072 @it25; validated vs teacher at
   it 25: per member L1 200, L2 199-200, L3 200, L4 199-200, L5 198-199, L6 198, L7 199, L8 192-198 within 12 deg. The
   whole net is in the refiner basin. (Caution: eps-0.1 consensus COUNTS on an unconverged deep committee are junk,
   e.g. 1472/1600 at it 15 with members at loss 0.55; validate vs teacher.)
5. Running: mlp9_peel_q200k (full pin recipe at q 200k), mlp9_nopeel_q1M.
Commands: python run.py --variant mergedbest512 --arch 3072,200,200,200,200,200,200,200,200,100 --device cuda --outer 30 --window 20 --q 200000 --qg-chunk-gb 2 --disk-seg-gb 2 --balance 1 --balance-refresh 5 --cluster-eps 0.1 --cluster-eps-l1 0.02 --masked-consensus --pop-save-every 5 --tag mlp9_nopeel_q200k
