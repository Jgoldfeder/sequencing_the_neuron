# Record: tanh recipe demonstrations, all depths (campaign of 2026-10-16/17, reruns 2026-10-18)

One command per depth: `solve_tanh.py` = committee (run.py mergedbest512, --tanh --layer-consensus --align-inverse,
30 Adam steps/block disagreement sampler, 8 members, batch 512) -> layer-1 isolation + tail-decay scales (deep
nets only) -> K-FAC jet refine with layer 1 frozen -> K-FAC polish (frozen) -> K-FAC final polish (all free).
Solver budgets for every depth: `--refine-cg 2000 --refine-iters 8 --polish-cg 2000 --polish-iters 6
--final-cg 2000 --final-iters 4`, 512 points x 64 directions per jet stage (131,584 queries per stage).
Teacher seed 0, student seed 0, CIFAR-shaped input 3072, hidden 200, output 100. One RTX 3090 per depth
(three depths ran concurrently, which slows every iteration; the hours below are wall clock as run).

## Campaign 1 (2026-10-16 17:46 -> 2026-10-17 15:26). Logs: results/logs/solve_tanh__demo_<D>.log, committee__demo_<D>.log

| depth | committee flags | committee: certified, max/mean eps, queries, h | layer-1 stage: assembled max/mean, queries | final max / mean eps | total queries | total h | final net |
|---|---|---|---|---|---|---|---|
| 5L (3 hidden) | q 48k, window 30 | 600/600, 2.40e-4 / 1.98e-5, 1.49M, 0.70 | shallow path (layer 1 from committee) | **4.25e-6 / 1.11e-7** | 1.62M | 1.47 | recon/_final__demo_5L.pt |
| 6L | q 96k, window 30 | 799/800, 5.24e-4 / 3.13e-5, 2.98M, 0.68 | shallow path | **3.81e-5 / 1.32e-6** | 3.11M | 1.50 | recon/_final__demo_6L.pt |
| 8L | q 300k, window 30 | 947/1200, 1.37 / 6.91e-4, 9.30M, 2.36 | 4.73e-3 / 5.38e-5, 127.66M | **1.94e-6 / 8.85e-9** | 137.36M | 7.69 | recon/_final__demo_8L.pt |
| 9L | q 800k, window 20, epochs 5, --disk | 684/1400, 8.75 / 5.94e-2, 24.8M, 3.43 | 1.20 / 4.33e-3, 136.63M | 7.09e-1 / 9.72e-3 (FAILED) | 161.82M | 7.97 | recon/_final__demo_9L.pt |
| 10L | q 1.2M, window 20, epochs 5, --disk | crashed in iteration 21 (NaN in the alignment); at it 20: 1199/1600 certified, trunk <= 2e-2 | - | - | 25.2M | 3.1 | recon/_pop__..._demo_10L (it 20) |
| 11L | q 1.8M, window 20, epochs 5, --disk | 441/1800, 10.5 / 2.24e-1, 55.8M, 12.81 | 5.88e-1 / 2.81e-3, 138.31M | 3.80e-1 / 4.13e-3 (FAILED) | 194.50M | 20.20 | recon/_final__demo_11L.pt |

Stage detail for 8L (the deep success): committee 1.37/6.9e-4 -> assembled (exact L1 directions + scales,
committee trunk) 4.73e-3/5.4e-5 -> refine L1 frozen 2.42e-6/3.0e-8 -> polish 2.57e-6/3.0e-8 -> final (all free)
1.94e-6/8.9e-9. Layer-1 isolation + scales cost 127.66M queries (isolation ~0.6M/neuron, scales ~0.3M/neuron).

Extra 8-step all-free polish on the shallow results (+131,584 queries each, results/logs/final2__demo_{5L,6L}.log):
5L 4.25e-6 -> **4.18e-11** (mean 2.5e-12), 6L 3.81e-5 -> **3.89e-8** (mean 1.6e-9). Nets recon/_final2__demo_{5L,6L}.pt.

### Why 9L / 10L / 11L failed: a data-feed bug, not the recipe
All three used `--disk` (two-tier RAM+NVMe replay ring) with its default GPU-resident segment feeding
(diskring._segment_batches, added 2026-10-13 AFTER the successful 11-layer run RECORD_tanh_11layer_disk_q1.8M_ep5_s0.md,
which used the CPU-gather path). A synthetic wrap-around test (scratch ring_wrap_test*.py / ring_stress.py)
shows this path occasionally delivers minibatches whose target rows are garbage (uninitialised GPU memory).
Consequences seen in the campaign: 10L: a member went non-finite in iteration 21 -> NaN similarity matrix ->
crash; 9L: three trunk units (W6 x1, W7 x2) stuck at 0.4-0.6 error from iteration 16 on while the loss kept
falling; 11L: 441/1800 certified at it 30 vs 819 in the uncorrupted earlier run with identical flags.
Second, independent 9L defect: the tail-decay scale of layer-1 unit 85 (|w| 3.98, the largest-norm units are
the exposed ones) came out as 0.847 (79% off) because a neighbouring unit's slower exponential tail dominated
the fit window on both sides of one probe line and passed the slope-constancy test.

### Fixes (2026-10-18)
- diskring.py: a staging buffer may be refilled by the reader only after the DMA out of it has landed
  (release gating); bounce-buffer creation under a lock; failed cudaHostRegister clears the sticky CUDA
  error; staging buffers unregistered on close; `DISKRING_VERIFY=1` checksums every segment copy.
  (Status: see the campaign-2 section; the stress test is the acceptance criterion.)
- method.py: NaN guard after training (a non-finite member is re-seeded from the best finite member with 1%
  noise, its Adam state reset, logged as `[nan-guard]`); non-finite sampler queries replaced by random rows;
  member ranking treats non-finite losses as +inf. align.py: NaN/inf never reach linear_sum_assignment.
- peel/scale_l1_deep.py: direction-purity gate along the tail (|G.c|/|G| > 1 - 1e-4) and the agreeing pair
  must come from two different probe lines. Validated on the failing unit (0.79 -> 7.0e-8) and 5 controls.

## Campaign 2 (reruns, started 2026-10-18 01:16; log results/logs/demo/schedule2.log, tags demo2_9L/10L/11L)
9L: q 1.2M, window 20, epochs 5, --disk, 30 outer (GPU 2). 10L: q 1.2M, same (GPU 1). 11L: q 1.8M, same,
**40 outer** (GPU 0). Results are appended here when they finish.

### 9L rerun (demo2_9L): SUCCESS. q 1.2M, window 20, epochs 5, --disk (CPU-gather feed), 30 outer, GPU 0, 2026-10-18 09:44 -> 23:23 (13.64 h)

| stage | max eps | mean eps | queries | cumulative | h |
|---|---|---|---|---|---|
| 1 committee (1178/1400 certified; W2-W7 + head complete, 101 L1 rows) | 4.61 | 1.44e-2 | 37.20M | 37.20M | 7.71 |
| 2a layer-1 isolation (200/200 < 1e-8, no gap-fill) + scales (median 1.1e-7, max 5.8e-6), assembled | 1.52e-2 | 5.98e-4 | 145.33M | 182.53M | 10.70 |
| 2b K-FAC refine, L1 frozen (cg 2000 x 8) | 2.13e-6 | 8.53e-8 | 0.13M | 182.66M | 10.70 |
| 3 K-FAC polish, L1 frozen (cg 2000 x 6) | 1.90e-6 | 8.73e-8 | 0.13M | 182.79M | 12.34 |
| 4 K-FAC final polish, all free (cg 2000 x 4) | **1.68e-6** | **3.42e-8** | 0.13M | 182.92M | 13.64 |

Final per-matrix max: W1 1.3e-6, b1 1.7e-6, W2 5.1e-7, then 1e-7..5e-9 through the trunk (head 5e-9). Net: recon/_final__demo2_9L.pt.
Note: the solver's starting trunk was the quorum-CONSENSUS net (built from the it-30 population with
_partial_consensus in the most-agreeing frame, aligned to the best member's unit order; scratch swap_guess_9L_v4.py),
not the best member: the best member (6) had three units in W6/W7/W8 at 0.45/0.72/0.38 while the consensus had them at
5e-3 (the same symptom that sank the first 9L). The pipeline's assemble/refine then went 1.5e-2 -> 2.1e-6 in one stage.
Lesson for solve_tanh: use the consensus net as the trunk guess by default when it is available.
The final-stage max is the layer-1 scale floor (1.7e-6); every step was still accepted with rho ~ 1, so more all-free
steps / larger CG would keep lowering it (5L/6L went 4e-6 -> 4e-11 and 4e-5 -> 4e-8 with 8 extra steps).

### 10L rerun (demo2_10L): SUCCESS. q 1.2M, window 20, epochs 5, --disk (CPU-gather feed), 30 outer, GPU 1, 2026-10-18 09:39 -> 23:50 (14.16 h)

| stage | max eps | mean eps | queries | cumulative | h |
|---|---|---|---|---|---|
| 1 committee (1211/1600 certified; W2-W8 + head complete, L1 directions 200, magnitudes 52) | 7.75 | 1.23e-1 | 37.20M | 37.20M | 7.90 |
| 2a layer-1 isolation (200/200 < 1e-8, no gap-fill) + scales (median 1.2e-7, max 5.0e-6), assembled | 2.01e-2 | 1.01e-3 | 143.88M | 181.08M | 11.08 |
| 2b K-FAC refine, L1 frozen (cg 2000 x 8) | 1.06e-5 | 3.89e-7 | 0.13M | 181.21M | 11.08 |
| 3 K-FAC polish, L1 frozen (cg 2000 x 6) | 3.11e-6 | 1.15e-7 | 0.13M | 181.34M | 12.83 |
| 4 K-FAC final polish, all free (cg 2000 x 4) | **2.92e-6** | **6.18e-8** | 0.13M | 181.48M | 14.16 |

Final per-matrix max: W1 1.5e-6, b1 2.9e-6, trunk 7e-7..7e-8, head 1e-8. Net: recon/_final__demo2_10L.pt. Trunk guess = best member
(its trunk was clean, 2e-3..2e-2; no consensus swap needed). The committee stage took 7.9 h because three jobs shared the box
(iterations 20 min instead of 8); solo it is ~3.5 h.

## Demo table (one command each; max / mean parameter error vs the black box, total queries, wall)
| depth | committee q / window | final max eps | final mean eps | total queries | wall h | net |
|---|---|---|---|---|---|---|
| 5L | 48k / 30 | 4.25e-6 (4.2e-11 with 8 extra all-free steps) | 1.11e-7 (2.5e-12) | 1.62M (+0.13M) | 1.5 | recon/_final__demo_5L.pt (_final2__) |
| 6L | 96k / 30 | 3.81e-5 (3.9e-8 extra) | 1.32e-6 (1.6e-9) | 3.11M (+0.13M) | 1.5 | recon/_final__demo_6L.pt (_final2__) |
| 8L | 300k / 30 | 1.94e-6 | 8.85e-9 | 137.36M | 7.7 | recon/_final__demo_8L.pt |
| 9L | 1.2M / 20, ep 5, --disk | 1.68e-6 | 3.42e-8 | 182.92M | 13.6 (shared box) | recon/_final__demo2_9L.pt |
| 10L | 1.2M / 20, ep 5, --disk | 2.92e-6 | 6.18e-8 | 181.48M | 14.2 (shared box) | recon/_final__demo2_10L.pt |
| 11L | 1.8M / 20, ep 5, --disk, 40 outer | (running, demo2_11L) | | | | |

### Automated after the 9L rerun (2026-10-19 00:10): consensus starting net
`solve_tanh.py --trunk consensus` (default; `--trunk best` = old behaviour) / `peel/diag/consensus_guess.py`: the solver's
starting net is the 5-of-8 quorum consensus (method._partial_consensus, most-agreeing frame) aligned into the best
member's unit order, used WHOLE (own layer 1 + trunk). Gates: every layer-1 row must match the best member's (|cos| >= 0.99)
and the consensus must not be a > 3x worse black-box fit on 4096 fresh queries; otherwise the best member is kept.
Tested offline on the two populations: 9L -> consensus (reproduces the manual swap: W6/W7/W8 5e-3 instead of 0.45/0.72/0.38);
10L -> order check fails on 2 units -> best member (which was clean). The running 11L picks this up automatically
(the pipeline is a subprocess launched after the committee).

### 12L at q 1.8M (demo2_12L): committee insufficient. q 1.8M, window 20, epochs 5, 30 outer, GPU 1 (shared box with 11L), 2026-10-19 00:07 -> 16:09 (16.0 h)

| stage | max eps | mean eps | queries | cumulative | h |
|---|---|---|---|---|---|
| 1 committee (0/2000 certified; loss 0.105; best member trunk units at 0.3-0.7 in W2, W4-W6) | 13.4 | 2.26e-1 | 55.80M | 55.80M | 10.05 |
| 2a layer-1 isolation (195 direct + 5 orphans, all < 1e-8) + scales (200/200 confirmed), assembled | 8.59e-1 | 3.47e-3 | 159.06M | 214.86M | 12.77 |
| 2b K-FAC refine, L1 frozen | 5.62e-1 | 4.53e-3 | 0.13M | 215.00M | 12.77 |
| 3 K-FAC polish, L1 frozen | 5.63e-1 | 3.89e-3 | 0.13M | 215.13M | 14.57 |
| 4 K-FAC final polish, all free | **5.63e-1** | **3.83e-3** | 0.13M | 215.26M | 16.03 |

Final per-matrix max: W4 0.56, W5 0.33, b5 0.19, W6 0.23 (the committee's unsolved units, outside the solver's basin);
everything else 2e-3..4e-2. NOTE: the all-free final stage DEGRADED the exact layer 1 (W1 2.1e-6 -> 2.3e-3) by trading
error against the bad trunk units — with an unsolved trunk the frozen-L1 net (recon/_l1fixed_refined__demo2_12L.pt) is
the better artefact. Consensus starting net declined (no quorum). Net: recon/_final__demo2_12L.pt.

### 11L at q 1.8M, second clean run (demo2_11L): committee insufficient. q 1.8M, window 20, epochs 5, 30 outer, GPU 0 (shared box with 12L), 2026-10-19 00:04 -> 16:27 (16.4 h)

| stage | max eps | mean eps | queries | cumulative | h |
|---|---|---|---|---|---|
| 1 committee (461/1800 certified; loss 0.103; best member units at 0.45 W2, 0.43 W3, 4.0 W9, 0.49 head) | 10.4 | 2.27e-1 | 55.80M | 55.80M | 10.06 |
| 2a layer-1 isolation (198 direct + 2 orphans, all < 1e-8) + scales (200/200), assembled | 4.10 | 5.54e-3 | 150.92M | 206.72M | 13.22 |
| 2b K-FAC refine, L1 frozen | 3.25 | 1.48e-2 | 0.13M | 206.85M | 13.22 |
| 3 K-FAC polish, L1 frozen | 1.86 | 1.48e-2 | 0.13M | 206.98M | 14.93 |
| 4 K-FAC final polish, all free | **1.69** | **1.69e-2** | 0.13M | 207.11M | 16.38 |

Final per-matrix max: b9 1.7, W9 0.70, W2 0.33, W3 0.22, rest 5e-2..1.5e-1. The solver dragged the 4.0 unit to 1.7 but
spread error everywhere else (mean 5.5e-3 assembled -> 1.7e-2; exact W1 7e-6 -> 2.4e-2). As with 12L, with an unsolved
trunk the assembled / frozen-L1 nets are the better artefacts. Consensus starting net declined (461 certified, order
check failed). Comparison: the 2026-10-13 run with identical flags reached 819/1800 and 8.8e-4 after the pipeline, so at
11 layers q 1.8M / 30 iterations is NOT reliable (1 of 2 clean runs). Next: q 2.4M (demo3_11L_q2.4M), same everything else.
