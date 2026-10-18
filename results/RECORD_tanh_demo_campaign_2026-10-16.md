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
