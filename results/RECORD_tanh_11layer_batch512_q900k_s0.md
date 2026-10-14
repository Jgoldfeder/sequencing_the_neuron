# Record: tanh 3072x200^9x100 (11-layer), mergedbest512, q=900k, window 20, batch 512, epochs 10, seed 0, teacher seed 0

Reconstructed from the session transcript on 2026-10-12. UPDATE 2026-10-13: the run's full results json was still in git (commit 2355d3d) and is recovered as results/RECORD_tanh_11layer_batch512_q900k_s0.json (all 30 iterations, per-layer consensus included). The run's own files
(results/mergedbest512_tanh__3072x200x200x200x200x200x200x200x200x200x100__s0.json, the _final.pt and the
_pop__ snapshot) were OVERWRITTEN by the untagged batch-4096 run of the same architecture launched 2026-10-12
12:26. Numbers below are exact copies of what the overwritten json / the screen printed; "-" = not captured.
The run was launched 2026-10-11 and killed at iteration 30 (complete).

## Per iteration: best loss / mean_eps / agree (CIFAR test argmax) / certified units [per layer, of 200] / cumulative wall s
| it | loss | mean_eps | agree | certified | per layer | wall s |
|---|---|---|---|---|---|---|
| 1 | 0.570 | 0.118 | 0.326 | 0 | | 152 |
| 2 | 0.561 | 0.134 | 0.392 | 0 | | 395 |
| 3 | 0.537 | 0.148 | 0.459 | 0 | | 722 |
| 4 | 0.513 | 0.155 | 0.486 | 0 | | 1137 |
| 5 | 0.497 | 0.162 | 0.519 | 0 | | 1638 |
| 6 | 0.476 | 0.170 | 0.549 | 0 | | 2380 |
| 7 | 0.458 | 0.180 | 0.572 | 0 | | 3207 |
| 8 | 0.442 | 0.192 | 0.571 | 0 | | 4118 |
| 9 | 0.429 | 0.198 | 0.599 | 0 | | 5117 |
| 10 | 0.421 | 0.208 | 0.612 | 0 | | 6200 |
| 11 | 0.404 | 0.221 | 0.603 | 0 | | 7370 |
| 12 | 0.393 | 0.232 | 0.627 | 0 | | - |
| 13 | 0.382 | 0.245 | 0.658 | 0 | | - |
| 14 | 0.371 | 0.248 | 0.658 | 0 | | - |
| 15 | 0.356 | 0.264 | 0.685 | 0 | | - |
| 16 | 0.342 | 0.270 | 0.689 | 1 | L9:1 | - |
| 17 | 0.307 | 0.271 | 0.735 | 0 | | - |
| 18 | 0.262 | 0.262 | 0.783 | 2 | L7:2 | - |
| 19 | 0.205 | 0.256 | 0.820 | 96 | L6:1 L7:73 L8:22 | - |
| 20 | 0.200 | 0.254 | 0.820 | 125 | L6:5 L7:82 L8:38 | - |
| 21 | 0.187 | 0.253 | 0.824 | 171 | L6:9 L7:95 L8:60 L9:7 | - |
| 22 | 0.174 | 0.259 | 0.833 | 218 | L6:10 L7:95 L8:73 L9:40 | - |
| 23 | 0.160 | 0.257 | 0.835 | 290 | L6:9 L7:93 L8:86 L9:102 | - |
| 24 | 0.146 | 0.255 | 0.836 | 296 | L6:10 L7:94 L8:96 L9:96 | - |
| 25 | 0.133 | 0.250 | 0.839 | 344 | L5:8 L6:42 L7:96 L8:107 L9:91 | - |
| 26 | 0.128 | 0.250 | 0.842 | 415 | L5:9 L6:86 L7:101 L8:114 L9:105 | ~28064 |
| 27 | 0.128 | 0.249 | 0.842 | 460 | L4:1 L5:11 L6:128 L7:103 L8:117 L9:100 | - |
| 28 | 0.128 | 0.249 | 0.842 | 476 | L5:11 L6:150 L7:101 L8:116 L9:98 | - |
| 29 | 0.128 | 0.244 | 0.841 | 484 | L5:11 L6:159 L7:99 L8:119 L9:96 | - |
| 30 | 0.127 | 0.243 | 0.836 | 491 | L5:11 L6:165 L7:98 L8:116 L9:101 | - |

(agree at it 10-12 as printed by the json: 0.6115 / 0.6025 / 0.6265; L1-L4 certified 0 throughout except L4:1 at it 27.)

## Matrix-level consensus (rows: direction / magnitude / full, of 200) and act-matched mean eps per layer
| it | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | W9 |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 111/31/0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 157/56/0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10 | 167/78/0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12 | 171/113/0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 13 | 174/122/0, eps 0.58 | 0, 3.9e-2 | 0, 6.5e-2 | 0, 7.9e-2 | 0, 8.0e-2 | 0, 7.2e-2 | 0, 3.9e-2 | 0, 1.6e-2 | 74/74/0, 1.7e-2 |
| 14 | 176/127/0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 16 | 184/146/0, 0.76 | 0, 3.0e-2 | 0, 5.4e-2 | 0, 7.0e-2 | 0, 6.8e-2 | 0, 4.4e-2 | 0, 1.5e-2 | 182/182/7, 1.3e-2 | 193/193/52, 1.3e-2 |
| 18 | 189/158/0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20 | 190/156/0, 0.87 | 109/38/23, 1.2e-2 | 9/6/0, 5.1e-3 | 36/31/0, 4.4e-3 | 157/153/26, 3.8e-3 | 200/195/197, 3.6e-3 | 195/182/172, 3.5e-3 | 198/168/174, 5.8e-3 | 198/175/181, 8.2e-3 |
| 25 | 194/141/0 | 85/50/12 | 48/39/1 | - | - | - | - | - | - |
| 30 | 192/137/0, 0.86 | 140/74/30, 8.9e-3 | 94/68/2, 3.7e-3 | 192/188/107, 3.4e-3 | 200/198/199, 2.9e-3 | 200/198/200, 2.6e-3 | 200/187/184, 2.7e-3 | 200/175/196, 5.1e-3 | 199/190/198, 7.8e-3 |

L1 units paired across members (ident_self): it 16: 0, it 18: 2, it 20: 149, it 25: 188, it 30: 184.

## Summary
Top-down convergence: deep layers lock in at it 16-20 (W5-W9 to 3-8e-3 by it 20-30), W4 by it 30, W2/W3 partial,
W1 never (direction 192/200 but magnitude 137 and full 0; L1 eps 0.86). Final: 491/1800 units certified, loss 0.127,
mean_eps 0.243 (dominated by W1).

## Comparison run (batch 4096, otherwise identical; its files now occupy the untagged names)
it 30: loss 0.258, mean_eps 0.057, certified 186 (all L1); W1 200/189/199 eps 5.5e-3 (first committee to finish L1
magnitudes on a tanh net), W2 133/32/11 1.8e-2, W3-W7 0 rows at 5.0e-2/7.5e-2/-/8.5e-2/7.0e-2, W8 0 at 2.0e-2,
W9 0 at 8.0e-3 (43 full at it 25, lost by 30), head cols 200/200 at 3.6e-3. Bottom-up convergence: the mirror image.
Per-iteration wall: 234 s/iter to it 5 vs 328 (1.4x faster); 3984 s to it 9 vs 5117.
