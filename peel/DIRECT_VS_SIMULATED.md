# Paired direct-tail versus inverse-simulated labels

Identical student initialization, fixed hidden inputs, minibatch permutations,
optimizer, learning-rate schedule and update counts. Only training labels differ.
Teacher is the user's cached seed-1 model; exact prefix of four layers; student
is the remaining 200 -> 200 -> 200 -> 100 tail. FP64 teacher/inverse, FP32 student
and labels (matching production training). 16,384 uniform [-1,1] training inputs,
2,048 independent validation inputs, 120 epochs, batch 512, L1 loss, Adam 1e-3,
learning rate divided by ten at 60% and 85% of epochs. Three initialization seeds.
No active query search, growing buffer, committee, restart, or kink refinement.

Final validation L2 error relative to the true teacher-tail output norm:

| Seed | Direct labels | Simulated labels |
| --- | ---: | ---: |
| 1 | 21.62% | 187.37% |
| 2 | 21.35% | 181.03% |
| 3 | 21.55% | 186.55% |

All initial relative errors were approximately 100%. Thus direct supervision
substantially improves function approximation; simulated supervision ends farther
from the true function than initialization. The results are consistent across
these three initializations. The simulated labels themselves have 205.7% relative
error against direct labels on the combined training/validation inputs.

However, neither arm recovers the true parameters. Direct-arm normalized L5
weight maximum errors remain .36–.42, with bias errors .62–.74. The clean control
therefore does not establish successful parameter recovery under this protocol.
The experiment demonstrates a causal adverse effect of changing the labels,
not that label corruption alone explains the entire production stall. It also
cannot identify whether the remaining clean-control difficulty is data coverage,
optimization, or another issue. Fixed random queries are different from production's
adaptive query search and accumulated dataset.

Reproduce from code/ with an idle GPU:

```sh
OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python peel/test_direct_vs_simulated.py
```

`direct_vs_simulated.json` contains configuration, oracle errors, validation
curves, final aligned parameter errors and run times. `direct_vs_simulated.log`
contains progress. The cached teacher is required. Production was not modified.
