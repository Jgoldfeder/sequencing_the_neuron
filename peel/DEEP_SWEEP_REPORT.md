# Fast-sweep test on 3072 -> 3072 -> 3072 -> 3072

Two hidden layers and a linear output. Synthetic seed-0 teacher, float64,
Gaussian biases std .1 and unit-weight normalization. Target guesses use Gaussian
weight noise .01/sqrt(3072), bias noise .005. Two neurons per case, two independent
placements each (the production default permits six). This is a bounded probe,
not a full-layer solve or exhaustive retry test. Production dispatch unchanged.

| Case | Accepted | Teacher queries | Seconds |
| --- | ---: | ---: | ---: |
| first_full | 0/2 | 24,600 | 117.0 |
| second_direct | 0/2 | 24,600 | 115.1 |
| second_simulated | 0/2 | 24,600 | 119.8 |

`first_full` uses forward queries to the complete teacher for its first layer.
`second_direct` is a privileged true-tail control, bypassing the first layer.
`second_simulated` calls the full teacher after inverse-prefix mapping. The latter
uses an EXACT prefix and LU solves, stronger assumptions than a recovered prefix
and the older normal-equation inverse. All rejected rows remain original guesses;
returned parameter errors for these rows are not recovery accuracy measurements.

No sampled case passed. Rank-one residuals were order one, versus the 1e-9 gate.
Even the direct-tail control failed: square width changes the isolation problem
as well as the cost. The original successful 3072->256->100 model has a much
smaller, underdetermined guessed weight matrix and only 100 output coordinates.
Here both the right-inverse factorization and slope-jump SVDs are 3072-scale.

A teacher-internal diagnostic (NOT used by the solver), `check_wide_sweep_geometry.py`,
reconstructed the same query centers using LU and inspected actual crossings.
For second-layer channel 0, the two brackets did not cross the intended neuron
and crossed 13 and 30 other neurons. For channel 1, they crossed 9 and 5 other
neurons; only the first crossed the target. Small parameter noise is amplified
when using the square guessed inverse to prescribe all preactivations. These
sweeps therefore do not isolate the target kink.

A width-200 control used the same two-neuron/two-placement protocol. First-full
and simulated-tail cases accepted 0/2. True-tail access accepted 2/2, with max
parameter error 4.18e-12. This illustrates that the failures cannot all be
attributed to prefix inversion, or all to depth alone.

Reproduce from code/ on an idle GPU:

```sh
OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python peel/test_deep_shallow_sweep.py
```

Data: `deep_shallow_sweep.json`, `.log`, `deep_shallow_sweep_200.json`,
`wide_sweep_geometry.json`. Timings are exploratory and include per-case setup;
the small geometry diagnostic briefly shared the test GPU during the last case.
