# Normalized peeled training and reachable disagreement search

Experimental method: `peel/normalized_peel.py`, now available in `run.py` as
`--normalized-peel`. Use it instead of `--peel-direct`; the flags are mutually
exclusive. CNN behavior is unchanged.

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --variant mergedbest512 \
  --arch 3072,200,200,200,200,200,200,200,100 \
  --outer 60 --q 48000 --window 60 --device cuda \
  --peel --fast-peel-partial --peelrestart --design-refine \
  --normalized-peel --retry
```

Before the first complete frozen prefix, ordinary training/sampling runs.
Afterwards, each member has a normalized tail and its own Adam state. Raw-input
queries and labels stay in the usual replay buffer. Tail weights are exported to
original coordinates after training, so consensus and refiners keep their existing
interfaces. Solved rows are masked during normalized training and excluded from
write-back. Stages rebuild after parameter/pin/frontier changes, including retries;
otherwise normalized Adam moments persist. The schedule follows the ordinary
optimizer learning rate. Calibration uses no teacher queries.

Normalized sampling uses committee output variance (plus the teacher when
`--cheat` is explicitly enabled), rather than the variant's ordinary disagreement
metric. `--cheat-solo` retains per-member query ownership. It optimizes batches of
up to 256 queries with `--qg_lr` and the configured query step count; original
coordinates are bounded by 4 and standardized proposal coordinates by 3 after
each update. The existing query-box/diversity options do not configure this new
sampler. Calibration and extra prefix computations add overhead; this is not a
speed optimization. Unsupported: CNN, non-leaky-ReLU, hard labels, factored tails,
and ensemble-boost. Ordinary `--design-refine` remains the separate refiner.

CLI wiring, production reconstruction, incomplete-prefix deferral, optimizer
persistence and pinned-row preservation are checked by
`test_normalized_peel_cli.py`, `test_normalized_peel_pipeline.py`, and
`test_normalized_peel_state.py`. The CLI test intercepts dataset loading and
reconstruction; the pipeline test separately executes actual training and
refinement. No long seven-hidden-layer run has been performed.

Given a recovered prefix and a guessed tail:

1. Forward random bounded original inputs through the recovered prefix. Fit a
   regularized full-covariance whitening transform for the frontier inputs.
   Freeze these coordinates for the training stage. No teacher calls are used.
2. Rewrite the first remaining student layer in whitened coordinates. Its
   predictions are preserved up to float32 rounding. Adam optimizes these new
   parameters, rather than merely normalizing data and undoing it before the old
   layer. Export converts the learned tail back to the original coordinates.
3. Fit a linear proposal from whitened hidden features to original-input latent
   coordinates. It is a regression map, not an exact prefix inverse. Disagreement
   optimization searches standardized latent coordinates, with per-query random
   original-input residuals to explore outside the proposal's span.
4. Bound original inputs with tanh, evaluate the actual recovered prefix, and
   evaluate student disagreement. Gradients go only to the query variables.
   Prefix activation boundaries may be crossed: no narrow local-cell constraint.
5. Query the forward-only teacher at the selected original inputs. Train the
   peeled student on their actual whitened hidden features and teacher labels.

The search keeps each sample's best-scoring iterate, ensuring its measured
committee disagreement does not decrease. This is not a guarantee of useful
teacher error or generalization. Weak prefix variation, approximate prefix
weights, poor committee diversity and nonlinear proposal mismatch remain risks.
No normalization can manufacture unobservable directions. Calibration reports
how many covariance directions lie above its regularization floor.

## Run the synthetic good-guess benchmark

From `code/`:

```bash
OPENBLAS_NUM_THREADS=1 python peel/test_normalized_peel.py
OPENBLAS_NUM_THREADS=1 python peel/test_normalized_peel.py \
  --arch 3072,200,200,200,100 --frontier 2 \
  --output peel/normalized_peel_wide.json
```

`--frontier 1` trains L2 onward after L1 is recovered; `--frontier 2` trains L3
onward. `--device cuda:0`, `--seed`, and `--rounds` are also supported. These tests
construct an exact prefix and noisy tail; they are not a full extraction run.
The peer is fixed in this bounded benchmark. Tests require increased disagreement,
nontrivial hidden movement, decreased training and held-out error, preserved
initial predictions, and an export that leaves the prefix untouched. JSON results
and the exported student `.pt` are saved at the chosen output stem.

## Use with existing models

```python
from peel.normalized_peel import NormalizedPeel
stage = NormalizedPeel(guess, frontier=1, calibration=2048)
x, sampling = stage.sample([other_guess], count=256, steps=30)
with torch.no_grad():
    y = teacher(x)  # forward only, one query per input
training = stage.fit(x, y, epochs=30)
reconstruction = stage.export()
```

`other_guess` consumes original inputs. Keep a replay buffer across rounds, as
in the benchmark. Start a new stage when the frozen prefix changes. Teacher
weights are never inspected by the implementation. Only the fixture creates
its stipulated good guesses from teacher weights. Training remains float32;
calibration and prefix/proposal computations use float64.

## Initial measured results

- `128,32,32,32,8`, frontier 1: held-out MSE 0.0179 → 0.001701; 768 training queries, 512 held-out queries, 0.5s for sampling/training/export checks (calibration excluded).
- `3072,200,200,200,100`, frontier 2: held-out MSE 0.09193 → 0.02323; 768 training queries, 512 held-out queries, 9.4s for sampling/training/export checks (calibration excluded).
