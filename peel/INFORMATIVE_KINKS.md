# Reconstruction-designed kink sampling

Experimental strategy for `kink_solve.recover_layer(..., sampling="design")`.
The default remains `sampling="track"`.

The sampler uses only the reconstructed prefix and current layer to generate
candidate inputs. It projects independent random inputs at scales 1, 4, and 16
onto the guessed neuron surface, with a model residual check. It evaluates
predicted prefix/sibling crossings across each uncertainty bracket.

To choose teacher queries, it removes the guessed normal's largest coordinate
from augmented hidden features, whitens these against the collected features
(or candidate pool initially), and uses pivoted QR to select complementary
points. Brackets with fewer predicted foreign crossings get a soft preference.
Strict isolation is available for ablation, but full uncertainty brackets are
almost never isolated in the deepest layers of this fixture.

Selected brackets are scanned and fingerprinted using the existing teacher
query locator. No teacher weights, hidden activations, or gradients enter the
recovery. The benchmark uses teacher weights to create the stipulated initial
guess and to measure error only. Downstream reconstruction weights are zeroed.

The experiment collects at least 400 points per neuron, reserves every seventh
point from the fit, and uses the existing robust hyperplane solve. Acceptance
checks fit inliers, held-out residuals, agreement with the guess, singular-value
gap, and a conditioning-scaled numerical noise indicator. These are heuristic
checks, **not a parameter accuracy certificate for an approximate prefix**.
In particular, coherent prefix errors can bias both fit and held-out samples.

The return convention through `kink_solve.recover_layer` matches the existing
API: accepted rows have unit augmented `[w,b]` norm; callers may convert to the
unit-weight gauge. The helper in `peel/informative_kinks.py` returns unit-weight
rows. Missing/failed rows retain their original guess.

## Reproduce (from code/)

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  python peel/test_informative_kinks.py --layers 6 --channels 200 \
  --output peel/informative_full_layer6.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  python peel/test_informative_kinks.py --recursive \
  --checkpoint peel/informative_recursive.pt --output peel/informative_recursive.json
```

`--resume peel/informative_recursive.pt --recursive` continues after the last
fully accepted layer. It restores the student and initial-guess noise generator.
A failed layer is not frozen. The teacher fixture is regenerated from `--seed`
(default 0). This benchmark is synthetic: good layer guesses still come from
teacher weights plus prescribed noise.

Ablations: `--selection random` removes QR selection and isolation preference;
`--scales 1` removes multiscale input sampling; `--selection isolated` requires
strict guessed isolation across the entire uncertainty bracket; `--selection
track` uses the previous solver. Exact-prefix tests default to 12 neurons at
layers 3, 5, and 6. Recursive tests always attempt all neurons per layer (200 by default).

CPU integration check:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python peel/test_informative_smoke.py
```

Small exact-prefix ablations before batching/seed-generation optimization:
- Layer 6 design + multiscale: 12/12, max error 4.38e-13, 1,206,068 queries.
- Layer 6 random + multiscale: 12/12, max error 6.74e-13, 1,746,838 queries.
- Layer 6 design at scale 1: 10/12; accepted max error 8.43e-13.
- Strict isolation: 0/12 after eight rounds, no candidates queried.

These compare adaptive runs, not identical fixed query sets. Current fast
candidate generation also passed 12/12 at layer 6 (max error 3.93e-13).
Full-width and recursive results are recorded in the JSON/log artifacts above.

## Full-width validation (seed 0)

The recursive design run completed all seven hidden layers (1,400 neurons),
with 200/200 accepted in every layer. Maximum unit-weight/bias coordinate errors:

| Layer | Tracked baseline | Designed sampling |
|---|---:|---:|
| 0 | 7.89e-13 | 2.13e-14 |
| 1 | 1.63e-11 | 4.24e-13 |
| 2 | 5.55e-09 | 3.54e-13 |
| 3 | 5.13e-04 | 1.02e-12 |
| 4 | 8.24e-02 | 3.53e-12 |
| 5 | stopped | 1.09e-11 |
| 6 | stopped | 2.52e-11 |

The baseline stopped at layer 4 with 191/200 accepted; its accepted rows
also contained large errors. Both runs used the same benchmark teacher and
initial guesses. The sampling strategies consume randomness differently.

Design hidden-layer queries: 85,567,440; output fit: 4,000.
Sum of hidden-layer wall times: 606.0 s, with other
benchmark jobs sharing the GPU during parts of the run. Through layer 4,
design used 6.03x the baseline's teacher queries.

Output-layer parameter error: 3.600e-10. Maximum function discrepancy
on 2,000 fresh Gaussian inputs: 5.820e-12. This is an empirical test,
not a bound over all inputs.

Separately, the deepest hidden layer with an exact prefix passed 200/200
with maximum parameter error 6.50e-13 (18,473,144 queries). Recursive prefix
error still limits final hidden accuracy to 2.52e-11 and output parameter
accuracy to 3.60e-10. This improves recovery markedly but does not establish
1e-13 parameter recovery throughout the network.

The successful sampler combines broad independent candidates with information
selection and a soft isolation preference. Strict full-bracket model isolation
failed in the ablation; isolation alone is not the demonstrated fix.

Validation is on this synthetic seed-0 fixture with prescribed good guesses.
The CPU smoke check also covers a different rectangular-prefix fixture.
The production runner now exposes `--design-refine` (see below). The old
`--loc-refine` still selects legacy tracking. The full-size trained-committee
run has not been benchmarked; small seeded production-path tests pass.


## Configurable network depth and the role of guesses

The benchmark now accepts any positive `--hidden-layers N` (default 7),
`--width W` (default 200), `--input-dim D` (default W), `--output-dim O`
(default 100), and `--device cuda` or `--device cpu`. There is no fixed
seven-layer cap; successful recovery at arbitrary depths is not guaranteed.

From code/, for example:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  python peel/test_informative_kinks.py --recursive \
  --hidden-layers 10 --width 200 --output-dim 100 \
  --output peel/run_depth10.json --checkpoint peel/run_depth10.pt
```

Hidden-layer count excludes the input and linear output. Thus the command
above has ten 200-neuron hidden layers. For an exact-prefix test of its last
hidden layer instead, omit `--recursive` and pass `--layers 9 --channels 200`.
`--layers` lists zero-based layer indices to test; `--hidden-layers` sets the
network depth. Resume with the same architecture and seed flags plus
`--resume peel/run_depth10.pt`. The old seven-layer checkpoints remain readable.

The runner prints teacher-query counts, wall time, mean absolute parameter
error, mean per-neuron maximum error, median per-neuron maximum error, and
maximum parameter error for each hidden layer. Output fitting now records
its time and mean error too. Results are saved incrementally to JSON. It
stops recursive peeling if a layer has any rejected or unsolved neurons.

This is explicitly a good-guess refinement benchmark. At each hidden layer,
the guessed weights are the teacher's weights plus Gaussian noise with
standard deviation 0.01/sqrt(layer input dimension), and bias noise has
standard deviation 0.005. Earlier layers are the actual recovered layers
in recursive mode; future reconstructed layers are zeroed. Teacher parameters
are used by the benchmark to construct the stipulated guess and report error;
the solver itself uses only teacher forward queries.

The solver relies on the guess for projecting candidate inputs to a predicted
kink, sizing the search bracket, fingerprinting which crossing belongs to the
target neuron, choosing coordinates for information selection, and checking
the recovered row's angle/sign. A random or absent guess is outside the tested
regime. Multiscale random candidate inputs do not make the method guess-free.

Runner validation: a recursive 6→8→8→8→4 CPU network recovered all
24 hidden neurons (worst hidden error 4.42e-14). Checkpoint resume
preserved the hidden results and refitted output parameters below 3e-14;
the least-squares refit was not bitwise identical. A ninth-hidden-layer
test on a much narrower 6→[8]*9→4 fixture executed but recovered only
1/2 tested neurons; its accepted error was 9.09e-8. Configurable depth
is not a guarantee of accurate extraction at arbitrary depth.


## Production peel integration: `run.py --design-refine`

Add `--design-refine` to the existing command. This routes the MLP epsilon-gated,
fast committee-consensus, partial, and peel-try refinement calls through
`recover_layer(..., sampling="design")`, including the first hidden layer.
It implies `--f64`. Existing peel triggers, learned/consensus guesses,
scale matching, restart behavior, and `--peel-direct` query generation stay in
place. Do not combine it with `--loc-refine` or `--xspace-refine`.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
python run.py --variant mergedbest512 \
  --arch 200,200,200,200,200,200,200,100 \
  --teacher-full 3072,200,200,200,200,200,200,200,100 \
  --teacher-drop 1 --teacher-seed 1 --seed 1 \
  --outer 60 --q 48000 --window 60 --device cuda \
  --cheat --cheat-pop 8 --cheat-solo --fast-peel --peel \
  --cheat-peel-mean 3e-3 --cheat-peel-max 1e-1 \
  --peelrestart --peel-direct --f64 --design-refine --tag design
```

The thread limits keep small CPU factorizations from oversubscribing cores.
`--tag design` gives this experiment separate output filenames.

The training `--q` budget does not cap refinement probes. Refinement prints its
oracle-query count and saves cumulative `peel_refinement_queries` in logs and
final results separately from the existing training `queries` field.
Incomplete prefixes defer deeper refinement in a multi-layer peel attempt.
The final model overlays fp64 frozen rows before delivery; population snapshots
include the fp64 `frozen` dictionary. Pre-endgame reconstruction checkpoints
also include `refined_state`, preserving full-precision rows alongside the
ordinary training population state.

Validation: `peel/test_design_pipeline.py` exercises the actual reconstruction
loop with a small seeded good-guess committee: epsilon triggering, committee
consensus triggering, peel restart, direct-frontier mode, both hidden layers,
scale matching, an intentionally incomplete prefix, query accounting, and fp64
checkpoint/final-model preservation. It does not substitute a fake refiner.
The requested CLI configuration was checked with teacher training replaced
and reconstruction intercepted; the full 60-iteration job was NOT launched.
The existing `--cheat` training mode remains a teacher-assisted diagnostic;
the new refinement algorithm itself uses the guesses and teacher forward queries.

## Jacobian-jump width experiment

A separate forward-query prototype and widths 200/400 comparison are documented
in [JUMP_KINKS_REPORT.md](JUMP_KINKS_REPORT.md). It reduces teacher samples but
does not match the point solver precision with an exact prefix; production peel
still uses the existing method.

## Production round slowdown: CPU thread pools

The production command originally inherited 48-thread NumPy/SciPy BLAS pools,
while standalone benchmarks restricted CPU threads. This severely slows the
small per-neuron SVD/QR operations. `recover_layer` now scopes a one-thread
limit via `threadpoolctl` to refinement and restores the previous pools on
return or exception. An already running Python process must be restarted to
load this change. The current process is not modified or stopped.

A paired GPU-1 test on the same eight channels, same guesses and RNG seed, one
sampling round: original pools 17.99 s; scoped limit 0.344 s (after warmup).
Both made 44,170 teacher queries and collected identical per-neuron point
counts. See `check_refinement_threads.py`, `refinement_threads_check.json` and
`.log`. Forward-only recovery smoke tests still pass. A separate full
200-channel one-round probe took 9.3 s / 1,080,650 queries; it deliberately
stopped before collecting sufficient points to fit. This is a synthetic
fixture, not the user's running committee state.

The increasing outer-training iteration times are separate: with `--window 60`,
each iteration trains over the growing retained query buffer until 60 rounds
have accumulated (or a restart clears it). Thread limiting inside refinement
does not change that training workload.

## Stalled-neuron retry fix

See [ADAPTIVE_RETRIES.md](ADAPTIVE_RETRIES.md) for narrower/wider bracket retries,
validated fits below the sampling target, production failure snapshots, and a
200/200 L2 check on the cached trained teacher.

## Wide-input shallow models

For one-hidden-layer models with input dimension >=1024 and hidden width no
larger than the input dimension, design dispatch now uses isolated affine sweeps.
The saved 3072->256->100 consensus refined all 256 neurons in about 20 seconds.
See [SHALLOW_SWEEPS.md](SHALLOW_SWEEPS.md) for measurements and limitations.
