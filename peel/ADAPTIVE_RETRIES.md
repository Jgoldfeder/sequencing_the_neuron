# Retrying stalled design refinement

Production `--design-refine` now explores narrower and wider search brackets.
Previously, yield below 10% permanently doubled eps up to .16. A crowded search
interval can itself cause low yield, so that rule could make retries worse.
The new schedule starts at .02, tries .01, .005, .0025, .04, .08, .16, and cycles.
It keeps a productive width between periodic exploration rounds (every eight
rounds). It does not relax the teacher-kink fingerprint or fitting gates.

The requested sample count (normally 400 at width 200) is now a target. At budget
exhaustion, the solver may fit fewer points if there are at least
`max(Din+48, ceil((Din+16)*7/6))`; the same inlier, held-out, angle, singular-gap,
and uncertainty checks must still pass. At Din=200 this minimum is 252. This
allows a 380-point neuron to be evaluated instead of rejected just for missing
400; 79 points still cannot identify a 200-dimensional weight.

## Validation

On the user's cached trained teacher, seed 1,
`3072,200,200,200,200,200,200,200,100`, with the first layer removed:

- Recovered all 200 neurons at hidden layer 2 (zero-based index 1).
- Maximum absolute parameter error: 1.53e-13 in unit-weight gauge.
- 30,096,748 teacher samples, 153.2 seconds, GPU 2.
- Exact normalized prefix; target guesses perturbed with independent Gaussian
  weight and bias noise of standard deviation .003. This is a controlled fixture,
  **not** the trained committee state from the pasted failure log.
- All neurons reached the full 400-point target; the smaller-fit fallback was
  not needed in this full-layer test.
- Channels 74/166 alone: old retry schedule 1,485,592 queries / 8.87 seconds;
  new schedule 500,948 queries / 3.04 seconds. Both recovered in this fixture.
  Channel numbers refer to the teacher frame and need not equal the committee's
  permuted indices in the original production run.
- Both saved depth-10 stragglers recovered (errors 2.20e-11 and 1.58e-11).
- CPU forward-only and production integration tests pass, including FP64 export,
  partial-prefix deferral, query accounting, and scale matching.
- A separate small test recovered from 175 points despite a target of 1,000,
  with error 5.61e-15 and unchanged validation gates.

Reproduce the trained-teacher fixture from `code/`, choosing an idle GPU:

```sh
CUDA_VISIBLE_DEVICES=1 python peel/test_adaptive_retries.py --all --strategies adaptive --output peel/adaptive_teacher_full.json
```

The script requires the existing cached teacher and never trains/downloads one.
Results and per-neuron/per-round diagnostics are in `adaptive_teacher_full.json`.
The two-channel comparison is in `adaptive_teacher_retry.json`.

## Production and failure replay

Use the same `run.py ... --design-refine` command after restarting Python.
No running process is modified. Training guesses remains a separate cost.
There is no guarantee that every future guess or prefix will be recoverable.

Incomplete production solves now save a uniquely named snapshot under
`peel/design_failures/`. It includes the original student/teacher states,
channel indices, query RNG state, solver settings and round diagnostics. Teacher
parameters are saved only for diagnostic replay; they are not used to solve.
Replay without retraining:

```sh
CUDA_VISIBLE_DEVICES=1 python peel/replay_design_failure.py peel/design_failures/layer1_TIMESTAMP.pt
```

Add `--failed-only` to test just the misses, or `--strategy legacy` to compare
retry behavior. Restricting channels changes RNG consumption and therefore is
not an exact replay of the original complete-channel query sequence.

The consensus error label now says `cons, training copies` to distinguish FP32
committee statistics from the recovered FP64 prefix. Warm-freeze messages now
state the recovered row count instead of calling a partial layer exact.

## Full-layer old-schedule comparison

With the same teacher, guesses, channels and query seed, the legacy widening
schedule recovered 200/200, using 35,277,348 queries in 161.6 s.
This comparison includes the new below-target fitting fallback in both
strategies; it isolates the retry schedule rather than replaying the entire old
implementation. See `legacy_teacher_full.json` for per-channel results.
