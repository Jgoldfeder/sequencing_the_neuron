# Handoff: deep-layer kink solver (`--loc-refine`)

## Where things are
- Solver: `code/kink_solve.py` (single file, ~800 lines). Entry point `recover_layer(teacher, cons, frontier, device, only_channels=None, gen=None, angle_gate=12.0, ..., direct=True)`
  -> `(W_ref, b_ref, refined_mask, n_queries)`, unit `[w|b]` rows for refined channels.
- Pipeline wiring: `code/method.py` `_mlp_refine_layer(...)`, branch `if loc and frontier > 0:` calls `kink_solve.recover_layer`.
  `run.py --loc-refine` sets `cfg.loc_refine`. Also added in method.py: `_mlp_prefix_jac` (exact, autograd-free prefix Jacobian),
  `deep_tries` kwarg on `_mlp_refine_layer`. `loc_refine.py` is dead code.
- Test scripts (scratchpad, copy them somewhere durable):
  - `/tmp/claude-1001/-home-judah/acffa24f-99ed-4702-a22e-fad9a5bc3978/scratchpad/test_layers.py 3 4 1`
    single layers of a random 200x...x100 net, EXACT prefix + synthetic 1e-2 guess; prints count / max / median err / queries / time.
  - `/tmp/claude-1001/-home-judah/acffa24f-99ed-4702-a22e-fad9a5bc3978/scratchpad/peel_recursive.py`
    the target test: recursive peel of a random `[200]*8+[100]` net, layer by layer (guess = recovered prefix + true layer + 1e-2 noise),
    output layer by least squares, prints per-layer error/time. Run: `CUDA_VISIBLE_DEVICES=0 python <path>` from `code/`.
  - checkpoint test fixture from the original run: `recon/_pop__mergedbest512_cheat__200x200x200x200x200x200x200x100__s1.pt`
    (`teacher_state` + 8 `pop_states`; layer 0 exact to 1e-8, layer 1 guess ~5e-3).

## Algorithm (direct mode, the one Judah wants: forward-only, NO prefix inversion)
Per layer l, all channels batched:
1. `polish_layer(..., full=True, need=48, max_rounds=1, return_points=True)`: ONE scan round -> a few exact "anchor" kink points per neuron.
   Seed random x, Newton-project onto the guessed kink of neuron j (model-only), scan the bracket along the guess normal (K=25+12l points),
   fingerprint each bend's normal against the guess with probes orthogonalized in h-space (`_probe_dirs`), accept ours.
2. `track_layer(...)`: from the anchors, TRACK the kink surface: random tangent step (0.5), project onto the model surface shifted by the
   last known model-vs-true offset, relocate the true kink along the normal with `locate_light` (8 queries: 3+3 collinear side points,
   intersection, 2 verification points at t*+-R/500). Bracket R = 5*(1e-2*|dh|/sqrt(Din))/|n|. ~240 points/neuron.
3. `solve_from_points`: null vector of [h 1] with leverage-aware (LOO-residual) trimming, on CPU.
4. Gate: inliers >= Din+8 and angle to guess <= angle_gate.
(`direct=False` = older two-stage path: seal refiner on a CPU fork pool -> polish. Works to 1e-14 at layer 1 but inverts the prefix; deprecated.)

## Measured (exact prefix, synthetic 1e-2 guess, RTX 3090 fp64)
| layer | solved | max err | median | q/neuron | wall |
| 0 | 200/200 | 5.8e-13 | 2.9e-13 | 5.5k | 11 s |
| 3 | 200/200 | 9.3e-11 | 1.5e-11 | 9.3k | 16 s |
| 4 | 198/200 | 2 unsolved | 2.0e-11 | 10.5k | 17 s |
Layer-1 checkpoint (real guess): 200/200, 3.8e-14 exact prefix / 7.6e-8 with its 1e-8 prefix (two-stage path numbers).

## Open items (in order)
1. Layer 4's two failures = channels with ZERO anchors from the single scan round. Fix (written, not applied): after the anchor round,
   rerun `polish_layer(full=True, need=8, max_rounds=12)` for `[c for c in rest if anchors.get(c) is None]` and merge.
2. Speed: `track_layer` does ~240 sequential steps (~10 s). Run `par=4` walks per channel per step (rows = act.repeat_interleave(par)),
   with preallocated point storage `P (n_chan, need, d)` instead of per-channel lists. (Written, not applied.)
3. Precision at depth is 1e-11 (scan-only gave 3e-13): tracked-point t* precision ~ eps_mach*scale/dsn; consider a tighter second
   locate or a larger bend-strength floor if 1e-12 is required.
4. A layer with ANY unsolved neuron poisons the next layer's prefix (the guess row stays at 1e-2). Either retry those neurons or
   make the peel stop; currently it continues.
5. Wire/verify end-to-end in `run.py --loc-refine` (the pipeline caller rescales unit rows to the guess gauge; never run end-to-end).

## Traps already hit (don't repeat)
- Bracket isolation against earlier-layer or sibling kinks kills seed yield exponentially with depth; unnecessary (fingerprint handles them).
- x-space fingerprint probes pass 4.5% of foreign kinks (prefix Jacobian anisotropy); orthogonalize in h-space; probe step dl << straddle e.
- Verification points at +-R/8 let a 2nd bend hide -> 1e-7..1e-5 point errors; use +-R/500.
- RANSAC useless when subset size ~ n; LS absorbs high-leverage outliers -> rank by LOO residual, fixed rounds, lowest median wins.
- cuSOLVER SVD of 240x201 is 87 ms; do the solve on CPU.
- fork pool after CUDA autograd crashes; spawn re-imports an unguarded main script (storm). Direct mode uses no pool.
- Bracket-width widening must stay enabled for nearly-always-active neurons (kink in the data tail); cap at 8x.


## Follow-up diagnosis (2026-09-23)

The direct path already had parallel tracking and anchor retries on inspection.
The dominant precision failure is **conditioning of the sampled feature matrix**,
not merely error in locating kinks. Avoiding explicit prefix inversion does NOT
make parameter recovery insensitive to prefix errors: solving `[h,1] v = 0`
still amplifies feature errors by the inverse smallest nonzero singular value.

Baseline recursive reproduction (zero-based layers, max unit-weight/bias error):
- L0: 200/200, 4.9e-13
- L1: 200/200, 2.3e-11
- L2: 200/200, 4.9e-9
- L3: 200/200, 6.0e-6
- L4: 192/200, 1.6e-2
- L5: 1/200, 8.6e-2
Stopped the baseline during L6; no completed end-to-end result claimed.

Controlled exact-prefix experiment: `peel/diagnose_kink_conditioning.py`, output
`peel/kink_conditioning.log`. First 12 neurons, same random teacher; compare
240 tracked points (step 0.5), 240 tracked points (step 5), and up to 280
independent scan points. Different point counts and random draws mean this is
an exploratory comparison, not an isolated step-size benchmark. All recovery
uses the query oracle; teacher weights are used only to measure errors.

At L3, median max-coordinate weight/bias errors were respectively 1.24e-11,
1.38e-12, and 1.63e-13. Median identifiable condition numbers
sigma_max/sigma_next-to-smallest were 1.36e5, 1.38e4, and 3.89e3.
Tracked points' median true-hyperplane residual was only 1.86e-15.
At L6, one tracked neuron had condition number 5.23e12 and parameter error
3.04e-5 despite maximum true-hyperplane point residual 1.78e-15. Projecting
its points onto the true feature-space hyperplane still yielded 2.87e-5
parameter error. Independent sampling reached 10/12 channels at L6 within
its budget, so it is not a complete recovery solution either.

The short walks branch from random stored points and collect local clusters.
Counting Din+40 points does not establish that all tangent directions are well
constrained. The 0.01 negative slope and deep prefix can make the locally
sampled features extremely anisotropic. Larger steps and independent anchors
help empirically; neither has yet been validated as an end-to-end fix.

The direct path discards `gap` returned by solve_from_points. Its acceptance
gate uses self-fitted residuals and a 12-degree angle, which cannot certify
machine-accurate weights. Even a tiny singular-value ratio alone is not a
complete accuracy certificate: the absolute identifiable singular scale and
feature/oracle uncertainty also matter.

Changes made:
- Retry anchors with need=48 instead of 8. With need=8 the per-channel query
  cap is 12, below the >=40 threshold that enables widening; retries could
  never widen a missed bracket. Exact-prefix L4 now gets anchors for 200/200.
- Recursive test raises on an incomplete mask before freezing the layer.
  This does not catch inaccurate rows falsely accepted by the solver.
- Added the reusable conditioning diagnostic; it sets one CPU thread.

Validation: after the retry fix L4 reports 200/200, but some accepted rows
still err by 1e-6 (logs: peel/kink_retry_layer4.log and
peel/kink_layer4_diagnostic.log). Diagnostic rows 13,134,141,195 had true point
residuals below 6e-15 but identifiable condition numbers around 7e10--4e11.
Thus the retry fix repairs coverage only; machine-accuracy recovery remains
unsolved. Small numerical differences between runs/CPU thread counts are
expected for these ill-conditioned fits.

Next numerical work: retain conditioning diagnostics; acquire fresh,
geometrically diverse points when weak directions remain; validate with an
independent sample set; use a conditioning-and-noise-based accuracy gate;
then rerun the recursive recovered-prefix experiment. Do not claim that
smaller locator brackets or more short-walk points alone resolve this.


## Reconstruction-designed sampling experiment

Implemented opt-in `recover_layer(..., sampling="design")` in
`peel/informative_kinks.py`. Full commands, ablations, and result tables are
in `peel/INFORMATIVE_KINKS.md`. Default tracking behavior is unchanged.

- Exact-prefix deepest hidden layer: 200/200, max error 6.50e-13.
- Recursive extraction: all seven hidden layers completed, 200/200 each;
  worst hidden error 2.52e-11; output parameter error 3.60e-10;
  max function gap 5.82e-12 on 2,000 fresh inputs.
- Old tracked baseline under the same harness stopped at layer 4;
  max error was already 5.13e-4 at layer 3.
- 85,567,440 hidden recovery queries, plus 4,000 output-fit queries;
  about 606 seconds summed hidden-layer time with competing benchmark jobs.
- Strict model isolation across the full uncertainty bracket produced no
  queried candidates in the layer-6 ablation. The improvement comes from
  diverse multiscale sampling plus information selection, with isolation
  preferred rather than required and teacher verification retained.
- This solves the previous depth stall on the synthetic fixture, but does
  not yet reach 1e-13 parameters recursively. No production pipeline run.

Artifacts: `peel/informative_recursive.json`, `peel/informative_recursive.log`,
`peel/informative_recursive.pt`, `peel/informative_full_layer6.json`, and
`peel/informative_track_baseline.json`. CPU integration check:
`python peel/test_informative_smoke.py` (single BLAS thread recommended).


## Production routing update

`run.py --design-refine` now selects the new sampler for all MLP hidden
refinement layers, including L0; implies fp64 prefix storage. Legacy flags
keep their old behavior. Details and the full user command are in
`peel/INFORMATIVE_KINKS.md`. Tested the actual small seeded reconstruction/
consensus/restart/freeze paths in `peel/test_design_pipeline.py`, query
accounting, incomplete-prefix deferral, and fp64 exports. The full user
60-iteration production job has not been launched.

## Update (Sep 24): ConvNet support, --fast-peel(-partial)
- `kink_solve.py` now works on `nets.ConvNet` layers via the unit model (`ConvNetUnits`); the CNN peel refiner
  (`method._cnn_refine_layer`) routes to it under `--loc-refine`. `peel/test_conv_kink.py 0 1 2 3` = synthetic LeNet check
  (conv1 9e-14, conv2 1.6e-11, conv3 ~4e-12 median, fc84 8e-11 with exact prefixes).
- `--fast-peel` now works on the CNN path and implies `--peel` with a 100% consensus quorum (any mode).
- `--fast-peel-partial` (CNN + MLP): per-neuron consensus peel; members aligned to a common frame; solved rows injected into
  every member at its own magnitude and pinned in place; frontier advances only when the layer is fully solved.
  MLP path validated end-to-end (784x32x10 smoke). CNN inject path not yet exercised by a run that reached consensus.
- Fixed `ConvNet.clone()` fp64->fp32 truncation (nets.py).
- LeNet command: `python run.py --variant mergedbest --conv lenet --cnn-act leaky_relu --device cuda --outer 60 --window 60
  --q 20000 --fast-peel-partial --peelrefresh --loc-refine` (committee; ~500 s/iter) or the `--cheat --peeltry --peelrefresh
  --loc-refine` variant (p=1, ~30 s/iter).
