# Scale-invariant MLP kink brackets

The new log confirms solved-weight transfer now holds (training-copy L2 error
~6e-8, stored fp64 ~1e-9). A separate sampling defect caused the 99-million-query
L2 refinement call.

In `candidate_pool`, the bracket radius divided by the guessed input-gradient
norm. Multiplying a neuron's weights and bias by a positive scalar leaves its
surface unchanged, but changed this radius inversely. Saved failed rows had
augmented norms around 0.02–0.05, leading to roughly 20–50x wider brackets than
the equivalent unit-normalized guesses. Those brackets cross many unrelated
boundaries and are difficult to isolate.

The fix normalizes local copies of the queried row and bias before candidate
geometry calculations. It changes neither stored model weights nor return gauge.
The same epsilon now describes the same geometric bracket under positive row
rescaling. No extra teacher access or parameter-error acceptance gate is added.

Validation:

- `test_design_scale.py`: locations, directions, radii and crossing scores agree
  across 100000x positive row rescaling at two frontiers.
- `test_informative_smoke.py`: callable-only recovery and query accounting pass.
- `test_mlp_partial_freeze.py` and `test_normalized_peel_pipeline.py`: pass.
- `replay_design_scale.py`: replays all 39 failed channels from the supplied run's
  `layer1_1790812936183464928.pt` snapshot, using its actual recovered prefix and
  guessed layer. No teacher prefix is substituted. Teacher weights are used for
  post-hoc scoring only. Results are in `design_scale_replay.json`, the log of
  the same name, and `design_scale_recovered.pt`. Row comparisons account for
  positive scale and permutation symmetry.

Restart Python to load the fix; existing running processes are unchanged.
This replay tests refinement from saved guesses, not convergence of a new full
training run. The saved recovered model is a diagnostic artifact, not a pipeline
resume checkpoint (it does not include committee/optimizer/pinning state).
