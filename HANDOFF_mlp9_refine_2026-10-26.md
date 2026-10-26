# Handoff: exact, query-efficient recovery of a 9-layer leaky-ReLU MLP from a committee guess

Repo dir: `/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code` (branch `october2026`, changes below NOT committed).

## Goal
Black-box (forward queries only) recovery of a trained leaky-ReLU MLP **3072 → 200×8 → 100** to machine precision,
starting from an 8-member committee snapshot that already has ~99% of units approximately right (~1e-2), using as
few extra queries as possible ("kink-refine the guess").

- Snapshot: `recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt`
  (keys: `dims`, `act`, `teacher_state`, `pop_states` = 8 members). The committee used 200k queries.
- The teacher's weights are read ONLY for scoring (`align.param_errors`); the refiner queries the teacher forward only.
- Error metric = max |param error| after permutation + scale alignment (per matrix W/b).

## Current driver: `refine_snapshot_mlp.py` (new)
```
CUDA_VISIBLE_DEVICES=0 python -u refine_snapshot_mlp.py <snapshot> --unit-report --out recon/_refined_mlp9_track.pt
# resume from saved layers:  --resume recon/_refined_mlp9_track_L1-5.pt --start-layer 6
```
Per hidden layer l (layers < l already refined = exact prefix):
1. **Guess member, teacher-free**: the member whose rows the other members reproduce most (cos > 0.999 after
   alignment). Picks member 2. (Member with lowest output error, member 6, was the worst member in layers 4–8.)
2. **L1**: `method._mlp_refine_layer(design=True)` → `kink_solve.recover_layer(sampling="design")` →
   `peel/hybrid_sweep.recover_first` (wide-input affine sweeps + rank-1 Jacobian jumps).
3. **L2+**: `kink_solve.recover_layer(sampling="track", track_step=2.0, track_need=2*Din)` (surface tracking from
   anchor kinks, null-vector fit of the kink points in h-space). Forward-only, no prefix inversion.
4. **Exactness check**: a second independent track (different seed); rows whose two solutions differ by more than
   max(1e-7, 50 × layer median gap), or that fail, are re-solved with design.
5. Units still unsolved → design from track's failures → **committee "orphan" rows** (every member's rows that match
   none of the refined rows, clustered across members, most-supported first) as alternative guesses → (just added,
   untested) **guess-free** row from `guessfree.recover_frontier_rows` (unexplained kinks + prefix-Jacobian mapback)
   as a guess for design.
6. Refined unit rows are installed scaled to the guess's magnitude (keeps the next layer's guess in its gauge).
7. Output layer: closed-form least squares on 20k Gaussian queries.
8. Stops at the first layer that is not 200/200 (one wrong unit corrupts every layer above).

## Best result so far (L1–L5 exact, stops at L6)
| layer | rows | max eps W | max eps b | cumulative queries |
|---|---|---|---|---|
| L1 | 200/200 | 2.2e-11 | 1.8e-10 | 12.0M |
| L2 | 200/200 | 3.1e-10 | 4.2e-10 | 15.8M |
| L3 | 200/200 | 2.9e-08 | 6.0e-10 | 19.3M |
| L4 | 200/200 | 6.0e-07 | 2.3e-09 | 23.2M |
| L5 | 200/200 | 7.4e-06 | 2.2e-08 | 27.3M |
| L6 | 198/200 | 3.7e-01 | 3.8e-02 | 35.3M (stopped) |

For comparison: design sampler at every layer (old guess, member 6) = 113M queries, failed at L5.
Log: `pool_logs/refine_snapshot_mlp9_track.log`. Run resumed at L6 with the new fallbacks:
`pool_logs/refine_snapshot_mlp9_L6.log` (in progress at time of writing).

## The two L6 blockers (teacher indexing; from `layer_members.py`, oracle diagnostic)
- **Teacher unit 191**: members 3/1/6 have it at cos 0.997/0.988/0.984 to the truth; the guess member (2) has
  cos −0.13. Missed because orphan clustering used cos 0.99 + support ≥ 2 → three singletons. Now clusters at 0.98 and
  allows support 1 (untested).
- **Teacher unit 172**: NO member learned it (best cos 0.14). Fires on 0.15% of N(0,1) inputs. Needs guess-free
  recovery (all other layer-≤6 units are exact, so its kinks are the unexplained ones). Concern: mapback goes through
  the depth-6 prefix Jacobian (cond ~1e13 at L5), so the guess-free row may be too coarse or rejected (`rel > 1e-3`).

## Open problems (where help is needed)
1. **Precision decays ~10–20× per layer with tracking** (L2 3e-10 → L5 7e-6). With an EXACT prefix, track at L3 gives
   1.6e-11 (step 2, need 400), so the per-layer loss is prefix error amplification by the track fit, not the fit itself.
   The design sampler only amplifies ~2× per layer (L2 6e-11 → L4 5.9e-9) but costs ~14M queries/layer versus
   ~3.5M for track + check. At this rate L8 ends up around 1e-3. Want: a cheap re-fit that does not amplify (e.g.
   re-solve layer l jointly with a correction of layer l−1, or a Gauss–Newton polish on all stored kink points).
2. **Units the committee never learned** (rare-firing, one-sided): need a guess-free path that works at depth.
3. **L1 cost**: 12M queries (60k/unit) because the input is 3072-wide; probably reducible.
4. Earlier attempts and their failure modes:
   - `lean_kink.py` / `lean_chain.py` (my own Jacobian-jump refiner + track): L1 fine, deep layers ill-conditioned; abandoned.
   - `kink_solve.polish_layer` as an exactness check: hangs/loops when kinks are rare; dropped.
   - Up-front committee row vote (replace guess rows the committee outvotes): made L3 worse (45 track failures); off.
   - Normalizing the guess's gauge before refining broke the L1 bias (refiner tolerances are gauge-dependent).

## Code changes (uncommitted)
- `refine_snapshot_mlp.py` (new): the driver above.
- `kink_solve.py`: `recover_layer(..., track_step=, track_need=)` kwargs passed to `track_layer` (MLP only; defaults unchanged).
- `peel/hybrid_sweep.py` (bug fix): the L1 bias re-measured at the plane's foot is accepted only if it matches the
  anchor's own estimate. Both "independent" regions share the same foot, so a foreign kink near it passed the
  agreement check: three L1 biases were 1e-3–3e-3 off with exact weights.
- `lean_kink.py`, `lean_chain.py`: superseded experiments.
- Diagnostics (scratchpad): `layer_members.py` (per-unit member cos vs teacher), `track_prec.py` (track precision vs
  step/need with an exact prefix).

## Ops notes
- GPUs 1 and 2 are busy with other jobs; use GPU 0. Runs are not bit-reproducible.
- Do NOT `pkill -f <pattern>` from a shell whose command line contains the pattern (it kills the shell). Kill by PID:
  `ps -eo pid,comm,args | awk '$2=="python" && /refine_snapshot_mlp/ {print $1}'`.
