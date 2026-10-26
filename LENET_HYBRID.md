# Leaky-ReLU LeNet: one recovery command

`lenet_hybrid.py` connects the original eight-member committee to direct C5/F6 recovery, output-only sign selection, head fitting, and held-out output validation. It is for the existing **28×28 leaky-ReLU LeNet**, not the separate 1998 tanh LeNet-5 implementation.

Run from this code directory using the existing Python environment (PyTorch, SciPy, and the repository's normal dependencies). A GPU with fp64 support is recommended. The default teacher is the same saved teacher used in the LeNet audit; the script does not train or download a new teacher.

## Fresh start

```bash
./run_lenet_end_to_end.sh --device cuda:0 --output runs/lenet_fresh
```

This starts from randomly initialized students: **C1/C2/partial-C5 committee recovery → automatic handoff → remaining C5 → F6 → head → validation**, with no recovered record supplied. The launcher runs `lenet_hybrid.py`, whose default also starts with the first recovery phase.

The default `--committee-recipe historical` follows the two recorded first-phase stages:

1. C1/C2: eight-member `mergedbest`, seed 0, q=20000, batch 128, row balancing, directional LR 0.03 with decay 0.85, kink refinement and partial pinning, layer refresh, and five retries. Default budget: 60 iterations per attempt. The directional settings are recorded in `pool_logs/lenetpeel_fixed.log`.
2. C5: restart from the C1/C2 record produced by this run, using eight-member **teacher-gradient disagreement** (`cheat=True`, `cheat_bb_ref=True`) and epsilon 0.02, as in `pool_logs/lenet_c5_cheat.log`, which produced the 103-row source record. Default budget: 40 iterations per attempt. This stage uses teacher gradients; it must not be described as an output-only first phase. No historical recovered prefix is loaded by default.

These settings restore the documented method and the logged directional settings; the original snapshots do not contain full configuration or optimizer/RNG states, so this is not a claim of bit-for-bit historical replay. `--committee-recipe ordinary` explicitly selects the previous ordinary-disagreement variant.

`--handoff-c5` defaults to 103 and is an **early handoff target, not an eligibility gate**. If the committee ends below it, the driver still attempts direct C5 recovery with the best saved record, provided C1/C2 are complete. It only reports incomplete after that attempt fails its actual numerical checks or budget. An incomplete C1/C2 prefix remains unsupported by the direct C5 engine.

Use `--committee-outer`, `--committee-c5-outer`, `--committee-q`, `--committee-epochs`, `--committee-retries`, and `--committee-seed` to change budgets. The budget-limited first phase and direct phase can still fail; completion is reported only after independent output validation.

## Use the recovered prefix we already have

```bash
python -u lenet_hybrid.py \
  --record recon/_record_lenet_c1c2_c5_105.pt \
  --device cuda:0 --output runs/lenet_from_record
```

Despite the filename, this input contains 103 C5 rows. This command does **not** load the completed model or any other audit intermediates. It discovers and verifies the remaining C5 and F6 rows itself. The signed C1/C2 prefix is required for this mode; unsigned historical search banks are rejected.

## Interrupt and resume

```bash
python -u lenet_hybrid.py --continue-run runs/lenet_from_record --device cuda:0
```

Interrupt with Ctrl-C. The direct stage writes an atomic `state.pt` containing masks, unsigned-row state, anchors, pending candidates, random-generator state, and query counts. Interrupted committee runs retain their iteration snapshots; the same `--continue-run` resumes committee training if handoff has not occurred. Each committee segment uses a distinct snapshot prefix, and the driver terminates only its own worker/process group at handoff. Native iteration numbers can restart during refresh/retry, so their iteration-named files can be overwritten within the same segment; logs remain the full run history. The handoff chooses the best readable certified prefix, not merely the most recent file.

To try direct recovery immediately from a committee run below its target:

```bash
python -u lenet_hybrid.py --continue-run runs/lenet_fresh --handoff-now --device cuda:0
```

Stop the active driver first with Ctrl-C; do not run two writers against one output directory.

Use a **new** output directory for a new run; existing directories are rejected. Resume checks teacher identity. `--max-rounds` (default 12 per direct frontier) and `--lines` (default 32 per round) bound direct discovery. An exhausted budget leaves a resume checkpoint. You can increase `--max-rounds` when continuing.

## Outputs

- `complete.pt`: written only after every hidden row is signed, the head is fitted, and fresh output validation passes.
- `validation.json`: output errors at six input scales, actual direct-phase query count, and completion status.
- `state.pt`, `progress.json`: resumable state and progress, including failures or interruptions.
- `committee.log`, `committee_*_it*.pt`: original committee logs/snapshots for fresh-start runs.
- `failure.json`: explicit incomplete/error information. This is historical; a resumed run may subsequently complete, so consult `validation.json` and the latest `progress.json`.

The head uses 4096 fit labels. Validation uses **new** samples at scales 0.25, 0.5, 1, 1.5, 3, and 6 (1024 inputs each by default). `--output-tolerance` defaults to `1e-6`. Passing output validation is a functional check; it is not a claim of bit-exact parameter equality. The oracle contract is fp64 forward outputs, as in the existing kink solver.

The direct engine receives only an output-query adapter. Unknown teacher weights are not used for direct search, fitting, acceptance, or signs. The historical C5 committee recipe uses teacher gradients for query selection, as explicitly recorded in `phase1_recipe.json`; `ordinary` does not. Both recipes retain existing parameter-error diagnostics. No diagnostic teacher coefficients are used as direct-phase guesses or pins.

```python
import torch
from nets import ConvNet

ck = torch.load('runs/lenet_from_record/complete.pt', map_location='cpu', weights_only=False)
model = ConvNet(ck['input_shape'], ck['conv_cfgs'], ck['fc_dims'],
                ck['out_dim'], ck['act']).double()
model.load_state_dict(ck['state_dict'])
model.eval()
```

Inspect the plan without allocating a GPU or creating output files:

```bash
python lenet_hybrid.py --plan
```

CPU regression tests:

```bash
python tests/test_lenet_hybrid.py
```

Implementation checks cover positive-gauge invariance on import, exact recovery-state/RNG resume, rejection of incomplete prefixes, and cleanup of temporary numerical-locator overrides. Integration validation results are recorded in `audit_lenet/hybrid_driver_validation.json` when available.

The unified driver was run from the original 103-row record through C5=120, F6=84, and the head, including real interruption/resume. It passed 3072 fresh output checks at six scales with maximum absolute error **5.9786e-9**. The checkpoint and logs are in `audit_lenet/hybrid_driver_20261021/`; the machine-readable test report is `audit_lenet/hybrid_driver_validation.json`. This validates the record-to-completion path; fresh committee convergence was not demonstrated in this integration test.
