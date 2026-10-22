# LeNet implementation audit and repairs, 2026-10-21

**Subsequent result:** direct output-only recovery completed C5 120/120, F6 84/84, and a fitted head. See [the recovery report](RECOVERY_lenet_2026-10-21.md). The integration smoke-test limits below describe the earlier audit stage.

This audit identifies implementation failures in the LeNet path and repairs them. It does not infer that C5 intrinsically requires known downstream layers. No MLP experiments were rerun. The earlier fixed-dataset oracle-suffix experiments do not establish the cause of the production stall.

## 1. Boundary inversion uses the wrong member coordinates — reproduced and fixed

**Code:** `_cnn_frontier_boundary_queries`, `_spatial_boundary_normals`, `_cnn_layer_mine` in `method.py`.

Boundary inversion computes frontier features using member 0, then selects raw frontier filters from every member. Pinned prefixes can retain different positive channel scales because rows are injected at each member's own magnitude and warm rebuilding preserves those scales. Consequently a raw filter from member j does not represent its hyperplane in member 0's input coordinates.

For aligned exact prefixes, let `f_j` be the prefix channel scales removed by canonicalization. Then `h_j = h_0 * (f_j / f_0)`. A member-j filter applied to `h_0` must multiply its input columns by `f_j / f_0`. Conversely, a tail receiving reference features must first convert those features into its own input gauge. Pooling preserves these channel-scale relations.

**Reproduction:** eight copies of the actual LeNet teacher, differing only by function-preserving prefix scalings; their fp64 output differences are below 1e-14. Run the production C5 boundary sampler, 960 queries and 250 inversion steps, and evaluate every requested member/neuron's actual preactivation:

| | Member 0 | Members 1–7 |
|---|---:|---:|
| Before: median absolute target preactivation | 7.9e-7 | 0.325–0.565 |
| Before: fraction within 1e-3 of target boundary | 100% | 0–0.83% |
| After: median absolute target preactivation | 7.8e-7 | 6.8e-7–9.3e-7 |
| After: fraction within 1e-3 | 100% | **100% for every member** |

**Repair:** map dense and spatial boundary filters into the inversion prefix's coordinates. Layer-space mining now maps reference features into each member's coordinates too. Its existing oracle-only teacher branch aligns a teacher copy before coordinate conversion; no teacher weights are provided to honest learners.

**Scope:** this bug affects populations with different prefix gauges, particularly warm peeling. A cold resume with identical shared prefix scales need not trigger it. The reproduction establishes an actual sampling failure, not that this single bug explains every historical plateau.

Artifacts: `probe_c5_boundary_gauge.py`, `c5_boundary_gauge_before_fix.json`, `c5_boundary_gauge_after_fix.json`. Algebraic checks cover spatial conv2, C5, and the conv-to-F6 transition; errors are at most about 2e-15 in fp64 (`c5_fix_validation.json`).

## 2. CNN partial-row restoration could freeze random rows — reproduced and fixed

**Code:** `_repin_partial` in `method.py`.

The rebuild helper installed gradient masks for CNN partial rows but restored the stored coefficients only for non-CNN networks. A newly rebuilt CNN whose frontier rows were not already included in `frozen` could therefore mark random frontier rows as exact and stop their gradients.

**Reproduction:** build a fresh LeNet population with the verified conv1/conv2 prefix, then invoke the production restoration helper with the actual 103-row C5 record. Before repair, both members report 103 live frozen rows, but normalized pinned-filter errors are **0.365 and 0.319**. After repair they are **1.36e-8 and 1.74e-8** (fp32 training copies).

**Repair:** restore CNN rows through the existing gauge-aware row-transfer routine before marking their gradient masks live, just as the helper already did for other networks.

**Scope:** if a warm rebuild already preserved the correct rows, the omission need not damage that call. I have not established that a specific historical stalled run took a rebuild path with missing recorded rows. This is a verified helper-contract failure, not evidence that all historical records are corrupt.

Artifacts: `probe_cnn_repin.py`, `cnn_repin_before_fix.json`, `cnn_repin_probe.json`.

## 3. Residual Jacobians used a rounded prefix as if exact — reproduced and fixed

**Code:** residual-seed integration in `method.py`; rank selection in `residual_seed.py`.

The pipeline cloned a live fp32 member and promoted it to fp64 before finite differences. Promotion preserves the already-rounded prefix coefficients. That contaminates the map from image-space finite differences into C5 coordinates and the projection off known rows.

**Reproduction against the actual record:** after diagnostically promoting one missing row into the known set, 16 remain. The old rounded input produces raw threshold rank **65**, with the first unwanted singular value **1.06e-4**. The repaired production record view produces rank **16**, with its first unwanted singular value **3.97e-8**. This reproduces a mechanism for the historical best run's rank 176–179 despite only 14 missing rows.

**Repair:** construct the finite-difference model from the persistent fp64 record. For row redraws, restore the member's input scales on that precise record using function-preserving gauge moves. An impossible rank above the missing width now abstains instead of inserting noise directions into the seed sampler. The guard is defensive; it is not a rank-truncation substitute for correcting the prefix.

The residual module's description now says its span contains projected missing rows and its rank is bounded by missing width under the exact-prefix assumptions; it no longer claims an unconditional completeness certificate.

Artifacts: `validate_c5_fixes.py`, `c5_fix_validation.json`. The precise member-gauge prefix check differs by about 2e-15; the rounded-input guard returns no directions while preserving its raw rank for diagnosis.

## 4. Snapshot resume ignores the frozen bank — reproduced and fixed

**Code:** `run.py` CNN `--resume-record` loader and new `_cnn_checkpoint_record` in `method.py`.

The writer stores resumed rows in `frozen` and new rows in `partial_state`/`partial_mask`. The loader previously looked only at the latter. The real iteration-3 smoke-test snapshot contains **6/16/103** exact rows in its frozen bank, but the old loader sees only `{2: 0}`. Because this partial-mask dictionary exists, it can even take the nominal resume-record branch and then establish no solved prefix.

**Repair:** combine frozen and partial masks and coefficients in a canonical fp64 record before passing them into `reconstruct_cnn`. Explicit `--resume-record` now fails clearly if there are no recorded exact rows, rather than silently starting an unrelated reconstruction.

The actual snapshot round trip restores **6,16,103** rows with canonical teacher errors **5.24e-14, 5.88e-13, 3.10e-10**. Tests also cover frozen-only, combined, original-record, and absent-record cases. An actual CLI resume from the saved snapshot reports the same counts without rerunning the kink solver. This is a continuing implementation cause of the record-format issue described as I11 in the supplied audit; rebuilding one record file had not repaired the general loader.

Artifacts: `validate_cnn_checkpoint_resume.py`, `cnn_checkpoint_resume_validation.json`, `production_c5_resume_codefix_20261021_01.log`.

## 5. The later boundary-sampling intervention bypasses disagreement — measured

Using the actual `reconstruct_cnn` path with training/refinement disabled, the stage trace confirms **L1 calls disagreement; L2 attempts spatial boundaries, gets no seeds under the default spatial flag, and calls disagreement; C5 returns dense boundary seeds and skips disagreement**. This is a later intervention introduced after disagreement sampling had already failed. It does not explain the original layer gap. On the attributable 103-row C5 record:

- Default boundary path: **zero `gen_queries` calls**, one boundary-sampler call.
- Of 960 targets, **824 target already-pinned rows**, and only **136 target unpinned rows** (85.8% versus 14.2%).
- With `layer_space_init=False`: one actual disagreement-sampler call.

This is the honest path too, not just the documented inert-cheat issue. It changes the procedure at the C5 frontier. It is a deliberate existing policy, so I did not silently replace it or claim that removing it solves C5. The production log now explicitly prints `boundary-only (no disagreement ascent)`, `image-disagreement`, or `layer-disagreement`. The existing `--no-layer-space-init` selects actual image disagreement.

Artifacts: `trace_c5_sampler.py`, `c5_sampler_trace.json`, and its two production-call logs.

## Verification and changed files

Recovery changes are confined to `method.py`, `residual_seed.py`, and the CNN resume loader in `run.py`; pre-existing edits in those and other files were preserved. Only the audit's changes are isolated in `lenet_method_fixes.patch`, `lenet_residual_fixes.patch`, and `lenet_run_fixes.patch`. No commit was made.

Focused checks use the actual LeNet architecture, teacher, and attributable record. They test function invariance, requested boundary placement, recorded-row restoration, residual rank, and input-scale conversions. They do not copy exact F6/head into an honest learner. The synthetic teacher-row promotion in the numerical unit test is an explicit test oracle.

The integration smoke test uses the **actual `run.py` CLI and `mergedbest` preset (batch 128)**, the real eight-member committee, unknown trainable F6/head, fresh adaptive disagreement queries, local refinement, candidate verification, and corrected residual seeds. It is deliberately only three iterations at 4,000 sampler queries each: a bounded integration check, not a claim about convergence or final recovery. Its short-budget learning-rate schedule is not comparable to the historical overnight budgets. Checkpoints use a unique tag with `POP_SAVE_ALL=1`; no shared-tag snapshot is read or overwritten.

The production output is in `production_c5_codefix_20261021_01.log`. It completed three iterations, kept the 103 C5 pins, and reported rank 17 for 17 missing rows at every residual rebuild; no new pins were found in this short smoke test. The independent CLI resume check completed too. The original teacher and both input recovery records are untouched.

**Limit:** these repairs remove reproduced implementation failures. They do not establish a complete explanation for every prior C5 plateau or demonstrate 120/120 exact recovery. In particular, the boundary-coordinate bug need not affect cold-resumed, shared-gauge populations or runs that explicitly disabled boundary sampling.
