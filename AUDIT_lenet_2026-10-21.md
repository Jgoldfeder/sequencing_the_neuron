# LeNet black-box recovery: findings, logs and issues (audit, 2026-10-21)

Code directory: `/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code`.
All log paths are relative to it: logs in `pool_logs/`, snapshots and records in `recon/`.

## 1. Bottom line

- **Not solved.** The best state is conv1 6/6, conv2 16/16 and C5 106/120 exact, with F6 (84) and the head (10) unpinned.
- **Pinned rows are exact.** Every pinned row checks against the teacher at max 3.1e-10 in fp64.
- **The C5 ceiling is the same in every committee run.** Honest, cheat, q 20k, q 200k, residual and focused runs all stopped between 100 and 106 C5 rows.
- **Cause, measured.** 16 of the teacher's 120 C5 channels never fire on Gaussian queries, and 9 of those never fire on MNIST either. No member learns a channel that never fires, so members never disagree about it, so the sampler never goes near its kink.
- **What fixed reachability.** Residual-Jacobian seeds (honest, black-box) cut the silent C5 channels from 16 to between 0 and 1 per batch.
- **What did not follow.** The committee still learns those rows too slowly for members to agree on them: +3 pins in 11 hours.

## 2. Setup

- **Teacher:** LeNet-5 on MNIST, leaky ReLU (slope 0.01), avg-pool, 25 epochs, seed 0. Layers are conv1 6, conv2 16, C5 120 (5x5 conv acting as a dense layer), F6 84, head 10. File: `teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt` (git-tracked).
- **Method:** 8-member committee, disagreement sampler, fast-peel-partial pinning, kink refiner (`--loc-refine`, kink_solve), global kink certificate (`--verify-cands`), dedupe, row-norm balancing.
- **Records used to resume:**
  - `recon/_record_lenet_c1c2_exact.pt`: conv1 and conv2 exact, C5 empty. Built from `_pop__mergedbest_lenetpeel_fixed_..._it020.pt`.
  - `recon/_record_lenet_c1c2_c5_105.pt`: conv1, conv2 and **103** C5 rows. Built from the cheat arm's iteration-25 snapshot. The name says 105, but the snapshot was taken before the last two pins.

## 3. Findings, in order

| # | Finding | Evidence |
|---|---|---|
| F1 | conv1 and conv2 are solved by the committee plus peel, exactly (4e-14 / 6e-13). | `lenetpeel_fixed.log`, `lenet_q200k*.log` |
| F2 | The C5 ceiling is 100–102 at q 20k. | `lenetpeel_fixed.log` (100), `lenetpeel_dirlr2.log` (97), `lenetpeel_base.log` (88) |
| F3 | q 200k from scratch is worse per query and per hour than q 20k with pins. It plateaus at C5 about 40 consensus after 12 h. | `lenet_nopeel_q200k_c.log` |
| F4 | Pinning versus no pinning at q 200k ties on C5 at matched iterations. Early differences were noise. | `lenet_q200k*.log` vs `lenet_nopeel_q200k*.log` |
| F5 | The ceiling is caused by silent teacher channels: 16/120 on Gaussian queries, 10/120 on MNIST, 9 on both. All 24 near-silent channels can be pushed to +8 to +26 inside the box \|x\|∞ ≤ 1. | memory note `lenet-c5-silent-channels.md`; script in scratchpad |
| F6 | Pooling roughly triples the silent fraction: about 11% with pooling, 4% for stride-2 without pooling, 3 seeds each. Silent units also appear in deep MLPs (9-layer L8: 16/200). | same memory note |
| F7 | The residual committee from the 100-row record (eps 0.1) pinned 1 row in 2M queries. | `lenet_residual_q200k.log` |
| F8 | Consensus eps must be 0.02 at C5. At eps 0.1 the refiner gets guesses 40–80° off, and the 12° angle gate rejects genuine hyperplanes. | `lenet_c5_control_eps01.log`, `lenet_c5_cheat_eps01.log` |
| F9 | Cheat disagreement (teacher gradients) from the exact conv record reaches C5 104–105 by iteration 20 and then stays flat. It recovered 0 of 16 silent channels (oracle check of which channels were pinned). | `lenet_c5_cheat.log` |
| F10 | Oracle ascent tests: a reference with the open rows zeroed reaches 6/17 silent channels; open rows as random rows with bias −5 reach 17/17 in three rounds. "Reached" means one firing point in 512, with teacher rows used as the stand-in pin. Neither variant gained anything inside the pipeline sampler. | `lenet_c5_cheatfocus_zero.log`, `lenet_c5_cheatfocus_deadrow.log` (104) |
| F11 | Keep-alive (stopping members' unpinned units from dying) is negative: 0 pins, and 13–14 silent channels stay silent on every mined batch. | `lenet_c5_keepalive105.log` |
| F12 | **Residual-Jacobian seeds (honest).** Rank of the projected FD Jacobian equals the number of missing rows (17 for 17; singular values 2.4e-2 then 3e-5), for 32k queries. Seeds along that span take the silent C5 channels from 16 to 0–1 per batch. | `lenet_c5_resseed_noreinit.log`, `[oracle-excite]` lines |
| F13 | With seeds the committee pins 103 → 106 in 51 iterations (about 11 h, 1M queries), then stays flat. The best member reaches cosine 0.92–0.99 on about 7/17 open rows by iteration 20, but members never agree within 0.02. | `lenet_c5_resseed_noreinit.log` (106) |
| F14 | Redrawing the open rows inside the residual span: 104 (no gain). | `lenet_c5_resseed.log` |
| F15 | Refining the best member's non-consensus rows (`--solo-cands`, angle gate 30–45°): 0–1 of 16 per pass, with rejected angles at a median of about 81°. Each pass costs about 15 min and 13M queries. | `lenet_c5_solo.log`, `lenet_c5_solo80.log` (104) |
| F16 | The from-scratch run stays stuck at conv2 15/16 for over 50 iterations, the same conv2 straggler seen in earlier runs. | `lenet_full_resseed_conv2stall.log` |

## 4. Run index, 2026-10-18 to 10-21

Earlier runs (10-14 to 10-16) are the `lenetpeel_*.log` and `lenet_residual*.log` files. RESULTS_deep_committee_2026-10-16.md covers them.

| Log | What | Final |
|---|---|---|
| `lenet_q200k.log`, `lenet_q200k_oom_it5.log` | pinning, q 200k, from scratch | OOM at iteration 6 of the conv2 stage (bug I1) |
| `lenet_q200k_b.log` | pinning, q 200k, relaunch | OOM again at the same point (I1) |
| `lenet_q200k_b_dup_junk.log` | two copies writing to one log | junk (I4) |
| `lenet_nopeel_q200k.log`, `lenet_nopeel_q200k_oom_it5.log`, `lenet_nopeel_q200k_b.log` | no pinning, q 200k | OOM (I1, and I2 for the shared GPU) |
| `lenet_nopeel_q200k_c.log` | no pinning, q 200k, fixed sampler | 17 iterations, C5 about 40 consensus, killed |
| `lenet_residual_q200k.log` | residual committee from the 100-row record | 101/120, killed |
| `memprobe_pin.log` | memory probe used to find bug I1 | diagnostic |
| `lenet_c5_cheat_inert.log` | cheat with boundary seeds on | inert: cheat never ran (I5) |
| `lenet_c5_cheat_oom.log` | cheat arm | OOM sharing a GPU |
| `lenet_c5_control_eps01.log`, `lenet_c5_cheat_eps01.log` | eps 0.1 arms | angle-gate rejects (F8), restarted |
| `lenet_c5_nolsi_dropped.log`, `lenet_c5_control_redundant.log` | dropped arms | not meaningful |
| `lenet_c5_cheat.log` | **cheat, eps 0.02, from c1c2 record** | **104–105**, source of the 103-row record |
| `lenet_c5_keepalive_gpu2_abandoned.log`, `lenet_c5_keepalive_fromzero.log` | keep-alive, from zero C5 | wrong starting point, stopped |
| `lenet_c5_keepalive105_v1.log`, `lenet_c5_keepalive105.log` | keep-alive from 103 (the second with diagnostic) | 103, negative (F11) |
| `lenet_c5_cheatfocus_v1_maskbug.log` | focused cheat, mask bug (I6) | invalid |
| `lenet_c5_cheatfocus_zero.log` | focused cheat, zeroed reference | stopped early |
| `lenet_c5_cheatfocus_deadrow.log` | focused cheat, dead-row reference | 104 after 31 iterations (ran 11 h after a failed kill, I3) |
| `lenet_c5_resseed_noreinit.log` | **residual seeds 0.5** | **106**, best result |
| `lenet_c5_resseed.log` | residual seeds + redraw | 104 |
| `lenet_c5_resseed_q1_noconsensus.log` | quorum one-eighth attempt | 0 candidates (the code floors the quorum at 2) |
| `lenet_c5_solo_everyiter.log` | solo refine every iteration | too slow, stopped |
| `lenet_c5_solo.log`, `lenet_c5_solo80.log` | solo refine every third iteration, seeds 0.5 / 0.8 | 104 / 104 |
| `lenet_full_resseed_crash_conv2.log` | from scratch with seeds | crash at the conv2 frontier (I7) |
| `lenet_full_resseed_conv2stall.log` | from scratch, fixed | conv2 15/16 forever (F16) |
| `c5_residual_kink.log` | direct residual kink search (c5_residual_kink.py) | unfinished, abandoned |

Snapshots: `recon/_pop__mergedbest[_cheat]_<tag>_cnn__1x28x28__...__s0.pt` hold the best member, partial_mask and partial_state. Tag collisions are listed under I3.

## 5. Code added or changed (uncommitted unless noted)

- **`residual_seed.py` (new).** Attacker-side prefix, residual subspace, seeds, `reinit_unpinned_`.
- **`oracle_focus.py` (new, cheat only).** Teacher-to-member channel map, `MaskedTeacher`, `DeadRowTeacher`, `Padded`.
- **`c5_residual_kink.py` (new, unfinished).** Directed kink search along the residual directions.
- **`method.py` / `run.py` flags:**
  - honest: `--residual-seeds/-smax/-every/-reinit`, `--solo-cands/-start/-every`, `--keep-alive`;
  - cheat: `--cheat-focus`;
  - diagnostic, uses the teacher: `--oracle-excite`.
- **`method.py` other changes:**
  - sampler chunk fix for ConvNets (`_per_sample_hidden`);
  - `empty_cache` before the CNN sampler;
  - `[sampler]` size log;
  - per-layer eps in `_cnn_consensus`.
  The chunk fix and size log were committed (91a491f and earlier).
- **`qg_tune.py --conv lenet`:** committed (b5fb923).
- **`cnn_balance.keep_alive_`:** added in cnn_balance.py.

## 6. Issues to audit

| ID | Issue | Effect | Status |
|---|---|---|---|
| I1 | `gen_queries` chunk estimate used `dims`, which ConvNet lacks. It counted 10 hidden floats instead of 6508, so chunk = q and memory hit 7–19 GB. | Four q 200k LeNet runs died at the first sampler call | fixed, verified (3.3 GB) |
| I2 | Runs were placed on GPUs already in use (mine or the user's) while a GPU was free. | OOMs, slower runs | process issue |
| I3 | **Kills hit the nohup wrapper or the wrong PID.** Four runs I reported as killed kept running 8–11 h, writing into renamed logs (`cheatfocus_deadrow`, `resseed_noreinit`, `full_resseed_conv2stall`) and sharing snapshot tags. **The snapshots `_pop__..._lenet_c5_resseed_...`, `..._cheat_lenet_c5_cheatfocus_...` and `..._lenet_full_resseed_...` were overwritten by more than one run and are not attributable.** | Results from logs are valid; those snapshots are not | process issue; kill by the python PID with ppid 1 |
| I4 | A relaunch ran twice into one log for 6 minutes. | `lenet_q200k_b_dup_junk.log` | discarded |
| I5 | With a frozen prefix the CNN path uses frontier boundary seeds and never calls `gen_queries`, so `--cheat` does nothing without `--no-layer-space-init`. | First cheat arm inert | documented |
| I6 | `--cheat-focus` read the pinned set from `partial_mask` (empty after a resume) and required cosine > 0.999; the real value is about 0.993 because of balancing gauge. | All 120 channels were masked | fixed: union of pinned sources, threshold 0.9 |
| I7 | Residual seeds were applied at the conv2 frontier, where rows are patches (shape mismatch). | Crash | fixed: only when rows span the whole frontier input |
| I8 | `--cluster-quorum` is a fraction of members, floored at 2 inside the CNN loop. | A one-member quorum cannot be expressed | open |
| I9 | The oracle claim was overstated. I said "focused objective 16/16". In fact the zero-mask reference reached 4–6, and 16–17 came only from the dead-row reference. "Reached" meant one firing point and the pin was simulated with teacher rows. | Misreported to the user, corrected in conversation | corrected |
| I10 | The 8-hour autonomous window went unmonitored: the session stalled about 8 h after the last command. | No adjustments made | process issue |
| I11 | `_record_lenet_c1c2_c5_105.pt` holds 103 C5 rows despite its name. The first resume fell back to a fresh start because conv1/conv2 sat in `frozen`, not `partial_mask`. | One wasted launch; record rebuilt | fixed in the file, misleading name |
| I12 | `--oracle-excite` and the cheat flags read teacher internals. | Any honest claim must exclude them | by design, flagged in the help text |

## 7. Open questions

1. Why members fail to agree on rows that the seeds have made active on 2–10% of points. Candidate causes are the data weighting, the balancing norm projection, and the per-member assignment ambiguity among 17 random rows.
2. Whether up-weighting the seeded points in training speeds the learning of open rows. Not tested.
3. Whether reading rows directly off rank-1 residual-Jacobian jumps at seed points, which is closer to the guess-free route, is acceptable as the completion step.
4. The conv2 straggler in from-scratch runs (15/16). It has appeared in every from-scratch run since 10-15 and was never diagnosed.
5. F6 and the head have never been attempted from an exact C5 record.
