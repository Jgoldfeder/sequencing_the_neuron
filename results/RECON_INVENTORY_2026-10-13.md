# recon/ inventory, 2026-10-13 (nothing deleted; stock-take only)

**UPDATE 2026-10-13 (later):** the X/Y query buffers were STRIPPED from all 52 raw checkpoints >= 1 GB (recon/ and code_parallel/recon/) at Judah's request: 1091 GB freed, every file keeps best_state / pop_states / teacher_state / cfg / seed / queries (+ a `buffers_stripped` record with the old X/Y shapes). Sizes listed below are the pre-strip sizes. Disk: 1.2 TB free after.

recon/ = 910 files, 1.27 TB. 47 files >= 1 GB hold 1.24 TB; the other 863 files hold 25 GB.
Disk: 1.8 TB, 100% full (267 MB free). Other big consumers: code_parallel/recon 60 GB (Aug-5 copy of the
code dir; its 33 GB _fast__12288x1024x200 file is byte-identical in size to recon/'s), spatial_neurons 105 GB.

## What a big file is
Every big file is the raw end-of-run checkpoint written by method.reconstruct: best_state + pop_states +
teacher_state (~34 MB together) + the FULL query ring buffer X,Y (window x q rows x 3072 fp32 + Y).
The weights are duplicated in the small files: _pop__<run>.pt (8 member states + teacher) and
<run>_final.pt (best member + final metrics). So the only content unique to a big file is the query buffer,
which is needed only to replay/polish offline (polish.py) without re-querying.

## Big files (>= 1 GB), by category
### A. Corrupt / unloadable (213 GB) - torch.load fails: zip central directory missing (killed mid-write)
- 193.3 GB  mergedbest512_tanh_one-epochs__3072x200^9x100  (10-13; disk filled during save; weights ARE in _pop__ one-epochs)
-  20.0 GB  mergedbest512_dirlr3__3072x200^5x100            (10-12; weights ARE in _pop__ dirlr3)

### B. Tanh deep-net runs, Oct 2026 (639 GB) - all loadable, all have json + _final + _pop__
- 228.4 GB  mergedbest512_tanh__3072x200^9x100  (10-12) = the UNTAGGED batch-4096 run; it overwrote the batch-512
            baseline's files (baseline survives only as results/RECORD_tanh_11layer_batch512_q900k_s0.md)
- 203.0 GB  mergedbest512_tanh__3072x200^7x100  (10-08) = 9-layer run (9 widths), q 800k w20
- 114.2 GB  mergedbest512_tanh__3072x200^6x100  (10-06) = 8-layer run, q 300k w30 (the one solved to 8e-5 post hoc)
-  36.6 GB  mergedbest512_tanh__3072x200^5x100  (10-06) = 7-layer, q 96k w30 (FAILED: loss 0.72, agree 0.43)
-  36.6 GB  mergedbest512_tanh__3072x200^4x100  (10-06) = 6-layer, q 96k w30 (solved)
-  18.3 GB  mergedbest512_tanh__3072x200^3x100  (10-05) = 5-layer
Also: _pop__ snapshots only (no raw file, no json) for the 10-layer 3072x200^8x100 run (killed at it 25; used for the post-hoc L1 pipeline), tanh more-epochs, less-epochs, one-epochs, dirlr3.
(Layer counts here = number of widths, e.g. 3072x200^9x100 = 11-layer.)

### C. Cheat-mode leaky-ReLU scaling series, Sep 14-23 (180 GB) - all loadable, json + pop present
- 51.8 GB 3072x512^4x100, 36.6 GB 3072x512^3x100, 36.6 GB 3072x512^2x100, 18.4 GB 3072x3072x100,
  9.2 GB 3072x2048x100, 4.6 GB 3072x1024x100, 2.3 GB 3072x512x100, 1.2 GB 3072x256x100, 2.0 GB 3072x64x64x100,
  3.5 GB x5 (200x200^k x100 seeds 0-3 + cheat_design), 1.8 GB 40x40x40x40x100, 1.8 + 1.1 GB anneal/stage0 smokes.

### D. ReLU 5-layer / 6-layer experiments, Sep 30 - Oct 6 (155 GB) - loadable, json present
- 18.3 GB each: snapB (plus 185 per-iteration pop snapshots, 5.7 GB), revboth, seed1 (x2 archs), untagged
  3072x200^3 and 3072x200^4, tanh 3072x200^3
- 6.1 GB each: target_control, target_pin, target_exact5, target_pin5 (the target-unsolved NEGATIVE result)
- 4.9 GB snap, 4.3 GB abstk / abseq (stacked-vs-sequential A/B), 1.9 GB 3072x200^6 w0, 20 GB dirlr3 (corrupt, see A)

### E. Pre-September / one-off (56 GB)
- 33.4 GB _fast__12288x1024x200 (08-05; keys dims,iter,pop_states,teacher_state only; DUPLICATE of code_parallel/recon's 33 GB file)
- 10.5 GB mergedbest512__784x128x80x40x32x16x10 (09-24)
- 6.1 + 1.5 + 1.2 GB v18_lbfgs 784x512x10 / iter15 / 3072x128x100 (08-02), 1.5 GB iter15_true.pt (08-02)
- 1.4 GB mergedbest512__12288x2048x200 (09-17), 1.2 GB v18_lbfgs_cheat 3072x256x100 (09-14)

## Small files (< 1 GB, 25 GB total)
- 625 _pop__ files, 17.6 GB: ~470 are per-iteration snapshots (POP_SAVE_ALL) of snapB (185), seed1/revboth (30 each),
  target_* variants (5-10 each), snap (43), revpeel (13); the rest are one end-of-run pop snapshot per run.
- 2.2 GB of non-run artifacts: the L1 isolation/scale/dual/refine products for the 8/9/10-layer tanh nets
  (_iso_l1_*, _scale_l1_*, _l1fixed_*, _polished_all, _dual_L2/L3_*, _best_scalefit, _tanhsolver, _consmean),
  consensus nets, *_final.pt.bak, Aug-3 diag logs, solved_units.pt / recovered_solved.pt / precision_solved.
- ~70 smoke-test runs (784x16x12x10, 784x32x10, 784x128x64x10, LeNet/CNN smokes, cnn dbg*), each < 0.3 GB.

## Which big files are NOT referenced by any live result
- A (corrupt): one-epochs raw, dirlr3 raw.
- Duplicate: _fast__12288x1024x200 exists twice (recon/ and code_parallel/recon/).
- Everything else loads and has a results json; whether it is still needed depends only on whether you
  want to replay its query buffer offline. Weights are preserved in _pop__ / _final regardless.
