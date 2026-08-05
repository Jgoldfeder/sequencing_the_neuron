# code_parallel — multi-GPU variant

Self-contained copy of the reconstruction code with **single-run population
parallelism** across GPUs. The serial code at the repo root is unchanged; this
directory is a drop-in parallel variant that takes the same command lines plus a
few extra flags.

## What is parallelized
- **Training (the win)** — the population of `--p` students is split across the
  GPUs given by `--gpus`: member `i` lives on GPU `i % ng` and trains there for all
  epochs on a full local copy of the query buffer, with **zero cross-GPU
  communication** (the members are independent — nothing to synchronize). One
  persistent worker *process* per GPU (not a thread), so the tiny per-step kernels
  don't contend on the GIL. This is the axis the problem is trivially parallel on:
  `p` independent trainings → up to `ng`× faster. See `parallel_pool.py`
  (`WorkerPool`, `reconstruct_mp`) and `parallel_bench.py` for the microbenchmark.
- **Query generation** — runs on the master GPU with the full (gathered) committee.
  The batch is small (`q ≈ 1500`), so replicating the committee onto every GPU each
  call costs ~2× more than it saves; the committee is already gathered to master
  for consensus anyway.
- **Endgame solve** — both the fast fp32 staged solve and the `--polish` float64
  last-mile shard the query set across GPUs and sum per-shard gradients each LBFGS
  closure (exact same math as single-GPU), giving ~Nx on the forward/backward
  (`solver_polish_parallel_`, one closure for both dtypes).

## Extra flags (vs the serial code)
- `--gpus 0,1,2` — GPU indices; members split across them (routes to `reconstruct_mp`).
  `--gpus all` uses every visible CUDA device.
- `--polish` — replace the fast fp32 consensus solve with the full float64
  endgame (push_precision recipe: re-query teacher in float64, staged MSE→MAE
  LBFGS). Auto data-parallel when `--gpus` has >1 device.
- `--endgame-f64` — float64 endgame precision (else fast float32-on-GPU).
- `--pop-save-every N` — snapshot the population every N iters.

Every other flag (`--fast`, `--combine`, `--window`, …) behaves exactly as in the
serial code.

## Example
```
python run.py --variant v18_lbfgs --arch 3072,256,100 --outer 60 --q 1500 \
    --window 60 --gpus 0,1,2 --p 9 --combine --fast --polish
```

## Notes
- `--fast` is fully supported here (stops at first consensus, dumps, then solves)
  — identical behavior to the serial path.
- If a run is hard-killed (`kill -9`), its spawn workers can be orphaned and keep
  holding GPU memory; a clean exit tears them down.
- `teachers/`, `recon/`, `results/`, `*.pt`, and `*.log` are git-ignored
  (regenerable artifacts).
