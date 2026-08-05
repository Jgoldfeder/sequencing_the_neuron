# code_parallel — multi-GPU variant

Self-contained copy of the reconstruction code with **single-run population
parallelism** across GPUs. The serial code at the repo root is unchanged; this
directory is a drop-in parallel variant that takes the same command lines plus a
few extra flags.

## What is parallelized
- **Training** — the population of `--p` students is split across the GPUs given
  by `--gpus` (one persistent worker process per GPU, so no GIL contention on the
  small per-step kernels). See `parallel_pool.py` (`WorkerPool`, `reconstruct_mp`).
- **Query generation** — the committee-disagreement query batch is split
  `n / num_gpus` across GPUs and gathered (`gen_queries_parallel` in `method.py`).
- **`--polish` endgame** — the float64 last-mile LBFGS shards the query set across
  GPUs and sums per-shard gradients each closure (exact same math as single-GPU),
  giving ~Nx on the fp64 forward/backward (`solver_polish_full_parallel_`).

## Extra flags (vs the serial code)
- `--gpus 0,1,2` — GPU indices; members split across them (routes to `reconstruct_mp`).
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
