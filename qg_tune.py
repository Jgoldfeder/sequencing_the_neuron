#!/usr/bin/env python
"""qg_tune.py -- find the fastest disagreement-sampler batching for THIS GPU and THIS network.

    python qg_tune.py --arch 3072,200,200,200,200,200,200,200,200,200,100 --act tanh [--p 8] [--tf32]

Independent of the run's --q: the sampler processes q in blocks, so only the block/chunk size matters.

Runs method.gen_queries (the real sampler) on randomly initialised members of the given architecture
with a growing forward chunk (chunk = block, one chunk per outer block), measures us/query and peak GPU
memory for each, stops at OOM, and prints the --qg-block-gb / --qg-chunk-gb flags of the smallest
setting within --tol of the fastest (bigger buys nothing once the card is saturated). Optionally also
measures --qg-tf32 at that setting. Timings are interleaved over --reps passes (min taken) so thermal
drift / boost-clock changes do not bias the comparison.
"""
import argparse
import subprocess
import time

import torch

from method import Cfg, gen_queries
from nets import MLP


def gpu_state():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                               "--format=csv,noheader,nounits", "-i", "0"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="comma-separated widths, e.g. 3072,200,200,100")
    ap.add_argument("--act", default="leaky_relu")
    ap.add_argument("--p", type=int, default=8, help="committee size")
    ap.add_argument("--min-rows", type=int, default=4096)
    ap.add_argument("--max-rows", type=int, default=1 << 21)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.03, help="pick the smallest setting within tol of the best")
    ap.add_argument("--tf32", action="store_true", help="also measure --qg-tf32 at the chosen setting")
    ap.add_argument("--disagree", default=None, help="disagreement mode override (default: Cfg default)")
    args = ap.parse_args()

    dims = [int(x) for x in args.arch.split(",")]
    d, D, hid, P = dims[0], dims[-1], sum(dims[1:-1]), args.p
    dev = "cuda"
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU: {torch.cuda.get_device_name(0)} ({total / 1e9:.0f} GB) | net {'x'.join(map(str, dims))} "
          f"({args.act}) | {P} members", flush=True)
    pop = [MLP(dims, act=args.act).to(dev) for _ in range(P)]
    extra = {"disagree": args.disagree} if args.disagree else {}

    # memory per query row of the two budgets, exactly as method.gen_queries converts GB -> rows
    leaf_b = 4 * d * 4                                   # leaf + grad + 2 Adam moments
    chunk_b = P * P * D * 4 + 3 * P * hid * 4            # pairwise tensor + autograd graph (formula; actual is lower)

    def to_flags(rows):
        return rows * leaf_b / 1e9, rows * chunk_b / 1e9

    cands = []
    r = args.min_rows
    while r <= args.max_rows:
        cands.append(r)
        r *= 2

    def measure(rows, tf32=False, block_rows=None):
        block_rows = block_rows or rows
        bg, cg = to_flags(block_rows)
        cfg = Cfg(q=block_rows, p=P, act=args.act, qg_block_gb=bg + 1e-9, qg_chunk=rows, qg_tf32=tf32, **extra)
        gen = torch.Generator(device=dev); gen.manual_seed(0)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.time()
        gen_queries(pop, cfg, d, dev, gen)
        torch.cuda.synchronize()
        return (time.time() - t0) / block_rows * 1e6, torch.cuda.max_memory_allocated() / 1e9

    # warm-up
    measure(min(cands))
    res = {}
    dead = set()
    for rep in range(args.reps):
        for rows in cands:
            if rows in dead:
                continue
            try:
                us, mem = measure(rows)
            except torch.cuda.OutOfMemoryError:
                dead.add(rows); torch.cuda.empty_cache()
                print(f"  rows {rows:>8d}: OOM", flush=True)
                continue
            res.setdefault(rows, []).append((us, mem))
            print(f"  pass {rep} rows {rows:>8d}: {us:7.1f} us/query  peak {mem:5.1f} GB  [{gpu_state()}]", flush=True)
    ok = [r for r in cands if r in res]
    if not ok:
        print("nothing fit on the GPU"); return
    best_us = {r: min(u for u, _ in res[r]) for r in ok}
    mem = {r: max(m for _, m in res[r]) for r in ok}
    fastest = min(ok, key=lambda r: best_us[r])
    pick = next(r for r in ok if best_us[r] <= best_us[fastest] * (1 + args.tol))

    print("\nrows/chunk (one chunk per block)   us/query   peak GB")
    for r in ok:
        mark = " <- pick" if r == pick else (" (fastest)" if r == fastest else "")
        print(f"  {r:>10d}   {best_us[r]:8.1f}   {mem[r]:6.1f}{mark}")
    # does a larger outer block with the chosen chunk help? (it never did on the 3090; check anyway)
    big = pick * 2
    if big in ok or big <= max(ok):
        try:
            # interleaved against a fresh baseline (the card drifts thermally over the sweep)
            pairs = [(measure(pick, block_rows=big), measure(pick)) for _ in range(args.reps)]
            us2, mem2 = min(a for a, _ in pairs); us1 = min(b[0] for _, b in pairs)
            gain = us2 <= us1 * (1 - args.tol)
            print(f"  block {big} with chunk {pick}: {us2:7.1f} vs {us1:7.1f} us/query (block = chunk), peak {mem2:.1f} GB "
                  f"({'faster -> larger block' if gain else 'no gain'})")
            pick_block = big if gain else pick
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); pick_block = pick
    else:
        pick_block = pick
    bg, cg = to_flags(pick_block)
    _, cg_chunk = to_flags(pick)
    line = f"--qg-block-gb {bg:.2f} --qg-chunk-gb {cg_chunk:.2f}"
    if args.tf32:
        us_t = min(measure(pick, tf32=True, block_rows=pick_block)[0] for _ in range(args.reps))
        us_f = min(measure(pick, block_rows=pick_block)[0] for _ in range(args.reps))
        print(f"  --qg-tf32 at this setting: {us_t:7.1f} vs {us_f:7.1f} us/query fp32 = {us_f / us_t:.2f}x "
              f"(sampler only; same disagreement objective, different individual points)")
        if us_t < us_f * (1 - args.tol):
            line += " --qg-tf32"
    print(f"\nRECOMMENDED: {line}")
    print(f"  = block {pick_block} queries, chunk {pick} rows, peak sampler memory ~{mem[pick]:.1f} GB "
          f"({mem[pick] / total * 1e9 * 100:.0f}% of the card); {best_us[pick]:.1f} us/query. "
          f"Independent of the run's --q (a block never exceeds q; q only sets how many blocks run).")


if __name__ == "__main__":
    main()
