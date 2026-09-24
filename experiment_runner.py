"""Escalating-budget experiment runner.

Wraps the reconstruction algorithm and escalates the per-iteration query count
``q`` until the reconstruction succeeds or a ceiling ``maxq`` is reached:

  * run the algorithm at the current ``q`` (total samples drawn = ``outer * q``);
  * if it SUCCEEDS -- aligned max parameter error < ``--threshold`` (default
    1e-3) -- stop;
  * otherwise, as long as ``q < maxq``, set ``q = min(2*q, maxq)`` and rerun.

Every attempt -- success or failure -- is recorded in full:

  1. wall time;
  2. max/mean parameter error for the whole network AND, separately, for each
     weight matrix and each bias vector (after scale-normalization + greedy
     permutation alignment, i.e. modulo the network's symmetries);
  3. total samples (queries actually issued to the black box);
  4. MAE and MSE loss -- on the accumulated query set D (what the surrogate was
     fit on) and on a held-out teacher test set (functional fidelity);
  5. the full run log: the structured per-outer-iteration records returned by
     ``reconstruct`` plus the captured stdout of the run.

Results (all attempts) are written to ``results_sweep/`` as one JSON per
experiment and re-flushed after every attempt, so a partial sweep is never lost.

Usage:
  python experiment_runner.py --variant v18_min --arch 3072,256,100 \
      --q 1500 --maxq 96000 --outer 40 --p 8 --seed 0
"""
import argparse
import io
import json
import os
import sys
import time
import traceback

import torch

from data import make_teacher, load_data
from method import (Cfg, reconstruct, l1_on, mse_on, agreement,  # noqa: F401
                    build_consensus, solver_polish_)
from nets import count_params, MLP
from align import align_clone_to, param_errors
from run import VARIANTS

OUTDIR = os.path.join(os.path.dirname(__file__), "results_sweep")
RECON = os.path.join(os.path.dirname(__file__), "recon")


class _Tee:
    """Duplicate writes to several streams: the live stdout (so the user still
    sees progress) and a capture buffer (so the run log can be recorded)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@torch.no_grad()
def layered_param_errors(recon, teacher):
    """Max/mean |error| for the whole net and, separately, for every weight
    matrix and bias vector -- after scale-normalization + greedy permutation
    alignment (align_clone_to clones, so `recon` is not mutated). The overall
    mean is the true parameter-count-weighted mean over all parameters."""
    t, r = align_clone_to(recon, teacher)
    per_layer, all_abs = [], []
    for li in range(len(t.layers)):
        wd = (t.layers[li].weight - r.layers[li].weight).abs()
        bd = (t.layers[li].bias - r.layers[li].bias).abs()
        per_layer.append({
            "layer": li,
            "weight": {"max_eps": wd.max().item(), "mean_eps": wd.mean().item(),
                       "count": wd.numel()},
            "bias": {"max_eps": bd.max().item(), "mean_eps": bd.mean().item(),
                     "count": bd.numel()},
        })
        all_abs.append(wd.reshape(-1))
        all_abs.append(bd.reshape(-1))
    cat = torch.cat(all_abs)
    return {"max_eps": cat.max().item(), "mean_eps": cat.mean().item(),
            "count": cat.numel(), "per_layer": per_layer}


@torch.no_grad()
def testset_losses(net, teacher, X, bs=4096):
    """Per-output-element MAE and MSE between the reconstruction and the teacher
    on held-out inputs X (functional fidelity on the data manifold)."""
    dev = next(net.parameters()).device
    mae, mse, n = 0.0, 0.0, 0
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(dev)
        d = net(xb) - teacher(xb)
        mae += d.abs().sum().item()
        mse += (d ** 2).sum().item()
        n += d.numel()
    return mae / n, mse / n


def fast_consensus_solve(dump_path, dims, device, teacher, cfg):
    """Replicate run.py --fast post-processing: load the stop-on-consensus dump
    (population + queries), build the 5/8 committee consensus, then tighten it
    with the staged MSE->MAE LBFGS solve. Returns (best_or_None, n_total,
    stopped_iter, Xf, Yf). `best` is None if no consensus could be built."""
    ck = torch.load(dump_path, map_location="cpu", weights_only=False)
    Xf, Yf = ck["X"], ck["Y"]
    n_total = len(Xf)
    if cfg.solverwindow and cfg.solverwindow > 0:      # restrict solve to tail
        keep = cfg.solverwindow * cfg.q
        if keep < n_total:
            Xf, Yf = Xf[-keep:], Yf[-keep:]
            print(f"[fast] solver window: last {cfg.solverwindow} iters = "
                  f"{len(Xf)}/{n_total} most-recent queries", flush=True)
    pop = []
    for s in ck["pop_states"]:
        m = MLP(dims).to(device)
        m.load_state_dict(s)
        pop.append(m)
    cons = build_consensus(pop, dims, quorum_ratio=0.625)
    if cons is not None:
        cons = cons.to(device)
        print(f"[fast] consensus at iter {ck['iter']} ({len(Xf)} queries): "
              f"max_eps {param_errors(cons, teacher)['max_eps']:.3e} -> "
              f"staged MSE->MAE solve...", flush=True)
        solver_polish_(cons, Xf, Yf, mse_steps=40, mae_steps=40, tag=" fast")
    return cons, n_total, ck["iter"], Xf, Yf


def run_attempt(teacher, dims, device, eval_pts, overrides, q, outer, p, seed,
                fast=False, slug="run"):
    """Run one reconstruction at per-iteration budget `q`, capturing full
    metrics and the run's stdout. With `fast`, the run stops at the first
    committee consensus, which is then staged-solved (run.py --fast); the solved
    net is what gets scored. Returns a record dict (no `success` field; the
    caller applies the threshold). On an exception the record carries an `error`
    traceback instead of metrics."""
    cfg_over = dict(overrides)
    dump_path = None
    if fast:
        os.makedirs(RECON, exist_ok=True)
        dump_path = os.path.join(RECON, f"_fast__{slug}__q{q}.pt")
        cfg_over["stop_on_consensus"] = True
        cfg_over["dump_path"] = dump_path
    cfg = Cfg(p=p, q=q, outer=outer, **cfg_over)

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, buf)
    t0 = time.time()
    err = None
    best = log = final = None
    stopped_iter = None
    total_samples = mae_q = mse_q = None
    try:
        best, log, final = reconstruct(teacher, dims, cfg, device, eval_pts,
                                       seed=seed, save_recon=None)
        total_samples = final["queries"]
        mae_q, mse_q = final["final_mae"], final["final_mse"]
        if fast and dump_path and os.path.exists(dump_path):
            cons, n_total, stopped_iter, Xf, Yf = fast_consensus_solve(
                dump_path, dims, device, teacher, cfg)
            if cons is None:
                print("[fast] no consensus formed; keeping trained best.",
                      flush=True)
            else:
                best = cons                            # solved net is scored
                total_samples = n_total
                mae_q = l1_on([best], Xf, Yf)[0]
                mse_q = mse_on([best], Xf, Yf)[0]
    except Exception:
        err = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        if dump_path and os.path.exists(dump_path):
            try:
                os.remove(dump_path)
            except OSError:
                pass
    wall = round(time.time() - t0, 1)

    rec = {
        "q": q,
        "outer": outer,
        "budget": outer * q,          # intended total samples for this attempt
        "wall_s": wall,
        "stdout": buf.getvalue(),     # (5) full captured run log
    }
    if err is not None:
        rec["error"] = err
        rec["total_samples"] = None
        return rec

    eps = layered_param_errors(best, teacher)          # (2)
    te_mae, te_mse = testset_losses(best, teacher, eval_pts)
    rec.update({
        "total_samples": total_samples,                # (3) actually issued
        "network": {"max_eps": eps["max_eps"],
                    "mean_eps": eps["mean_eps"],
                    "n_params": eps["count"]},
        "per_layer": eps["per_layer"],
        "mae_queryset": mae_q,                          # (4) loss on solve set D
        "mse_queryset": mse_q,
        "mae_testset": te_mae,                          # (4) loss on held-out
        "mse_testset": te_mse,
        "agree": agreement(best, teacher, eval_pts),
        "log": log,                                     # (5) structured records
    })
    if stopped_iter is not None:
        rec["stopped_iter"] = stopped_iter             # --fast consensus iter
    if final and "popavg" in final:
        rec["popavg"] = final["popavg"]
    return rec


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # the usual reconstruction args (mirrors run.py)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="784,64,10")
    ap.add_argument("--teacher-seed", type=int, default=0)
    ap.add_argument("--teacher-epochs", type=int, default=25)
    default_dev = ("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--device", default=default_dev)
    ap.add_argument("--outer", type=int, default=40,
                    help="outer iterations per attempt (held fixed across the "
                         "sweep; total samples per attempt = outer * q)")
    ap.add_argument("--p", type=int, default=8, help="committee size")
    ap.add_argument("--window", type=int, default=None,
                    help="override the variant's sample window (outer iters; "
                         "0 = keep all)")
    # the escalation knobs
    ap.add_argument("--q", type=int, default=1500,
                    help="starting per-iteration query count")
    ap.add_argument("--maxq", type=int, required=True,
                    help="ceiling on per-iteration query count; on failure q is "
                         "doubled (capped here) and the run is retried")
    ap.add_argument("--threshold", type=float, default=1e-3,
                    help="success criterion on the aligned network max_eps")
    # reconstruction modes passed through to reconstruct() (as in run.py)
    ap.add_argument("--combine", action="store_true",
                    help="the first outer iter a consensus forms, replace the "
                         "worst committee member with the polished consensus")
    ap.add_argument("--fast", action="store_true",
                    help="stop each attempt at the first committee consensus "
                         "and staged MSE->MAE solve it (far fewer queries); the "
                         "solved net's max_eps drives the q escalation")
    ap.add_argument("--solverwindow", type=int, default=None,
                    help="restrict the --fast solve to the last N outer iters "
                         "of queries (0/omit = all)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--qg_lr", type=float, default=None,
                    help="override the query-generator Adam lr on top of the "
                         "variant (Cfg.qg_lr, e.g. 0.1)")
    return ap.parse_args()


def main():
    args = parse_args()
    dims = [int(x) for x in args.arch.split(",")]
    device = args.device
    if args.maxq < args.q:
        print(f"[warn] maxq ({args.maxq}) < q ({args.q}); running a single "
              f"attempt at q={args.maxq}.", flush=True)

    torch.manual_seed(args.teacher_seed)
    print(f"[setup] teacher {dims} epochs={args.teacher_epochs} device={device}",
          flush=True)
    teacher = make_teacher(dims, epochs=args.teacher_epochs,
                           seed=args.teacher_seed, device=device, verbose=False)
    (_, _), (xte, _) = load_data(dims, device)
    eval_pts = xte[:2000]

    overrides = dict(VARIANTS[args.variant])
    if args.window is not None:
        overrides["window"] = args.window
    if args.combine:
        overrides["combine"] = True
    if args.solverwindow is not None:
        overrides["solverwindow"] = args.solverwindow
    if args.verbose:
        overrides["verbose"] = True
    if args.qg_lr is not None:
        overrides["qg_lr"] = args.qg_lr

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    arch_tag = "x".join(map(str, dims))
    slug = f"{args.variant}{tag}__{arch_tag}__s{args.seed}"
    outpath = os.path.join(args.outdir, f"sweep_{slug}.json")

    attempts = []
    succeeded, success_q = False, None

    def flush_results():
        out = {
            "variant": args.variant,
            "arch": dims,
            "n_params": count_params(teacher),
            "seed": args.seed,
            "teacher_seed": args.teacher_seed,
            "device": device,
            "outer": args.outer,
            "p": args.p,
            "start_q": args.q,
            "maxq": args.maxq,
            "success_threshold": args.threshold,
            "combine": args.combine,
            "fast": args.fast,
            "cfg_overrides": overrides,
            "succeeded": succeeded,
            "success_q": success_q,
            "n_attempts": len(attempts),
            "total_wall_s": round(sum(a["wall_s"] for a in attempts), 1),
            "total_samples_all_attempts":
                sum(a.get("total_samples") or 0 for a in attempts),
            "attempts": attempts,
        }
        tmp = outpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, outpath)

    print(f"[sweep] {args.variant} arch={arch_tag} "
          f"params={count_params(teacher)} q0={args.q} maxq={args.maxq} "
          f"outer={args.outer} p={args.p} threshold={args.threshold:g}",
          flush=True)

    q = min(args.q, args.maxq)
    while True:
        n = len(attempts) + 1
        print(f"\n{'=' * 64}\n[attempt {n}] q={q}  budget={args.outer * q} "
              f"samples  (maxq={args.maxq})\n{'=' * 64}", flush=True)
        rec = run_attempt(teacher, dims, device, eval_pts, overrides, q,
                          args.outer, args.p, args.seed,
                          fast=args.fast, slug=slug)

        if "error" in rec:
            rec["success"] = False
            attempts.append(rec)
            flush_results()
            print(f"[attempt {n}] ERROR at q={q} (recorded); stopping sweep.\n"
                  f"{rec['error']}", flush=True)
            break

        rec["success"] = rec["network"]["max_eps"] < args.threshold
        attempts.append(rec)
        flush_results()
        net = rec["network"]
        print(f"[attempt {n}] q={q} | max_eps={net['max_eps']:.3e} "
              f"mean_eps={net['mean_eps']:.3e} | samples={rec['total_samples']} "
              f"| MAE(D)={rec['mae_queryset']:.3e} MSE(D)={rec['mse_queryset']:.3e} "
              f"| agree={rec['agree']:.4f} | wall={rec['wall_s']}s | "
              f"{'SUCCESS' if rec['success'] else 'fail'}", flush=True)

        if rec["success"]:
            succeeded, success_q = True, q
            flush_results()
            break
        if q >= args.maxq:
            print(f"[sweep] reached maxq={args.maxq} without success; stopping.",
                  flush=True)
            break
        q = min(q * 2, args.maxq)

    flush_results()
    status = (f"SUCCEEDED at q={success_q}" if succeeded
              else "FAILED (reached maxq)")
    print(f"\n[done] {status} | attempts={len(attempts)} | "
          f"total wall={sum(a['wall_s'] for a in attempts):.1f}s | "
          f"total samples={sum(a.get('total_samples') or 0 for a in attempts)} "
          f"| -> {outpath}", flush=True)


if __name__ == "__main__":
    main()
