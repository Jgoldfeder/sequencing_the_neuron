"""Experiment runner: build/loads a shared teacher, runs a reconstruction
variant, writes results JSON. Usage:

  python run.py --variant v0_baseline --seed 0 --arch 784,64,10
"""
import argparse
import json
import os
import time

import torch

from data import make_teacher, load_data
from method import (Cfg, reconstruct, build_consensus, solver_polish_,
                    l1_on, agreement)
from nets import count_params, MLP
from align import param_errors

RESULTS = os.path.join(os.path.dirname(__file__), "results")

VARIANTS = {
    # paper-faithful baseline (Algorithms 1+2 as described)
    "v0_baseline": dict(),
    # ablation: the underfitting trap (expect failure) - validates C1
    "v0a_underfit": dict(epochs=2),
    # disagreement objective variants (Grok#1, GPT#1)
    "v1a_minpair": dict(disagree="min_pair"),
    "v1b_medianpair": dict(disagree="median_pair"),
    "v1c_variance": dict(disagree="variance"),
    # query-space variants (GPT#2, Grok#4)
    "v2_box": dict(query_box=1.5),
    "v3_div": dict(query_div_w=0.1),
    # dataset management (GPT#3, Grok#3)
    "v4_window": dict(window=20),
    # committee maintenance (Grok#2, GPT#5)
    "v5_maint": dict(maint_every=5, restart_worst=2),
    # last-mile float64 polish
    "v6_polish": dict(polish_f64=True),
    # random warm-start then committee (Grok#7, GPT#6)
    "v8_warmstart": dict(warmstart_iters=5),
    # adaptive tight-fit: stop inner epochs once D is fit (Kimi H1)
    "v9_fitdelta": dict(epochs=15, fit_delta=1e-3),
    # early stopping on App F signal (Kimi H8)
    "v10_earlystop": dict(stop_loss=1e-6, stop_patience=2),
    # combined best mechanisms
    "v7_combo": dict(disagree="min_pair", query_box=1.5, query_div_w=0.1,
                     window=20, maint_every=5, restart_worst=2),
    # ---- batch 2 (post-synthesis; Opus/GPT mechanisms) ----
    # squared fitting loss: residual-proportional gradients (Opus A2)
    "v11_mse": dict(fit_loss="mse"),
    # closed-form ridge LS solve of last layer every 5 iters + final (Opus A12)
    "v12_lastlayer": dict(lastlayer_every=5),
    # aligned population averaging, kappa=3 (Opus A9)
    "v13_popavg": dict(popavg_kappa=3.0),
    # float64 LBFGS squared-loss endgame (Opus A2+A3-lite)
    "v14_lbfgs": dict(lbfgs_polish=True),
    # fit-gated committee for query generation, kappa=3 (Opus T7)
    "v15_gate": dict(gate_kappa=3.0),
    # combo of orthogonal batch-1 winners
    "v16_combo2": dict(disagree="median_pair", window=20, warmstart_iters=5),
    # retuned early stop (v10's threshold never fired: final loss ~1e-4)
    "v10b_earlystop": dict(stop_loss=5e-4, stop_patience=2),
    # full stack: batch-1 winners + batch-2 mechanisms
    "v17_full": dict(disagree="median_pair", window=20, warmstart_iters=5,
                     fit_loss="mse", lastlayer_every=5, popavg_kappa=3.0,
                     lbfgs_polish=True, gate_kappa=3.0),
    # v17 attribution ablations (which ingredients are load-bearing?)
    "v17a_nolf": dict(disagree="median_pair", window=20, warmstart_iters=5,
                      fit_loss="mse", lastlayer_every=5, popavg_kappa=3.0,
                      gate_kappa=3.0),                      # no LBFGS
    "v17b_nopop": dict(disagree="median_pair", window=20, warmstart_iters=5,
                       fit_loss="mse", lastlayer_every=5,
                       lbfgs_polish=True, gate_kappa=3.0),  # no pop-avg
    "v17c_nomse": dict(disagree="median_pair", window=20, warmstart_iters=5,
                       lastlayer_every=5, popavg_kappa=3.0,
                       lbfgs_polish=True, gate_kappa=3.0),  # L1 fit loss
    "v17d_noll": dict(disagree="median_pair", window=20, warmstart_iters=5,
                      fit_loss="mse", popavg_kappa=3.0,
                      lbfgs_polish=True, gate_kappa=3.0),   # no last-layer
    # minimal stack: is popavg/lbfgs dead weight in v17d?
    "v18_min": dict(disagree="median_pair", window=20, warmstart_iters=5,
                    fit_loss="mse", gate_kappa=3.0),
    # v18_min + closed-form last-layer LS solve (test on wider nets: does the
    # solver help where the last layer dominates error, unlike 784x32x10?)
    "v18_ll": dict(disagree="median_pair", window=20, warmstart_iters=5,
                   fit_loss="mse", gate_kappa=3.0, lastlayer_every=5),
    # v18_min + float64 LBFGS squared-loss endgame
    "v18_lbfgs": dict(disagree="median_pair", window=20, warmstart_iters=5,
                      fit_loss="mse", gate_kappa=3.0, lbfgs_polish=True),
    # v18_min + both solvers
    "v18_ll_lbfgs": dict(disagree="median_pair", window=20, warmstart_iters=5,
                         fit_loss="mse", gate_kappa=3.0,
                         lastlayer_every=5, lbfgs_polish=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="784,64,10")
    ap.add_argument("--teacher-seed", type=int, default=0)
    ap.add_argument("--teacher-epochs", type=int, default=25)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--outer", type=int, default=40)
    ap.add_argument("--q", type=int, default=1500)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--window", type=int, default=None,
                    help="override variant's sample window (in outer iters; "
                         "0 = keep all). Default: use the variant's value.")
    ap.add_argument("--combine", action="store_true",
                    help="the first outer iter the cluster consensus becomes "
                         "available, replace the worst committee member with "
                         "the consensus net.")
    ap.add_argument("--solver-polish", action="store_true",
                    help="at the end of each outer iter, tighten every member "
                         "with the staged LBFGS recipe (MSE then MAE).")
    ap.add_argument("--solverwindow", type=int, default=None,
                    help="window (in outer iters) of recent queries to "
                         "solver-polish on (default 10).")
    ap.add_argument("--verbose", action="store_true",
                    help="print per-member polish detail (loss before->after, "
                         "#evals, time).")
    ap.add_argument("--fast", action="store_true",
                    help="stop training the first iter a consensus forms, then "
                         "staged MSE->MAE solve it, and emit the solved "
                         "reconstruction (much fewer queries).")
    ap.add_argument("--save-recon", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="save the pre-endgame reconstruction (student + full "
                         "population + query set + teacher) to recon/*.pt so "
                         "the solvers can be re-tried offline via polish.py. "
                         "On by default; pass --no-save-recon to disable.")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dims = [int(x) for x in args.arch.split(",")]
    device = args.device
    torch.manual_seed(args.teacher_seed)

    print(f"[setup] teacher {dims} epochs={args.teacher_epochs} "
          f"device={device}", flush=True)
    teacher = make_teacher(dims, epochs=args.teacher_epochs,
                           seed=args.teacher_seed, device=device,
                           verbose=False)
    (_, _), (xte, _) = load_data(dims, device)
    eval_pts = xte[:2000]

    overrides = dict(VARIANTS[args.variant])
    if args.window is not None:
        overrides["window"] = args.window
    if args.combine:
        overrides["combine"] = True
    if args.solver_polish:
        overrides["solver_polish"] = True
    if args.solverwindow is not None:
        overrides["solverwindow"] = args.solverwindow
    if args.verbose:
        overrides["verbose"] = True
    tag = f"_{args.tag}" if args.tag else ""
    arch_tag = "x".join(map(str, dims))
    recon_dir = os.path.join(os.path.dirname(__file__), "recon")
    fast_dump = os.path.join(recon_dir, f"_fast__{arch_tag}__s{args.seed}.pt")
    if args.fast:
        os.makedirs(recon_dir, exist_ok=True)
        overrides["stop_on_consensus"] = True
        overrides["dump_path"] = fast_dump
    cfg = Cfg(p=args.p, q=args.q, outer=args.outer, **overrides)
    print(f"[run] {args.variant} seed={args.seed} params={count_params(teacher)} "
          f"budget={cfg.outer * cfg.q} queries"
          + (" [--fast]" if args.fast else ""), flush=True)
    t0 = time.time()
    save_recon = None
    if args.save_recon and not args.fast:
        os.makedirs(recon_dir, exist_ok=True)
        save_recon = os.path.join(
            recon_dir, f"{args.variant}{tag}__{arch_tag}__s{args.seed}.pt")
    best, log, final = reconstruct(teacher, dims, cfg, device, eval_pts,
                                   seed=args.seed, save_recon=save_recon)

    if args.fast and os.path.exists(fast_dump):
        # reconstruct stopped + dumped at the first consensus; build it and run
        # the staged MSE->MAE solve on the queries collected so far.
        ck = torch.load(fast_dump, map_location=device, weights_only=False)
        Xf, Yf = ck["X"].to(device), ck["Y"].to(device)
        pop = []
        for s in ck["pop_states"]:
            m = MLP(dims).to(device); m.load_state_dict(s); pop.append(m)
        cons = build_consensus(pop, dims, quorum_ratio=0.625)
        if cons is None:
            print("[fast] no consensus formed; keeping trained best.", flush=True)
        else:
            cons = cons.to(device)
            print(f"[fast] consensus at iter {ck['iter']} ({len(Xf)} queries): "
                  f"max_eps {param_errors(cons, teacher)['max_eps']:.3e} -> "
                  f"staged MSE->MAE solve...", flush=True)
            solver_polish_(cons, Xf, Yf, mse_steps=40, mae_steps=40,
                           verbose=args.verbose, tag=" fast")
            errs = param_errors(cons, teacher)
            best = cons
            final = {
                "final_max_eps": errs["max_eps"],
                "final_mean_eps": sum(errs["mean_eps_per_matrix"]) /
                len(errs["mean_eps_per_matrix"]),
                "final_max_eps_per_matrix": errs["max_eps_per_matrix"],
                "final_agree": agreement(cons, teacher, eval_pts),
                "queries": len(Xf), "stopped_iter": ck["iter"],
                "wall_s": round(time.time() - t0, 1),
            }
        os.remove(fast_dump)
    out = {
        "variant": args.variant,
        "cfg": {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in cfg.__dict__.items()},
        "arch": dims,
        "n_params": count_params(teacher),
        "seed": args.seed,
        "teacher_seed": args.teacher_seed,
        "device": device,
        "log": log,
        **final,
    }
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(
        RESULTS, f"{args.variant}{tag}__{arch_tag}__s{args.seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[done] {path} | max_eps={final['final_max_eps']:.3e} "
          f"agree={final['final_agree']:.4f} wall={final['wall_s']}s",
          flush=True)


if __name__ == "__main__":
    main()
