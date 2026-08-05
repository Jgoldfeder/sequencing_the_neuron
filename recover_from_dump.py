"""Recover the fast-path reconstruction from an existing `--fast` dump WITHOUT
re-running the query loop. Builds the teacher-free 5/8 consensus from the dumped
committee and runs the staged MSE->MAE LBFGS solve on the queries collected so
far, reporting max_eps / loss at each stage.

The LBFGS solve is full-batch, so at high input dim the whole query set can't sit
on the GPU (e.g. 1.08M x 12288 float32 = 53 GB). We solve on a memory-safe random
subsample of `--samples` queries (logged, never silent) and report the cap.

  CUDA_VISIBLE_DEVICES=0 python recover_from_dump.py \
      recon/_fast__12288x1024x200__s0.pt --samples 100000
"""
import argparse
import os

import torch

from nets import MLP
from method import build_consensus, l1_on, solver_polish_
from align import param_errors


def load_net(dims, state, dev):
    m = MLP(dims).to(dev)
    m.load_state_dict(state)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quorum", type=float, default=0.625,
                    help="consensus quorum ratio (0.625 = 5/8, the --fast value)")
    ap.add_argument("--samples", type=int, default=0,
                    help="cap the LBFGS solve to a random subsample of this many "
                         "queries; 0 (default) = use the FULL set (streamed to "
                         "the GPU in chunks, so it never has to sit there at once)")
    ap.add_argument("--solve-bs", type=int, default=8192,
                    help="chunk size for streaming queries to the GPU in the solve")
    ap.add_argument("--solverwindow", type=int, default=0,
                    help="restrict the solve to the last N outer iters of queries "
                         "(the most-recent tail, X[-N*q:]), mirroring the in-loop "
                         "polish's solverwindow; 0 (default) = all collected "
                         "queries. Applied before --samples.")
    ap.add_argument("--loss-samples", type=int, default=100000,
                    help="query subsample for reporting train L1 (full set is "
                         "slow at 1e6+ queries)")
    ap.add_argument("--mse-steps", type=int, default=40)
    ap.add_argument("--mae-steps", type=int, default=40)
    ap.add_argument("--float64", action="store_true",
                    help="after the float32 solve, run a float64 refinement pass "
                         "to break the float32 precision floor. OFF by default "
                         "(fp64 is ~40x slower on GeForce GPUs); opt in with "
                         "--float64.")
    ap.add_argument("--f64-samples", type=int, default=50000,
                    help="query subsample for the float64 refine. fp64 is ~40x "
                         "slower on GeForce GPUs, so the refine runs on a bounded "
                         "subset of the already-fit set; 0 = the full selected set "
                         "(hours at 12288-in). Default 50000.")
    ap.add_argument("--out", default="",
                    help="where to save the solved net (default: alongside dump)")
    args = ap.parse_args()
    dev = args.device

    print(f"[load] reading {args.dump} ...", flush=True)
    ck = torch.load(args.dump, map_location="cpu", weights_only=False)
    dims = ck["dims"]
    X, Y = ck["X"], ck["Y"]            # stay on CPU (too big for GPU at 12288-in)
    N = len(X)
    print(f"[load] dims={dims}  dumped_iter={ck['iter']}  queries={N}", flush=True)

    # committee + teacher weights are small -> straight onto the compute device
    teacher = load_net(dims, ck["teacher_state"], dev)
    pop = [load_net(dims, s, dev) for s in ck["pop_states"]]

    # subsample for quick loss reporting (X/Y indexed on CPU, moved per-batch)
    gL = torch.Generator().manual_seed(0)
    li = torch.randperm(N, generator=gL)[:min(args.loss_samples, N)]
    Xl, Yl = X[li], Y[li]

    # --- best single committee member (context for the comparison) ---
    losses = l1_on(pop, Xl, Yl)
    bi = min(range(len(pop)), key=lambda i: losses[i])
    member_eps = param_errors(pop[bi], teacher)["max_eps"]
    print(f"\n[member ] best single member = #{bi}: "
          f"train_L1={losses[bi]:.3e}  max_eps={member_eps:.3e}", flush=True)

    # --- consensus: THE model --fast produces ---
    cons = build_consensus(pop, dims, quorum_ratio=args.quorum)
    if cons is None:
        print(f"[consensus] no consensus at quorum={args.quorum}; "
              f"try a lower --quorum.", flush=True)
        return
    cons_eps = param_errors(cons, teacher)["max_eps"]
    cons_loss = l1_on([cons], Xl, Yl)[0]
    verdict = "LOWER" if cons_eps < member_eps else "higher"
    print(f"[consensus] quorum={args.quorum} (raw, pre-solve): "
          f"train_L1={cons_loss:.3e}  max_eps={cons_eps:.3e}  "
          f"-> {verdict} than best member ({member_eps:.3e})", flush=True)

    # --- staged MSE->MAE LBFGS solve. Queries stay on CPU; solver_polish_
    #     streams `--solve-bs` chunks to the GPU and accumulates the full-batch
    #     gradient, so the whole set (30 GB at 12288-in) never sits on the card.
    #     Query selection: --solverwindow first (most-recent tail), then an
    #     optional --samples random cap; default is the FULL set. ---
    # (1) solver window: keep the last `solverwindow` outer iters of queries.
    #     Queries are appended chronologically, so this is the tail X[-keep:],
    #     matching the in-loop polish (method.py: X[-solverwindow*q:]).
    Xw, Yw = X, Y
    if args.solverwindow and args.solverwindow > 0:
        q = max(1, N // max(1, ck["iter"]))
        keep = args.solverwindow * q
        if keep < N:
            Xw, Yw = X[-keep:], Y[-keep:]
            print(f"[solve ] solver window: last {args.solverwindow} iters "
                  f"(~{q}/iter) = {len(Xw)}/{N} most-recent queries", flush=True)
    W = len(Xw)
    # (2) optional random subsample cap on top of the window.
    if args.samples and args.samples < W:
        S = args.samples
        gS = torch.Generator().manual_seed(1)
        si = torch.randperm(W, generator=gS)[:S]
        Xs, Ys = Xw[si], Yw[si]
        print(f"[solve ] NOTE: --samples capped the solve to a {S}/{W} random "
              f"subsample of the windowed queries", flush=True)
    else:
        S = W
        Xs, Ys = Xw, Yw               # streamed to the GPU in chunks
    net = cons
    bs = args.solve_bs

    @torch.no_grad()
    def report(tag):
        me = param_errors(net, teacher)["max_eps"]
        l = l1_on([net], Xs, Ys, bs=bs)[0]
        print(f"  {tag:14s} L1={l:.3e}  max_eps={me:.3e}", flush=True)
        return me

    print(f"[solve ] float32 staged MSE->MAE on {S} queries "
          f"(device={dev}, chunk={bs}):", flush=True)
    report("consensus")
    solver_polish_(net, Xs, Ys, mse_steps=args.mse_steps, mae_steps=0, bs=bs)
    report("+LBFGS-MSE")
    solver_polish_(net, Xs, Ys, mse_steps=0, mae_steps=args.mae_steps, bs=bs)
    final_eps = report("+MAE")

    # --- float64 refinement: the float32 solve floors around ~1e-2/1e-3 param
    #     error; a float64 pass breaks that floor. fp64 is ~40x slower on GeForce
    #     cards, so refine on a bounded subsample of the (already-fit) set. ---
    if args.float64:
        Sd = S if args.f64_samples <= 0 else min(args.f64_samples, S)
        if Sd < S:
            gd = torch.Generator().manual_seed(2)
            di = torch.randperm(S, generator=gd)[:Sd]
            Xd, Yd = Xs[di], Ys[di]
        else:
            Xd, Yd = Xs, Ys
        net = net.double()            # promote params to fp64 to break the floor
        print(f"[solve ] float64 refine on {Sd}/{S} queries "
              f"(device={dev}, chunk={bs}):", flush=True)
        solver_polish_(net, Xd, Yd, mse_steps=args.mse_steps, mae_steps=0, bs=bs)
        report("+f64-MSE")
        solver_polish_(net, Xd, Yd, mse_steps=0, mae_steps=args.mae_steps, bs=bs)
        final_eps = report("+f64-MAE")
        net = net.float()             # back to fp32 for saving (max_eps << fp32 eps)

    out = args.out or os.path.join(os.path.dirname(args.dump),
                                   "recovered_solved.pt")
    torch.save({
        "dims": dims,
        "state": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "final_max_eps": final_eps,
        "consensus_max_eps": cons_eps,
        "best_member_max_eps": member_eps,
        "queries_total": N,
        "solve_samples": S,
        "solve_window_iters": args.solverwindow,
        "float64_refine": bool(args.float64),
    }, out)
    print(f"\n[save  ] solved reconstruction -> {out}", flush=True)
    print(f"[result] best_member={member_eps:.3e}  consensus={cons_eps:.3e}  "
          f"consensus+solve={final_eps:.3e}", flush=True)


if __name__ == "__main__":
    main()
