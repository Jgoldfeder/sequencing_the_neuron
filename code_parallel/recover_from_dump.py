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
from method import build_consensus, l1_on
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
    ap.add_argument("--samples", type=int, default=100000,
                    help="query subsample for the full-batch LBFGS solve "
                         "(memory-bound at high input dim)")
    ap.add_argument("--loss-samples", type=int, default=100000,
                    help="query subsample for reporting train L1 (full set is "
                         "slow at 1e6+ queries)")
    ap.add_argument("--mse-steps", type=int, default=40)
    ap.add_argument("--mae-steps", type=int, default=40)
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

    # --- staged MSE->MAE LBFGS solve on a memory-safe subsample ---
    S = min(args.samples, N)
    if S < N:
        print(f"[solve ] NOTE: solving on a {S}/{N} random subsample of queries "
              f"(full set is {N * dims[0] * 4 / 1024**3:.0f} GB, too big for GPU)",
              flush=True)
    gS = torch.Generator().manual_seed(1)
    si = torch.randperm(N, generator=gS)[:S]
    Xs, Ys = X[si].to(dev), Y[si].to(dev)
    net = cons

    @torch.no_grad()
    def report(tag):
        me = param_errors(net, teacher)["max_eps"]
        l = (net(Xs) - Ys).abs().mean().item()
        print(f"  {tag:14s} L1={l:.3e}  max_eps={me:.3e}", flush=True)
        return me

    def phase(kind, steps):
        opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=20,
                                history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            r = net(Xs) - Ys
            loss = (r ** 2).mean() if kind == "mse" else r.abs().mean()
            loss.backward()
            return loss
        for _ in range(steps):
            opt.step(closure)

    print(f"[solve ] staged MSE->MAE on {S} queries (device={dev}):", flush=True)
    report("consensus")
    phase("mse", args.mse_steps); report("+LBFGS-MSE")
    phase("mae", args.mae_steps)
    final_eps = report("+MAE")

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
    }, out)
    print(f"\n[save  ] solved reconstruction -> {out}", flush=True)
    print(f"[result] best_member={member_eps:.3e}  consensus={cons_eps:.3e}  "
          f"consensus+solve={final_eps:.3e}", flush=True)


if __name__ == "__main__":
    main()
