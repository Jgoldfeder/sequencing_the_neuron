"""End-to-end fast reconstruction:

  1. train the surrogate committee (v18_lbfgs config) with the real LR schedule,
     stopping the FIRST outer iter a tight-cluster consensus forms (n/a -> answer);
  2. build the teacher-free consensus from that committee;
  3. run a staged second-order solve on the queries collected so far:
        LBFGS-MSE  (descend into the basin; its gradient vanishes near 0-residual)
     -> LBFGS-MAE  (constant gradient finishes the flat directions MSE abandons);
  4. report loss / max_eps at each stage.

This recovers a wide net to ~1e-5 max parameter error from a fraction of the
query budget -- e.g. 784x512x10 to 1.8e-5 from 480k queries -- vs a stalled
best-single-member of ~0.25.

  python fast_reconstruct.py --arch 784,512,10 --q 32000 [--stop-iter N]
"""
import argparse
import os
import torch

from data import make_teacher, load_data
from method import Cfg, reconstruct, build_consensus, l1_on
from run import VARIANTS
from align import param_errors


def staged_solve(net, X, Y, teacher, mse_steps=40, mae_steps=40):
    def phase(kind, steps):
        opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=20,
                                history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            r = net(X) - Y
            loss = (r ** 2).mean() if kind == "mse" else r.abs().mean()
            loss.backward()
            return loss
        for _ in range(steps):
            opt.step(closure)

    def report(tag):
        me = param_errors(net, teacher)["max_eps"]
        print(f"  {tag:16s} loss={l1_on([net], X, Y)[0]:.3e}  "
              f"max_eps={me:.3e}", flush=True)
    report("consensus")
    phase("mse", mse_steps); report("+LBFGS-MSE")
    phase("mae", mae_steps); report("+MAE")
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="784,512,10")
    ap.add_argument("--q", type=int, default=32000)
    ap.add_argument("--outer", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stop-iter", type=int, default=0,
                    help="stop at this outer iter; 0 = auto-stop at first consensus")
    args = ap.parse_args()

    dims = [int(x) for x in args.arch.split(",")]
    dev = args.device
    torch.manual_seed(args.seed)
    teacher = make_teacher(dims, epochs=25, seed=args.seed, device=dev, verbose=False)
    (_, _), (xte, _) = load_data(dims, dev)

    dump = os.path.join(os.path.dirname(__file__), "recon", "_fast_dump.pt")
    os.makedirs(os.path.dirname(dump), exist_ok=True)
    over = dict(VARIANTS["v18_lbfgs"]); over["window"] = args.outer
    cfg = Cfg(p=8, q=args.q, outer=args.outer, dump_path=dump,
              dump_at_iter=args.stop_iter,
              stop_on_consensus=(args.stop_iter == 0), **over)

    print(f"[1] training committee ({args.arch}), stopping at "
          f"{'first consensus' if args.stop_iter == 0 else f'iter {args.stop_iter}'}...",
          flush=True)
    reconstruct(teacher, dims, cfg, dev, xte[:2000], seed=args.seed, save_recon=None)

    ck = torch.load(dump, map_location=dev, weights_only=False)
    X, Y = ck["X"].to(dev), ck["Y"].to(dev)
    pop = []
    from nets import MLP
    for s in ck["pop_states"]:
        m = MLP(dims).to(dev); m.load_state_dict(s); pop.append(m)
    print(f"[2] building consensus from iter-{ck['iter']} committee "
          f"({len(X)} queries)...", flush=True)
    cons = build_consensus(pop, dims, quorum_ratio=0.625)
    if cons is None:
        print("    no consensus formed yet; try a later --stop-iter"); return
    print(f"[3] staged MSE->MAE solve on the {len(X)} queries collected so far:",
          flush=True)
    staged_solve(cons.to(dev), X, Y, teacher)
    os.remove(dump)


if __name__ == "__main__":
    main()
