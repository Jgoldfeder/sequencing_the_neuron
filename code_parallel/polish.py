"""Re-apply the endgame solvers (float64 LBFGS and/or closed-form last-layer
least-squares) to a saved reconstruction checkpoint, WITHOUT re-running the
query loop. Sweep solver knobs cheaply against a fixed reconstruction.

Prereq: produce a checkpoint first, e.g.
  python run.py --variant v18_min --arch 784,128,10 --outer 60 --window 60 \
                --device cuda --save-recon

Then, e.g.:
  python polish.py recon/v18_min__784x128x10__s0.pt --lbfgs --steps 200 --samples 30000
  python polish.py recon/v18_min__784x128x10__s0.pt --lastlayer
  python polish.py recon/v18_min__784x128x10__s0.pt --lastlayer --lbfgs
  python polish.py recon/v18_min__784x128x10__s0.pt --source popbest   # best population member
"""
import argparse

import torch

from nets import MLP
from align import param_errors
from method import polish_lbfgs, solve_last_layer_, l1_on, Cfg


def _load_net(dims, state):
    net = MLP(dims)
    net.load_state_dict(state)
    return net


def _fmt(errs):
    return (f"max_eps={errs['max_eps']:.3e}  "
            f"per-matrix=[" + ", ".join(f"{x:.2e}"
                                        for x in errs["max_eps_per_matrix"]) + "]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="path to recon/*.pt from run.py --save-recon")
    ap.add_argument("--source", choices=["best", "popbest"], default="best",
                    help="which student to polish: the saved 'best' (default) "
                         "or the lowest-loss population member ('popbest').")
    ap.add_argument("--lastlayer", action="store_true",
                    help="apply the closed-form ridge last-layer LS solve first")
    ap.add_argument("--lbfgs", action="store_true",
                    help="apply the float64 LBFGS squared-loss endgame")
    ap.add_argument("--steps", type=int, default=60,
                    help="LBFGS outer steps (default 60)")
    ap.add_argument("--samples", type=int, default=12000,
                    help="LBFGS subsample size (default 12000)")
    ap.add_argument("--ridge", type=float, default=1e-6,
                    help="ridge for the last-layer LS solve (default 1e-6)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    dims = ck["dims"]
    X, Y = ck["X"], ck["Y"]
    teacher = _load_net(dims, ck["teacher_state"])

    if args.source == "popbest":
        pop = [_load_net(dims, s) for s in ck["pop_states"]]
        losses = l1_on(pop, X, Y)
        bi = min(range(len(pop)), key=lambda i: losses[i])
        net = pop[bi]
        print(f"[source] popbest = member {bi} (train L1 {losses[bi]:.3e})")
    else:
        net = _load_net(dims, ck["best_state"])
        print("[source] saved 'best' (pre-endgame)")

    print(f"[ckpt]  {args.ckpt}  dims={dims}  queries={ck.get('queries','?')}  "
          f"samples={len(X)}")
    print(f"[before] {_fmt(param_errors(net, teacher))}")

    if args.lastlayer:
        solve_last_layer_(net, X, Y, ridge=args.ridge)
        print(f"[+lastlayer ridge={args.ridge:g}] "
              f"{_fmt(param_errors(net, teacher))}")

    if args.lbfgs:
        cfg = Cfg(**{k: ck["cfg"][k] for k in ck["cfg"] if k in Cfg.__dataclass_fields__})
        net = polish_lbfgs(net, X, Y, cfg, max_samples=args.samples,
                           steps=args.steps)
        print(f"[+lbfgs steps={args.steps} samples={args.samples}] "
              f"{_fmt(param_errors(net, teacher))}")

    if not (args.lastlayer or args.lbfgs):
        print("(no solver selected; pass --lbfgs and/or --lastlayer)")


if __name__ == "__main__":
    main()
