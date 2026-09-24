"""fact_scratch.py -- FROM-SCRATCH L1 recovery with SQUARE-FACTORIZED weights
(exact reparam, no nonlinearity between factors) under ordinary Adam + the
disagreement recovery signal.

W_l = A_l B_l with A_l square (out x out), B_l (out x in). Since A_l is full-rank
capable, {A_l B_l} = {W_l} exactly -- identical function class at every step. The
factorization only changes the optimizer geometry (implicit left/right
preconditioner dW ~ -eta (AA^T G + G B^T B)). Crucially it is a PER-MEMBER reparam
under stochastic Adam, so members stay DIVERSE -> the disagreement engine keeps
working (unlike MAC which homogenised the population).

Variants:
  baseline   : all layers ordinary W.
  fact_tail  : W1 ordinary (the object we recover), tail W2,W3 factorized. <- Judah's pick
  fact_all   : every layer factorized.

Consensus/scoring always collapses factors -> materialize() builds a plain MLP
with W_eff = A B per layer, then the usual sigmoid-gauge L1 metrics apply.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mac_lift import build_teacher, l1_error, l1_consensus            # noqa: E402
from mac_scratch import disagree_queries                              # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nets import MLP                                                   # noqa: E402


class FactMember:
    def __init__(self, dims, seed, device, fact_layers, lr):
        torch.manual_seed(seed)
        self.dims = dims
        self.device = device
        self.layers = []          # each: ('plain', W, b) or ('fact', A, B, b)
        params = []
        for l in range(len(dims) - 1):
            out, inp = dims[l + 1], dims[l]
            std = (2.0 / (inp + out)) ** 0.5
            b = torch.zeros(out, device=device, requires_grad=True)
            if l in fact_layers:
                B = (torch.randn(out, inp, device=device) * std).requires_grad_(True)
                A = torch.eye(out, device=device).clone().requires_grad_(True)
                self.layers.append(("fact", A, B, b)); params += [A, B, b]
            else:
                W = (torch.randn(out, inp, device=device) * std).requires_grad_(True)
                self.layers.append(("plain", W, b)); params += [W, b]
        self.params = params
        self.opt = torch.optim.Adam(params, lr=lr)

    def weff(self, l):
        L = self.layers[l]
        return (L[1] @ L[2]) if L[0] == "fact" else L[1]

    def bias(self, l):
        return self.layers[l][-1]

    def __call__(self, x):
        for l in range(len(self.layers) - 1):
            x = torch.sigmoid(x @ self.weff(l).T + self.bias(l))
        L = len(self.layers) - 1
        return x @ self.weff(L).T + self.bias(L)

    @torch.no_grad()
    def materialize(self):
        net = MLP(self.dims, act="sigmoid").to(self.device)
        for l in range(len(self.layers)):
            net.layers[l].weight.copy_(self.weff(l).detach())
            net.layers[l].bias.copy_(self.bias(l).detach())
        return net

    def fit(self, X, Y, epochs, batch, gen):
        n = len(X)
        for _ in range(epochs):
            perm = torch.randperm(n, generator=gen, device=X.device)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                self.opt.zero_grad()
                ((self(X[idx]) - Y[idx]) ** 2).mean().backward()
                self.opt.step()


FACT = {"baseline": [], "fact_tail": [1, 2], "fact_all": [0, 1, 2]}


def run(variant, teacher, dims, args, device):
    gen = torch.Generator(device=device).manual_seed(123)
    pop = [FactMember(dims, 10 + m, device, FACT[variant], args.lr)
           for m in range(args.P)]
    Xall = Yall = None
    for o in range(args.outer):
        if o < args.warm_iters:
            Xq = torch.randn(args.q, dims[0], generator=gen, device=device) * 0.5
        else:
            Xq = disagree_queries(pop, args.q, dims[0], device, gen,
                                  steps=args.qg_steps, lr=args.qg_lr)
        with torch.no_grad():
            Yq = teacher(Xq)
        Xall = Xq if Xall is None else torch.cat([Xall, Xq])
        Yall = Yq if Yall is None else torch.cat([Yall, Yq])
        for m in pop:
            m.fit(Xall, Yall, args.epochs, args.batch, gen)
        if o % args.log_every == 0 or o == args.outer - 1:
            nets = [m.materialize() for m in pop]
            with torch.no_grad():
                tl = sum(((m(Xall) - Yall) ** 2).mean().item() for m in pop) / args.P
            errs = [l1_error(n, teacher)["rel"] for n in nets]
            cons = l1_consensus(nets, teacher)
            print(f"  [{variant:<9}] o{o:>3}  tMSE {tl:.2e}  "
                  f"L1rel mem {sum(errs)/len(errs)*100:5.1f}% "
                  f"(best {min(errs)*100:5.1f}%)  cons {cons['cons_rel']*100:5.1f}%",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="512,256,128,40")
    ap.add_argument("--wscale", type=float, default=4.0)
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--outer", type=int, default=60)
    ap.add_argument("--warm_iters", type=int, default=4)
    ap.add_argument("--q", type=int, default=1500)
    ap.add_argument("--qg_steps", type=int, default=30)
    ap.add_argument("--qg_lr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--variants", default="baseline,fact_tail,fact_all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    dims = [int(x) for x in args.dims.split(",")]
    teacher = build_teacher(dims, args.seed, args.wscale, dev)
    with torch.no_grad():
        z1 = (torch.randn(2000, dims[0], device=dev)
              @ teacher.layers[0].weight.T + teacher.layers[0].bias)
    print(f"[teacher] dims={dims} wscale={args.wscale} z1-preact-std {z1.std():.2f} "
          f"(from scratch + disagreement)\n")
    for v in args.variants.split(","):
        print(f"=== {v} ===")
        run(v, teacher, dims, args, dev)
        print()


if __name__ == "__main__":
    main()
