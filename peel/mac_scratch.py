"""mac_scratch.py -- FROM-SCRATCH (cold, random init) first-layer recovery,
population + disagreement query generation, plain-Adam members vs MAC members.

The standalone I/O harness (mac_lift.py) showed lifting ARRESTS drift but cannot
COLD-RECOVER L1 on its own -- because recovery is an identifiability problem that
needs the recovery signal, not just a better optimizer. Here we add the actual
recovery signal used by the real pipeline (method.reconstruct): adversarial
queries that MAXIMISE cross-member output disagreement. Members trained to agree
on those converge to the canonical (true) solution.

Question: from random init, does MAC member-training + disagreement recover L1,
and does it beat plain-Adam member-training on the saturating (hard) regime where
plain Adam drifts L1?

Two hidden layers (matches 3072->1024->512->100 and the MAC-1/MAC-2 derivation).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mac_lift import (build_teacher, aug, chol_factor, solve_layer,   # noqa: E402
                      l1_error, l1_consensus)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nets import MLP                                                   # noqa: E402


# --------------------------------------------------------------------------- #
# recovery signal: adversarial disagreement queries
# --------------------------------------------------------------------------- #
def disagree_queries(nets, q, d_in, device, gen, steps=30, lr=0.1, init_std=0.5):
    Z = torch.randn(q, d_in, generator=gen, device=device) * init_std
    Z.requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=lr)
    for s in range(steps):
        outs = torch.stack([net(Z) for net in nets])          # (P, q, out)
        loss = -outs.var(dim=0).mean()                        # maximise disagreement
        opt.zero_grad(); loss.backward(); opt.step()
    return Z.detach()


# --------------------------------------------------------------------------- #
# members
# --------------------------------------------------------------------------- #
class BaselineMember:
    def __init__(self, dims, seed, device, lr):
        torch.manual_seed(seed)
        self.net = MLP(dims, act="sigmoid").to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, X, Y, epochs, batch, gen):
        n = len(X)
        for _ in range(epochs):
            perm = torch.randperm(n, generator=gen, device=X.device)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                self.opt.zero_grad()
                ((self.net(X[idx]) - Y[idx]) ** 2).mean().backward()
                self.opt.step()


class MacMember:
    """MAC-2: lift Z1,Z2 (persistent, appended as the query pool grows);
    all three weight matrices by dual-shifted linear solves."""
    def __init__(self, dims, seed, device):
        torch.manual_seed(seed)
        self.net = MLP(dims, act="sigmoid").to(device)
        h1, h2 = dims[1], dims[2]
        self.Z1 = torch.empty(0, h1, device=device)
        self.Z2 = torch.empty(0, h2, device=device)
        self.L1 = torch.empty(0, h1, device=device)
        self.L2 = torch.empty(0, h2, device=device)

    @torch.no_grad()
    def append(self, Xnew):
        L = self.net.layers
        z1 = Xnew @ L[0].weight.T + L[0].bias
        z2 = torch.sigmoid(z1) @ L[1].weight.T + L[1].bias
        self.Z1 = torch.cat([self.Z1, z1]); self.Z2 = torch.cat([self.Z2, z2])
        self.L1 = torch.cat([self.L1, torch.zeros_like(z1)])
        self.L2 = torch.cat([self.L2, torch.zeros_like(z2)])

    def step(self, X, Y, cholX, K, rho1, rho2, z_steps, z_lr, ridge):
        L = self.net.layers
        W1, b1 = L[0].weight, L[0].bias
        W2, b2 = L[1].weight, L[1].bias
        W3, b3 = L[2].weight, L[2].bias
        Xa = aug(X)
        for _ in range(K):
            Z1 = self.Z1.clone().requires_grad_(True)
            Z2 = self.Z2.clone().requires_grad_(True)
            zopt = torch.optim.Adam([Z1, Z2], lr=z_lr)
            for _ in range(z_steps):
                H1 = torch.sigmoid(Z1)
                yhat = torch.sigmoid(Z2) @ W3.T.detach() + b3.detach()
                R1 = Z1 - (X @ W1.T.detach() + b1.detach())
                R2 = Z2 - (H1 @ W2.T.detach() + b2.detach())
                loss = (((yhat - Y) ** 2).mean()
                        + (self.L1 * R1).mean() + 0.5 * rho1 * (R1 ** 2).mean()
                        + (self.L2 * R2).mean() + 0.5 * rho2 * (R2 ** 2).mean())
                zopt.zero_grad(); loss.backward(); zopt.step()
            self.Z1, self.Z2 = Z1.detach(), Z2.detach()
            with torch.no_grad():
                W1n, b1n = solve_layer(Xa, self.Z1 + self.L1 / rho1, cholX)
                W1.copy_(W1n); b1.copy_(b1n)
                Ha = aug(torch.sigmoid(self.Z1)); cholH = chol_factor(Ha, ridge)
                W2n, b2n = solve_layer(Ha, self.Z2 + self.L2 / rho2, cholH)
                W2.copy_(W2n); b2.copy_(b2n)
                F2 = aug(torch.sigmoid(self.Z2)); cholF = chol_factor(F2, ridge)
                W3n, b3n = solve_layer(F2, Y, cholF)
                W3.copy_(W3n); b3.copy_(b3n)
                self.L1 += rho1 * (self.Z1 - (X @ W1.T + b1))
                self.L2 += rho2 * (self.Z2 - (torch.sigmoid(self.Z1) @ W2.T + b2))


# --------------------------------------------------------------------------- #
# from-scratch driver
# --------------------------------------------------------------------------- #
def run_scratch(solver, teacher, dims, args, device):
    gen = torch.Generator(device=device).manual_seed(123)
    if solver == "baseline":
        pop = [BaselineMember(dims, 10 + m, device, args.lr) for m in range(args.P)]
    else:
        pop = [MacMember(dims, 10 + m, device) for m in range(args.P)]
    nets = [m.net for m in pop]
    Xall = Yall = None
    traj = []
    for o in range(args.outer):
        if o < args.warm_iters:
            Xq = torch.randn(args.q, dims[0], generator=gen, device=device) * 0.5
        else:
            Xq = disagree_queries(nets, args.q, dims[0], device, gen,
                                  steps=args.qg_steps, lr=args.qg_lr)
        with torch.no_grad():
            Yq = teacher(Xq)
        Xall = Xq if Xall is None else torch.cat([Xall, Xq])
        Yall = Yq if Yall is None else torch.cat([Yall, Yq])
        if solver == "baseline":
            for m in pop:
                m.fit(Xall, Yall, args.epochs, args.batch, gen)
        else:
            cholX = chol_factor(aug(Xall), args.ridge)
            for m in pop:
                m.append(Xq)
                m.step(Xall, Yall, cholX, args.K, args.rho1,
                       args.alpha * args.rho1, args.z_steps, args.z_lr, args.ridge)
        if o % args.log_every == 0 or o == args.outer - 1:
            with torch.no_grad():
                tl = sum(((m.net(Xall) - Yall) ** 2).mean().item() for m in pop) / args.P
            errs = [l1_error(m.net, teacher)["rel"] for m in pop]
            cons = l1_consensus(nets, teacher)
            traj.append((o, tl, sum(errs) / len(errs), min(errs), cons["cons_rel"]))
            print(f"  [{solver:<8}] o{o:>3}  npool {len(Xall):>6}  tMSE {tl:.2e}  "
                  f"L1rel mem {sum(errs)/len(errs)*100:5.1f}% (best {min(errs)*100:5.1f}%)  "
                  f"cons {cons['cons_rel']*100:5.1f}%", flush=True)
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="256,128,64,20")
    ap.add_argument("--wscale", type=float, default=2.5)
    ap.add_argument("--P", type=int, default=5)
    ap.add_argument("--outer", type=int, default=40)
    ap.add_argument("--warm_iters", type=int, default=3)
    ap.add_argument("--q", type=int, default=1000)
    ap.add_argument("--qg_steps", type=int, default=30)
    ap.add_argument("--qg_lr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=6)      # baseline epochs / outer
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--K", type=int, default=3)           # MAC block iters / outer
    ap.add_argument("--rho1", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.3)   # rho2/rho1
    ap.add_argument("--z_steps", type=int, default=40)
    ap.add_argument("--z_lr", type=float, default=0.05)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--solvers", default="baseline,mac2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    dims = [int(x) for x in args.dims.split(",")]
    assert len(dims) == 4
    teacher = build_teacher(dims, args.seed, args.wscale, dev)
    with torch.no_grad():
        xs = torch.randn(2000, dims[0], device=dev)
        z1 = xs @ teacher.layers[0].weight.T + teacher.layers[0].bias
    print(f"[teacher] dims={dims} wscale={args.wscale} preact-std z1={z1.std():.2f} "
          f"(cold/from-scratch; alpha={args.alpha})\n")

    for solver in args.solvers.split(","):
        print(f"=== {solver} ===")
        run_scratch(solver, teacher, dims, args, dev)
        print()


if __name__ == "__main__":
    main()
