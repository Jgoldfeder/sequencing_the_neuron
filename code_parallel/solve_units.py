"""Agnostic Gauss-Seidel endgame solver (one hidden unit at a time).

Starting from a rough reconstruction (consensus/best member), sweep the hidden
units ONE AT A TIME. For each unit u, holding all others fixed at their CURRENT
value, isolate u's activation from the running residual and re-solve its input
row in closed form; then immediately fold the update back into the residual
before moving to the next unit (Gauss-Seidel -> no double-counting, so it is
stable where the all-at-once Jacobi update diverged). A last-layer LSQ closes
each sweep. Uses ONLY teacher query outputs Y (never teacher weights).

Speed: the (subsampled) query matrix lives resident on the GPU, so each unit is
two matvecs + a back-substitution against a shared, pre-factored X^T X. A full
1024-unit sweep is seconds, not a host<->device stream per unit.

  per unit u:
    a_u(x) = (resid(x)·w_out[:,u]) / ||w_out[:,u]||^2 + h_u(x)     ~= sigma(z_u)
    z_u    = sigma^{-1}(a_u)
    [w_in_u, b_u] = (X^T X)^{-1} X^T z_u        (shared Cholesky factor)
    resid -= w_out[:,u] * (sigma(z_u_new) - h_u_old)   # fold back immediately
"""
import argparse
import os
import time

import torch

from nets import MLP
from method import build_consensus, l1_on
from align import param_errors

ALPHA = 0.01


def leaky(z):
    return torch.where(z >= 0, z, ALPHA * z)


def inv_leaky(y):
    return torch.where(y >= 0, y, y / ALPHA)


def load_net(dims, state, dev):
    m = MLP(dims).to(dev); m.load_state_dict(state); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sweeps", type=int, default=4)
    ap.add_argument("--samples", type=int, default=250000,
                    help="query subsample kept resident on GPU")
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--start", choices=["consensus", "member"], default="consensus")
    args = ap.parse_args()
    dev = args.device

    print(f"[load] {args.dump}", flush=True)
    ck = torch.load(args.dump, map_location="cpu", weights_only=False)
    dims = ck["dims"]; Xc, Yc = ck["X"], ck["Y"]; N = len(Xc)
    D, H, O = dims[0], dims[1], dims[2]
    teacher = load_net(dims, ck["teacher_state"], dev)
    pop = [load_net(dims, s, dev) for s in ck["pop_states"]]

    if args.start == "consensus":
        net = build_consensus(pop, dims, quorum_ratio=0.625).to(dev)
    else:
        losses = l1_on(pop, Xc[:100000], Yc[:100000])
        net = pop[min(range(len(pop)), key=lambda i: losses[i])].clone()
    print(f"[start] {args.start}: max_eps={param_errors(net, teacher)['max_eps']:.3e}",
          flush=True)

    # ---- resident subsample on GPU ----
    S = min(args.samples, N)
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(0))[:S]
    Yg = Yc[idx].to(dev)                                  # (S, O)
    ones = torch.ones(S, 1, device=dev)
    # build [X | 1] in place (chunked) so we never hold two S x D copies at once
    Xa = torch.empty(S, D + 1, device=dev)
    Xa[:, D] = 1.0
    for i in range(0, S, 50000):
        j = min(i + 50000, S)
        Xa[i:j, :D] = Xc[idx[i:j]].to(dev)
    print(f"[gpu] resident query matrix {tuple(Xa.shape)} "
          f"({Xa.element_size()*Xa.nelement()/1024**3:.1f} GB)", flush=True)

    XtX = (Xa.T @ Xa).double()
    XtX += args.ridge * XtX.diag().mean() * torch.eye(D + 1, device=dev, dtype=torch.float64)
    L = torch.linalg.cholesky(XtX)

    # ---- running state ----
    Wf = torch.cat([net.layers[0].weight.data,
                    net.layers[0].bias.data[:, None]], 1).clone()   # (H, D+1)
    Wout = net.layers[1].weight.data.clone()                        # (O, H)
    bout = net.layers[1].bias.data.clone()                          # (O,)
    Hh = leaky(Xa @ Wf.T)                                           # (S, H)
    resid = Yg - (Hh @ Wout.T + bout)                               # (S, O)

    def push_and_report(tag):
        net.layers[0].weight.data.copy_(Wf[:, :D])
        net.layers[0].bias.data.copy_(Wf[:, D])
        net.layers[1].weight.data.copy_(Wout)
        net.layers[1].bias.data.copy_(bout)
        e = param_errors(net, teacher)["max_eps"]
        l1 = resid.abs().mean().item()
        print(f"  [{tag}] max_eps={e:.3e}  train_L1(sub)={l1:.3e}", flush=True)

    for s in range(args.sweeps):
        t0 = time.time()
        # (B) Gauss-Seidel over input rows, with a monotone guard: accept a
        # unit's re-solve only if it lowers the residual energy (inv_leaky
        # amplifies noise 1/alpha on negative-region samples, so an unguarded
        # update can produce a huge row and cascade to nan).
        naccept = 0
        for u in range(H):
            wo = Wout[:, u]
            n2 = wo.dot(wo).clamp_min(1e-12)
            a_u = (resid @ wo) / n2 + Hh[:, u]             # ~ sigma(z_u)
            ztar = inv_leaky(a_u)
            rhs = (Xa.T @ ztar).double()
            w_new = torch.cholesky_solve(rhs[:, None], L)[:, 0].float()
            h_new = leaky(Xa @ w_new)
            delta = (h_new - Hh[:, u])[:, None] * wo[None, :]   # change to Sout
            de = delta.pow(2).sum() - 2.0 * (resid * delta).sum()  # d||resid||^2
            if torch.isfinite(de) and de < 0:
                resid -= delta
                Hh[:, u] = h_new
                Wf[u] = w_new
                naccept += 1
        print(f"  [sweep {s+1}] accepted {naccept}/{H} unit updates", flush=True)
        push_and_report(f"sweep {s+1} input ({time.time()-t0:.0f}s)")
        # (A) last-layer closed-form ridge LSQ (given current activations).
        # Hidden activations are rank-deficient (near-constant columns), so use a
        # firm ridge and fall back to keeping the current output layer.
        Ha = torch.cat([Hh, ones], 1)                     # (S, H+1)
        A = (Ha.T @ Ha).double()
        A += 1e-4 * A.diag().mean().clamp_min(1e-12) * torch.eye(
            H + 1, device=dev, dtype=torch.float64)
        try:
            Wsol = torch.linalg.solve(A, (Ha.T @ Yg).double()).float()  # (H+1, O)
            Wout = Wsol[:H].T.contiguous()
            bout = Wsol[H]
            resid = Yg - (Hh @ Wout.T + bout)
            push_and_report(f"sweep {s+1} +LL")
        except Exception as e:
            print(f"  [sweep {s+1}] last-layer solve skipped ({e})", flush=True)

    out = os.path.join(os.path.dirname(args.dump), "solved_units.pt")
    torch.save({"dims": dims,
                "state": {k: v.cpu() for k, v in net.state_dict().items()},
                "max_eps": param_errors(net, teacher)["max_eps"]}, out)
    print(f"[save] -> {out}", flush=True)


if __name__ == "__main__":
    main()
