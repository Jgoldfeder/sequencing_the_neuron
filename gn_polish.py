"""Gauss-Newton / Levenberg polish of an MLP on the queries ALREADY collected (no new black-box queries).

For a (leaky) ReLU MLP the output is piecewise linear in the input and piecewise multilinear in the weights; once the
guess puts (almost) every buffered query in its correct activation pattern, the least-squares problem
    min_theta  sum_i || f_theta(x_i) - y_i ||^2
over the stored (x_i, y_i) is solved by a few damped Gauss-Newton steps.  Matrix-free: J v by torch.func.jvp,
J^T v by vjp, normal equations by CG, fp64, the data streamed in chunks.  Works for tanh too (smooth).

    net, info = gn_polish(net, X, Y, iters=20, cg_iters=200, score=lambda n: param_errors(n, teacher))
"""
import math
import time

import torch
from torch.func import functional_call, jvp, vjp


def _cg(A, b, iters, tol):
    x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rr = r @ r; b0 = math.sqrt(float(rr))
    for k in range(iters):
        Ap = A(p); a = rr / (p @ Ap); x += a * p; r -= a * Ap; rn = r @ r
        if math.sqrt(float(rn)) <= tol * b0:
            return x, k + 1
        p = r + (rn / rr) * p; rr = rn
    return x, iters


def gn_polish(net, X, Y, iters=10, cg_iters=200, cg_tol=1e-6, chunk=25000, device="cuda", score=None, verbose=True,
              tag="[gn-polish]"):
    t0 = time.time()
    net = net.to(device=device, dtype=torch.float64)
    keys = [k for k, p in net.named_parameters() if p.requires_grad]
    theta = {k: p.detach().clone() for k, p in net.named_parameters() if p.requires_grad}
    Xc = [X[i:i + chunk].to(device, torch.float64) for i in range(0, len(X), chunk)]
    Yc = [Y[i:i + chunk].to(device, torch.float64) for i in range(0, len(Y), chunk)]

    def f(p, x):
        return functional_call(net, p, (x,))

    def flat(d):
        return torch.cat([d[k].reshape(-1) for k in keys])

    def unflat(v):
        out, i = {}, 0
        for k in keys:
            n = theta[k].numel(); out[k] = v[i:i + n].view_as(theta[k]); i += n
        return out

    def loss_of(p):
        with torch.no_grad():
            return sum(float(((f(p, x) - y) ** 2).sum()) for x, y in zip(Xc, Yc))

    def grad(p):                                            # J^T r
        g = None
        for x, y in zip(Xc, Yc):
            r, fn = vjp(lambda q: f(q, x) - y, p)
            gi = flat(fn(r)[0]); g = gi if g is None else g + gi
        return g

    def JtJ(p, v):
        dv = unflat(v); out = None
        for x in Xc:
            _, jv = jvp(lambda q: f(q, x), (p,), (dv,))
            _, fn = vjp(lambda q: f(q, x), p)
            o = flat(fn(jv)[0]); out = o if out is None else out + o
        return out

    def describe(label):
        if score is None or not verbose:
            return
        with torch.no_grad():
            for k in keys:
                dict(net.named_parameters())[k].copy_(theta[k])
        e = score(net)
        print(f"{tag} {label}: max_eps {e['max_eps']:.3e} | mean per-matrix "
              f"{sum(e['mean_eps_per_matrix']) / len(e['mean_eps_per_matrix']):.3e}", flush=True)

    loss = loss_of(theta); n = sum(theta[k].numel() for k in keys)
    if verbose:
        print(f"{tag} {len(X):,} stored queries (no new ones) | {n:,} parameters | init loss {loss:.3e}", flush=True)
    describe("before")
    lam = None; hist = [loss]
    for it in range(iters):
        g = grad(theta)
        if float(g.norm()) < 1e-14:
            break
        if lam is None:
            v = torch.randn(n, dtype=torch.float64, device=device)
            lam = 1e-4 * float(v @ JtJ(theta, v)) / n
        ok = False
        for _ in range(10):
            d, ncg = _cg(lambda v: JtJ(theta, v) + lam * v, -g, cg_iters, cg_tol)
            cand = {k: theta[k] + dk for k, dk in unflat(d).items()}
            new = loss_of(cand)
            if new < loss:
                theta, loss, ok = cand, new, True; lam = max(lam / 3, 1e-300); break
            lam *= 4
        hist.append(loss)
        if verbose:
            print(f"{tag} it {it + 1:2d} | loss {loss:.3e} | lam {lam:.1e} | cg {ncg} | {'ok' if ok else 'REJECT'} | "
                  f"{time.time() - t0:.0f}s", flush=True)
        if not ok:
            break
        describe(f"it {it + 1:2d}")
        if hist[-2] - hist[-1] < 1e-12 * max(hist[-2], 1e-300):
            break
    with torch.no_grad():
        for k in keys:
            dict(net.named_parameters())[k].copy_(theta[k])
    return net, dict(iters=len(hist) - 1, loss_init=hist[0], loss_final=loss, wall_s=round(time.time() - t0, 1))
