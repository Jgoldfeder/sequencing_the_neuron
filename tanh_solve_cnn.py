"""tanh_solve_cnn.py -- the joint jet refiner of tanh_solve.py for SMOOTH ConvNets (tanh LeNet etc.).

Same objective and solver as tanh_solve.refine (see its docstring): from a whole-net guess (committee consensus,
~1e-3), refine ALL parameters jointly by matching the black box's order-1 jet at query points,

    min_theta  sum_i ||f_theta(x_i) - f(x_i)||^2 + w * ||J_theta(x_i) U_i - J_f(x_i) U_i||_F^2,

with matrix-free Gauss-Newton / Levenberg damping and CG on the normal equations (torch.func jvp/vjp, fp64).
The only MLP-specific part of tanh_solve, the hand-written forward Jacobian accumulation (model_jets), is replaced by
torch.func: f_theta = functional_call(net, theta, x) and J_theta U = jvp in the input along each direction, so any
nets.ConvNet works (conv + pooling + fc, C3 masks, RBF head).  Black-box side unchanged: tanh_solve.measure_jets
(4-point central differences along m random unit directions per point; the black box is never read).

Usage (see refine_cnn_snapshot.py for the CLI):
    net, info = refine_cnn(bb, guess, X, dirs=8, iters=30, score=lambda n: cnn_param_errors(n, teacher))
"""
import math
import time

import torch
from torch.func import functional_call, jvp, vjp

from tanh_solve import Oracle, measure_jets, _cg


def refine_cnn(bb, guess, X, dirs=8, iters=30, cg_iters=150, h=1e-3, jac_weight=1.0, seed=0, score=None,
               verbose=True, tag="[tanhsolver-cnn]", cg_tol=1e-4, chunk=None):
    t0 = time.time()
    dev = X.device
    X = X.double(); N, d = X.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    U = torch.randn(N, dirs, d, generator=gen, dtype=torch.float64).to(dev)
    U = U / U.norm(dim=2, keepdim=True)
    oracle = bb if isinstance(bb, Oracle) else Oracle(bb)
    q0 = oracle.n
    Y, JU = measure_jets(oracle, X, U, h=h)                         # (N, dout), (N, dout, m)
    if verbose:
        print(f"{tag} jets: {N} points x {dirs} dirs -> {oracle.n - q0} queries ({time.time() - t0:.1f}s)", flush=True)

    net = guess.clone().to(device=dev, dtype=torch.float64)
    buffers = {k: v for k, v in net.named_buffers()}
    keys = [k for k, p in net.named_parameters() if p.requires_grad]
    theta = {k: v.detach().clone() for k, v in net.named_parameters() if v.requires_grad}
    fixed = {k: v.detach().clone() for k, v in net.named_parameters() if not v.requires_grad}
    wj = math.sqrt(jac_weight)

    def f(p, x):
        return functional_call(net, {**p, **fixed, **buffers}, (x,))

    def residual(p):
        y = f(p, X)
        jus = [jvp(lambda x: f(p, x), (X,), (U[:, j],))[1] for j in range(dirs)]   # each (N, dout)
        ju = torch.stack(jus, dim=2)
        return torch.cat([(y - Y).reshape(-1), (wj * (ju - JU)).reshape(-1)])

    def loss_of(p):
        with torch.no_grad():
            r = residual(p)
        return float(r @ r)

    def flat(dct):
        return torch.cat([dct[k].reshape(-1) for k in keys])

    def unflat(v):
        out, i = {}, 0
        for k in keys:
            n = theta[k].numel(); out[k] = v[i:i + n].view_as(theta[k]); i += n
        return out

    def describe(p, label):
        if score is None or not verbose:
            return
        with torch.no_grad():
            net.load_state_dict({**p}, strict=False)
        e = score(net)
        per = " ".join(f"{v:.1e}" for v in e.get("max_eps_per_matrix", []))
        print(f"{tag} {label}: max_eps {e['max_eps']:.3e} | max per-matrix [{per}]", flush=True)

    loss = loss_of(theta); numel = sum(theta[k].numel() for k in keys)
    if verbose:
        print(f"{tag} {numel} free parameters | init loss {loss:.3e}", flush=True)
    describe(theta, "before")
    lam = None; hist = [loss]; n_accept = 0
    for it in range(iters):
        r, vjp_fn = vjp(residual, theta)
        g = flat(vjp_fn(r)[0]); gn = float(g.norm())
        if gn < 1e-14:
            break

        def JtJ(v):
            _, jv = jvp(residual, (theta,), (unflat(v),))
            return flat(vjp_fn(jv)[0])

        if lam is None:
            v = torch.randn(numel, generator=gen, dtype=torch.float64).to(dev)
            lam = 1e-3 * float(v @ JtJ(v)) / numel
        accepted = False
        for _try in range(12):
            delta, ncg = _cg(lambda v: JtJ(v) + lam * v, -g, cg_iters, cg_tol)
            cand = {k: theta[k] + dk for k, dk in unflat(delta).items()}
            new_loss = loss_of(cand)
            pred = float(-2.0 * (g @ delta) - (delta @ JtJ(delta)))
            rho = (loss - new_loss) / pred if pred > 0 else -1.0
            if new_loss < loss and rho > 1e-3:
                theta, loss, accepted = cand, new_loss, True
                lam = max(lam / 3.0, 1e-300) if rho > 0.75 else (lam * 2.0 if rho < 0.25 else lam)
                break
            lam *= 4.0
        hist.append(loss)
        if verbose:
            print(f"{tag} it {it + 1:3d} | loss {loss:.3e} | |g| {gn:.2e} | lam {lam:.1e} | cg {ncg} | rho {rho:.2f} | "
                  f"|d| {float(delta.norm()):.2e} | {'ok' if accepted else 'REJECT'} | {time.time() - t0:.0f}s", flush=True)
        if not accepted:
            break
        n_accept += 1
        describe(theta, f"it {it + 1:3d}")
        if len(hist) > 2 and hist[-2] - hist[-1] < 1e-12 * max(hist[-2], 1e-300):
            break
    with torch.no_grad():
        net.load_state_dict({**theta}, strict=False)
    describe(theta, "after")
    return net, dict(queries=oracle.n - q0, points=N, dirs=dirs, iters=n_accept, loss_init=hist[0], loss_final=loss,
                     loss_hist=hist, wall_s=round(time.time() - t0, 1))
