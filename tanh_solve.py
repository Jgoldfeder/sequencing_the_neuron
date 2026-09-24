"""tanh_solve.py -- joint jet refiner for SMOOTH (tanh / sigmoid) MLPs.

Given a black box f (queryable forward pass only) and a good whole-net guess
(every layer within ~1e-2), refine ALL parameters jointly by matching the
black box's order-1 jet -- value + input-Jacobian -- at a set of query points:

    min_theta  sum_i ||f_theta(x_i) - f(x_i)||^2 + ||J_theta(x_i) U_i - J_f(x_i) U_i||_F^2

Why the jet and not just values: value-only distillation floors around 1e-2
(the committee plateau) because the loss is nearly flat along the network's
deep parameter directions; the Jacobian rows are where those directions are
excited. Why a smooth net needs this at all: there are no kinks, so the exact
piecewise-linear refiners (verify_layer1 / kink_solve) do not apply, and a
smooth refiner is a nonlinear least squares. Its gauge is discrete only
(per-neuron sign flip + permutation), fixed by the guess, so the Gauss-Newton
normal matrix is nonsingular and the guess is the only correspondence needed.

Solver: matrix-free Gauss-Newton with Levenberg-Marquardt damping, normal
equations solved by CG using torch.func jvp/vjp (never forms J). fp64.

Black-box discipline: the solver sees ONLY `bb(x)`; teacher parameters are
never read. The Jacobian of the black box is measured by a 4-point central
finite-difference stencil (O(h^4)) along either every input coordinate
(dirs=0 -> full Jacobian, 4*d queries per point) or m random unit directions
per point (dirs=m -> 4*m queries per point; the union over points still spans
the full input space). The optional `score` callback is for DIAGNOSTICS only
(prints alignment eps vs the teacher); it does not influence the solve.
"""
import math
import time

import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp, vjp

from nets import MLP


class Oracle:
    """Query-counting wrapper around a black-box forward callable."""

    def __init__(self, fn, chunk=65536):
        self.fn, self.chunk, self.n = fn, chunk, 0

    @torch.no_grad()
    def __call__(self, X):
        self.n += len(X)
        out = [self.fn(X[i:i + self.chunk]) for i in range(0, len(X), self.chunk)]
        return torch.cat(out, 0)


# ----------------------------------------------------------------------------
# jet measurement (black-box side)
# ----------------------------------------------------------------------------
@torch.no_grad()
def measure_jets(bb, X, U=None, h=1e-3):
    """Order-1 jet of the black box at rows of X (N, d), fp64.
    U: (N, m, d) unit directions, or None for the full Jacobian (m = d).
    Returns Y (N, dout) and JU (N, dout, m) = J_f(x_i) @ U_i^T, via the 4-point
    stencil f'(x) ~ [8(f(x+h)-f(x-h)) - (f(x+2h)-f(x-2h))] / (12h)."""
    N, d = X.shape
    Y = bb(X)
    dout = Y.shape[1]
    m = d if U is None else U.shape[1]
    JU = torch.empty(N, dout, m, dtype=X.dtype, device=X.device)
    # block over directions so the (N, B, d) probe tensor stays ~<=256MB
    B = max(1, min(m, int(3.2e7 // (N * d))))
    for j0 in range(0, m, B):
        j1 = min(m, j0 + B)
        if U is None:
            V = torch.zeros(j1 - j0, d, dtype=X.dtype, device=X.device)
            V[torch.arange(j1 - j0), torch.arange(j0, j1)] = 1.0
            V = V[None].expand(N, -1, -1)
        else:
            V = U[:, j0:j1]
        Xb = X[:, None, :]
        f = [bb((Xb + s * h * V).reshape(-1, d)).reshape(N, j1 - j0, dout)
             for s in (1.0, -1.0, 2.0, -2.0)]
        D = (8.0 * (f[0] - f[1]) - (f[2] - f[3])) / (12.0 * h)   # (N, B, dout)
        JU[:, :, j0:j1] = D.transpose(1, 2)
    return Y, JU


# ----------------------------------------------------------------------------
# model jets (differentiable in the parameters)
# ----------------------------------------------------------------------------
def _act_and_deriv(z, act):
    if act == "tanh":
        a = torch.tanh(z)
        return a, 1.0 - a * a
    if act == "sigmoid":
        a = torch.sigmoid(z)
        return a, a * (1.0 - a)
    raise ValueError(f"tanh_solve supports smooth activations only, got {act!r}")


def model_jets(params, X, U, act, n_layers):
    """f_theta(X) and J_theta(X) U^T for an MLP with parameters `params`
    (functional dict 'layers.i.weight' / 'layers.i.bias'). Forward-accumulates
    the directional Jacobian: G_0 = U^T (N, d, m); G_l = D_l (W_l G_{l-1}).
    U=None -> full Jacobian (G_0 = I). Returns (Y (N,dout), JU (N,dout,m))."""
    a = X
    G = None if U is None else U.transpose(1, 2)            # (N, d, m)
    for l in range(n_layers):
        W, b = params[f"layers.{l}.weight"], params[f"layers.{l}.bias"]
        z = a @ W.t() + b
        G = W[None] if G is None else W @ G                  # (N, out, m)
        if G.dim() == 2:
            G = G[None].expand(len(X), -1, -1)
        if l < n_layers - 1:
            a, D = _act_and_deriv(z, act)
            G = D[:, :, None] * G
        else:
            a = z
    return a, G


# ----------------------------------------------------------------------------
# the solve
# ----------------------------------------------------------------------------
def _cg(Aop, b, iters, tol, Minv=None):
    """(Preconditioned) CG on A x = b (A SPD), x0 = 0. Minv: diagonal
    preconditioner (vector) or None. Returns (x, n_iter)."""
    x = torch.zeros_like(b)
    r = b.clone()
    z = r if Minv is None else Minv * r
    p = z.clone(); rz = r @ z
    b_norm = math.sqrt(float(r @ r))
    for k in range(iters):
        Ap = Aop(p)
        alpha = rz / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        if math.sqrt(float(r @ r)) <= tol * b_norm:
            return x, k + 1
        z = r if Minv is None else Minv * r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, iters


def _row_groups(theta, keys):
    """Group id per flat parameter entry: hidden/output neuron j of layer l
    owns row j of layers.l.weight AND layers.l.bias[j]. Used to average the
    noisy Hutchinson diag(J^T J) estimate into a per-neuron scale."""
    ids, base = [], 0
    for k in keys:
        t = theta[k]
        l = int(k.split(".")[1])
        n_out = t.shape[0]
        g = torch.arange(n_out, device=t.device)
        if t.dim() == 2:
            g = g[:, None].expand(-1, t.shape[1]).reshape(-1)
        ids.append(g + sum(theta[f"layers.{i}.weight"].shape[0] for i in range(l)))
    return torch.cat(ids)


def _diag_estimate(JtJ, groups, n_groups, numel, gen, dev, probes=8):
    """Per-neuron mean of diag(J^T J) by Rademacher Hutchinson, averaged
    within each row group (low variance: each group has ~in_features entries)."""
    acc = torch.zeros(numel, dtype=torch.float64, device=dev)
    for _ in range(probes):
        v = torch.randint(0, 2, (numel,), generator=gen).double().mul_(2).sub_(1).to(dev)
        acc += JtJ(v) * v
    acc /= probes
    gsum = torch.zeros(n_groups, dtype=torch.float64, device=dev).index_add_(0, groups, acc)
    gcnt = torch.zeros(n_groups, dtype=torch.float64, device=dev).index_add_(
        0, groups, torch.ones_like(acc))
    d = (gsum / gcnt)[groups]
    return d.clamp_min(1e-12 * float(d.max()))


def refine(bb, guess, X, act, dirs=0, iters=40, cg_iters=200, h=1e-3,
           jac_weight=1.0, seed=0, score=None, verbose=True, tag="[tanhsolver]",
           precond=False, cg_tol=1e-3):
    """Jointly refine every parameter of `guess` (MLP, any dtype) against the
    black box `bb` on query points X (N, d). Returns (net_fp64, info).

    dirs: 0 = full Jacobian (4*d queries/point); m>0 = m random directions per
    point (4*m queries/point). score: optional callable(net)->dict for
    diagnostics only (e.g. lambda n: param_errors(n, teacher))."""
    t0 = time.time()
    dev = X.device
    X = X.double()
    N, d = X.shape
    dims = list(guess.dims)
    n_layers = len(dims) - 1
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if dirs and dirs < d:
        U = torch.randn(N, dirs, d, generator=gen, dtype=torch.float64).to(dev)
        U = U / U.norm(dim=2, keepdim=True)
    else:
        U, dirs = None, d

    oracle = bb if isinstance(bb, Oracle) else Oracle(bb)
    q0 = oracle.n
    Y, JU = measure_jets(oracle, X, U, h=h)
    n_meas = oracle.n - q0
    if verbose:
        print(f"{tag} jets: {N} points x {dirs} dirs -> {n_meas} queries "
              f"({time.time() - t0:.1f}s); residual rows {N * Y.shape[1] * (1 + dirs)}",
              flush=True)

    net = MLP(dims, act=act).to(device=dev, dtype=torch.float64)
    net.load_state_dict({k: v.detach().to(dev, torch.float64)
                         for k, v in guess.state_dict().items()})
    keys = [k for k, _ in net.named_parameters()]
    theta = {k: v.detach().clone() for k, v in net.named_parameters()}
    wj = math.sqrt(jac_weight)

    def residual(p):
        y, ju = model_jets(p, X, U, act, n_layers)
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
            n = theta[k].numel()
            out[k] = v[i:i + n].view_as(theta[k]); i += n
        return out

    def describe(p, label):
        if score is None or not verbose:
            return
        with torch.no_grad():
            net.load_state_dict({k: p[k] for k in keys}, strict=False)
        e = score(net)
        per = e.get("max_eps_per_matrix", [])
        per_s = " ".join(f"{v:.1e}" for v in per)
        print(f"{tag} {label}: max_eps {e['max_eps']:.3e} | per-matrix [{per_s}]",
              flush=True)

    loss = loss_of(theta)
    if verbose:
        print(f"{tag} init loss {loss:.3e}", flush=True)
    describe(theta, "before")

    groups = _row_groups(theta, keys)
    n_groups = int(groups.max()) + 1
    numel = groups.numel()
    lam = None
    hist = [loss]
    n_accept = 0
    for it in range(iters):
        r, vjp_fn = vjp(residual, theta)
        g = flat(vjp_fn(r)[0])                              # J^T r
        gn = float(g.norm())
        if gn < 1e-14:
            break

        def JtJ(v):
            dv = unflat(v)
            _, jv = jvp(residual, (theta,), (dv,))
            return flat(vjp_fn(jv)[0])

        # Damping is lam*I (Levenberg), NOT Marquardt's lam*diag(J^T J): D-scaled
        # damping leaves weakly-excited parameters (saturated sigmoid neurons)
        # nearly undamped and the first step jumped somewhere GN could not
        # leave. The per-neuron Jacobi preconditioner (precond=True, or "auto"
        # = only when the diag estimate's spread is < 1e4) is OFF by default:
        # on the deep tanh net plain CG and PCG reach the same FD-noise loss
        # floor (~3e-18) and the same 8e-13 eps (the deepest layers' error at
        # that floor varies 1e-12..1e-10 between runs -- weakly determined
        # directions, not the solver); on the deep sigmoid net the Hutchinson
        # estimate is garbage on the saturated groups (diag spans 1e12) and
        # PCG HURTS (first step 47 -> 1.6 vs 0.17 plain). The 8-probe
        # estimate is also too noisy for "auto" (a clamped group reads as
        # spread 1e12 even on tanh). A tight CG tolerance matters far more.
        if lam is None:                                      # scale: mean diag(J^T J)
            v = torch.randn(numel, generator=gen, dtype=torch.float64).to(dev)
            lam = 1e-3 * float(v @ JtJ(v)) / numel
        Minv = None
        if precond:
            D = _diag_estimate(JtJ, groups, n_groups, numel, gen, dev)
            spread = float(D.max() / D.min())
            use = (spread < 1e4) if precond == "auto" else True
            if verbose and it == 0:
                print(f"{tag} diag(J^T J) per-neuron spread {spread:.1e} -> "
                      f"{'Jacobi-PCG' if use else 'plain CG'}", flush=True)
            if use:
                Minv = 1.0 / (D + lam)
        accepted = False
        for _try in range(12):
            delta, ncg = _cg(lambda v: JtJ(v) + lam * v, -g, cg_iters, cg_tol, Minv)
            cand = {k: theta[k] + dk for k, dk in unflat(delta).items()}
            new_loss = loss_of(cand)
            # gain ratio: actual / predicted (Gauss-Newton model) reduction
            pred = float(-2.0 * (g @ delta) - (delta @ JtJ(delta)))   # loss = ||r||^2
            rho = (loss - new_loss) / pred if pred > 0 else -1.0
            if new_loss < loss and rho > 1e-3:
                theta, loss, accepted = cand, new_loss, True
                if rho > 0.75:
                    lam = max(lam / 3.0, 1e-300)
                elif rho < 0.25:
                    lam *= 2.0
                break
            lam *= 4.0
        hist.append(loss)
        if verbose:
            print(f"{tag} it {it + 1:3d} | loss {loss:.3e} | |g| {gn:.2e} | "
                  f"lam {lam:.1e} | cg {ncg} | rho {rho:.2f} | |d| {float(delta.norm()):.2e} | "
                  f"{'ok' if accepted else 'REJECT'} | "
                  f"{time.time() - t0:.0f}s", flush=True)
        if not accepted:
            break
        n_accept += 1
        if len(hist) > 2 and hist[-2] - hist[-1] < 1e-12 * max(hist[-2], 1e-300):
            break

    with torch.no_grad():
        net.load_state_dict({k: theta[k] for k in keys}, strict=False)
    describe(theta, "after")
    info = dict(queries=oracle.n - q0, points=N, dirs=dirs, iters=n_accept,
                loss_init=hist[0], loss_final=loss, loss_hist=hist,
                wall_s=round(time.time() - t0, 1))
    return net, info


def sample_points(n, d, scales=(0.5, 1.0, 2.0), pool=None, seed=0, device="cpu"):
    """Query points for the jet fit: fresh Gaussian inputs at each scale in
    `scales` (cycled), optionally with half the points drawn from `pool`
    (e.g. the run's collected disagreement queries, which live where the
    committee found the function hard)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    parts = []
    if pool is not None and len(pool) > 0:
        k = min(n // 2, len(pool))
        idx = torch.randperm(len(pool), generator=gen)[:k]
        parts.append(pool[idx].double().cpu())
        n -= k
    G = torch.randn(n, d, generator=gen, dtype=torch.float64)
    s = torch.tensor([scales[i % len(scales)] for i in range(n)], dtype=torch.float64)
    parts.append(G * s[:, None])
    return torch.cat(parts, 0).to(device)
