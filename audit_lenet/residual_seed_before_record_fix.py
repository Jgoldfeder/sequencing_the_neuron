"""Residual-Jacobian seeds for the committee's sampler (honest, black-box; 2026-10-21).

Problem: frontier units the teacher never activates on the query distribution ("silent") give the committee no gradient
(only the 0.01 leak) and no disagreement, so no member learns them and the sampler never visits their kinks.
Teacher-side fix: with the prefix exact (pinned) and the frontier's pinned rows exact, the teacher's output Jacobian
w.r.t. the frontier input h, projected off the span of the pinned rows, is spanned EXACTLY by the unpinned rows
(leak slope 0.01 included) -> its numerical rank = #missing rows (a completeness certificate) and its right singular
vectors span the directions in which the missing units turn on.  Seeds: h0 = prefix(x0), target h0 + s v for random
unit v in that span, realised by attacker-side inversion through the (known) prefix.  The teacher labels the seeds like
any other query; the committee then sees the silent units active and learns / pins them through the normal pipeline.

Everything is attacker-side except the finite-difference Jacobians (2 x dim_h black-box queries per point).
"""
import math
import torch
import torch.nn.functional as F


def prefix(net, x, fr):
    """input of layer fr (flat) and all prefix pre-activations (flat, concatenated), for a nets.ConvNet."""
    h = x.view(x.shape[0], *net.input_shape) if x.dim() == 2 else x
    pre = []
    for i in range(fr):
        if i < net.n_conv:
            z = net.layers[i](h); pre.append(z.reshape(z.shape[0], -1)); h = net.act(z)
            if net.pools[i] > 0:
                h = F.avg_pool2d(h, net.pools[i])
        else:
            if h.dim() > 2:
                h = torch.flatten(h, 1)
            z = net.layers[i](h); pre.append(z); h = net.act(z)
    if h.dim() > 2:
        h = torch.flatten(h, 1)
    return h, (torch.cat(pre, 1) if pre else None)


def _prefix_jac(net, x0, fr):
    with torch.enable_grad():
        return torch.autograd.functional.jacobian(lambda x: prefix(net, x[None], fr)[0][0], x0.detach(), vectorize=True)


@torch.no_grad()
def _safe_eps(net, x0, Dx, fr, eps0=1e-6, halvings=16):
    """per h-direction step so x0 +- eps Dx e_i crosses no prefix kink (pinv columns can be ~1e3 long)."""
    _, z0 = prefix(net, x0[None], fr); z0 = z0[0] > 0
    eps = torch.full((Dx.shape[1],), eps0, device=x0.device, dtype=x0.dtype)
    for _ in range(halvings):
        E = eps[:, None] * Dx.T
        bad = ((prefix(net, x0[None] + E, fr)[1] > 0) != z0[None]).any(1) | ((prefix(net, x0[None] - E, fr)[1] > 0) != z0[None]).any(1)
        if not bad.any():
            return eps, 0
        eps = torch.where(bad, eps / 2, eps)
    return eps, int(bad.sum())


def frontier_rows(net, fr):
    W = net.layers[fr].weight.detach().double()
    return W.reshape(W.shape[0], -1)


@torch.no_grad()
def residual_subspace(bb, net64, fr, pinned, n_points=40, x_std=0.5, gen=None, rel_tol=1e-4, verbose=True):
    """bb: black box (fp64, flat input).  net64: fp64 member whose layers < fr are exact and whose frontier rows `pinned`
    are exact.  Returns (V (r, dim_h) orthonormal residual directions, singular values, #queries)."""
    dev = next(net64.parameters()).device; D = torch.double
    d_x = int(torch.tensor(net64.input_shape).prod())
    Wk = frontier_rows(net64, fr)[pinned.to(dev).bool()]
    dim_h = Wk.shape[1]
    if Wk.shape[0]:
        Q, _ = torch.linalg.qr(Wk.T); Pp = torch.eye(dim_h, device=dev, dtype=D) - Q @ Q.T
    else:
        Pp = torch.eye(dim_h, device=dev, dtype=D)
    rows = []; nq = 0; skipped = 0; tries = 0
    while len(rows) < n_points and tries < 3 * n_points:
        tries += 1
        x0 = torch.randn(d_x, device=dev, dtype=D, generator=gen) * x_std
        P = _prefix_jac(net64, x0, fr); Dx = torch.linalg.pinv(P)
        eps, nbad = _safe_eps(net64, x0, Dx, fr)
        if nbad:
            skipped += 1; continue
        E = eps[:, None] * Dx.T
        J = ((bb(x0[None] + E) - bb(x0[None] - E)) / (2 * eps[:, None])).T      # (out, dim_h)
        nq += 2 * dim_h
        rows.append(J @ Pp)
    R = torch.cat(rows); _, S, Vh = torch.linalg.svd(R, full_matrices=False)
    r = int((S > rel_tol * S[0]).sum()) if S[0] > 0 else 0
    if verbose:
        print(f"  [residual-seeds] frontier L{fr + 1}: {int(pinned.sum())} pinned rows, {len(rows)} Jacobian points "
              f"({skipped} skipped), residual rank {r} (sv {', '.join(f'{v:.1e}' for v in S[:min(len(S), r + 2)].tolist())}) | {nq:,} queries", flush=True)
    return Vh[:r], S, nq


def seeds(net64, fr, V, n, x_std=0.5, s_max=6.0, steps=200, lr=0.02, box=3.0, gen=None):
    """n inputs whose frontier input sits at h0 + s v, v random unit in span(V), s ~ U[-s_max, s_max] (attacker-side)."""
    dev = next(net64.parameters()).device; D = torch.double
    d_x = int(torch.tensor(net64.input_shape).prod())
    x0 = torch.randn(n, d_x, device=dev, dtype=D, generator=gen) * x_std
    with torch.no_grad():
        h0, _ = prefix(net64, x0, fr)
        c = torch.randn(n, V.shape[0], device=dev, dtype=D, generator=gen); v = c @ V; v = v / v.norm(dim=1, keepdim=True)
        s = (torch.rand(n, 1, device=dev, dtype=D, generator=gen) * 2 - 1) * s_max
        H = h0 + s * v
    x = x0.clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        loss = ((prefix(net64, x, fr)[0] - H) ** 2).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            x.clamp_(-box, box)
    return x.detach()


@torch.no_grad()
def reinit_unpinned_(member, fr, V_member, pinned, xs, fire_frac=0.1, gen=None, net64=None):
    """put the member's UNPINNED frontier rows in span(V) (random combination, row norm kept) and set each bias so the
    unit fires on `fire_frac` of the seed inputs xs: the member then gets full-strength gradient exactly where the
    teacher's missing units turn on (instead of the 0.01 leak), and members differ (random combinations) so they
    disagree there.  V_member: residual directions in THIS member's h gauge (rows = dim_h)."""
    W = member.layers[fr].weight; b = member.layers[fr].bias
    idx = (~pinned.to(W.device).bool()).nonzero().flatten()
    if len(idx) == 0 or V_member.shape[0] == 0:
        return 0
    h, _ = prefix(net64 if net64 is not None else member, xs.to(next(member.parameters()).dtype), fr)
    for j in idx.tolist():
        c = torch.randn(V_member.shape[0], device=V_member.device, dtype=V_member.dtype, generator=gen)
        w = c @ V_member; w = w / w.norm()
        nrm = W[j].norm().clamp_min(1e-3)
        wv = (w * nrm).to(W.dtype)
        z = h.to(wv.dtype).reshape(h.shape[0], -1) @ wv.reshape(-1)
        W[j] = wv.view_as(W[j]); b[j] = -torch.quantile(z.float(), 1 - fire_frac).to(b.dtype)
    return len(idx)
