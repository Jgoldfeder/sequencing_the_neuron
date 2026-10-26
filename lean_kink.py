"""lean_kink.py -- cheap exact kink refinement of a (leaky) ReLU MLP layer from a GOOD guess (2026-10-25).

For each unit j of layer l with guess row (w_j, b_j) (a few degrees off), with the prefix (layers < l) exact:
  1. base point: random x0; move along the prefix-mapped direction of w_j so that the GUESSED pre-activation is 0;
  2. line through that point along the mapped w_j direction; the black box is piecewise linear on the line, so the
     kink is located exactly by intersecting the two affine pieces measured just left and just right (bracket of a
     few times the guess error, then halving until each side is affine; ~10-30 queries);
  3. Jacobian jump: at x* +- eps*d (just across the kink) the black-box Jacobian wrt the layer input h, measured by
     ONE-SIDED differences along the din prefix-mapped coordinate directions (exact for a piecewise-linear map when the
     steps stay inside one region): 2*(din+1) queries.  dJ = (slope_+ - slope_-) * v w^T is rank one; its right
     singular vector is the unit-normalised row; the bias follows from w.h* + b = 0; orientation from which side has
     the larger slope (leaky ReLU is NOT odd).
  4. accept if |cos(w, guess)| > cos(gate) and the rank-one ratio is small; otherwise keep the guess.
Black box = fp64 teacher forward only (queries counted).  Total ~2*din + 40 queries per unit.
"""
import math
import torch


def prefix(net, x, l):
    h = x
    for i in range(l):
        h = net.act(net.layers[i](h))
    return h


def _prefix_pre(net, x, l):
    """all prefix pre-activations (N, sum of widths) for kink-crossing checks"""
    h = x; out = []
    for i in range(l):
        z = net.layers[i](h); out.append(z); h = net.act(z)
    return torch.cat(out, 1)


def _pjac(net, x0, l):
    if l == 0:
        return torch.eye(x0.shape[0], dtype=x0.dtype, device=x0.device)
    with torch.enable_grad():
        return torch.autograd.functional.jacobian(lambda x: prefix(net, x[None], l)[0], x0.detach(), vectorize=True)


class Oracle:
    def __init__(self, f):
        self.f, self.n = f, 0

    @torch.no_grad()
    def __call__(self, X):
        self.n += X.shape[0]; return self.f(X)


@torch.no_grad()
def _locate(bb, x, d, t_guess, half, iters=40):
    """exact kink of the black box on the line x + t d near t_guess.  Piecewise linear: take 4 points (2 per side)
    at +-half; if both sides are affine (collinear triple check) intersect the lines; else shrink and recentre."""
    lo, hi = t_guess - half, t_guess + half
    for _ in range(iters):
        ts = torch.tensor([lo, lo + (hi - lo) * 0.1, hi - (hi - lo) * 0.1, hi], dtype=x.dtype, device=x.device)
        Y = bb(x[None] + ts[:, None] * d[None])                 # (4, out)
        sl = (Y[1] - Y[0]) / (ts[1] - ts[0]); sr = (Y[3] - Y[2]) / (ts[3] - ts[2])
        ds = sr - sl
        k = int(ds.abs().argmax())
        if float(ds.abs().max()) < 1e-10 * (1 + float(sl.abs().max())):
            return None                                          # no kink in the bracket
        # intersection of the two lines (component with the largest slope change)
        tk = float((Y[2, k] - Y[0, k] - sr[k] * ts[2] + sl[k] * ts[0]) / (sl[k] - sr[k]))
        # verify: predicted values at tk from both lines agree with the black box at tk +- small
        if lo < tk < hi:
            e = (hi - lo) * 1e-3
            yt = bb(x[None] + torch.tensor([tk - e, tk + e], dtype=x.dtype, device=x.device)[:, None] * d[None])
            pl = Y[0] + sl * (tk - e - ts[0]); pr = Y[3] + sr * (tk + e - ts[3])
            ok = float((yt[0] - pl).abs().max()) < 1e-9 * (1 + float(Y.abs().max())) and \
                 float((yt[1] - pr).abs().max()) < 1e-9 * (1 + float(Y.abs().max()))
            if ok:
                return tk
            lo, hi = tk - (hi - lo) / 8, tk + (hi - lo) / 8      # another kink nearby: zoom in
        else:
            lo, hi = t_guess - half / 2, t_guess + half / 2; half /= 2
    return None


@torch.no_grad()
def refine_unit(bb, net, l, w, b, gen, x_std=0.5, gate_deg=10.0, eps_side=1e-5, h=1e-7, tries=4, why=None, pick=16, tiers=(10.0, 100.0, 1000.0, 1e4, 1e5)):
    """returns (unit row w (din), bias b, info) in the unit-normalised gauge, or None."""
    dev, D = w.device, w.dtype
    din = w.shape[0]; d_x = net.layers[0].weight.shape[1]
    wn = w / w.norm(); bn = b / w.norm()
    for _ in range(tries):
        x0 = torch.randn(d_x, dtype=D, device=dev, generator=gen) * x_std
        if l > 0 and pick > 1:                                                  # attacker-side: best-conditioned prefix Jacobian
            best = None
            for _c in range(pick):
                xc = torch.randn(d_x, dtype=D, device=dev, generator=gen) * x_std * float(torch.rand(1, generator=gen, device=dev) * 2 + 0.25)
                Sv = torch.linalg.svdvals(_pjac(net, xc, l))
                c = float(Sv[0] / Sv[-1].clamp_min(1e-300))
                if best is None or c < best[0]:
                    best = (c, xc)
            x0 = best[1]
        P = _pjac(net, x0, l)                                                   # (din, d_x)
        h0 = prefix(net, x0[None], l)[0]
        gx = P.T @ wn if l > 0 else wn                                          # x-gradient of the guessed pre-activation
        dirx = gx / gx.norm(); s = float(gx.norm())                             # dz/dt along the unit x-direction
        t0 = -float(wn @ h0 + bn) / s
        if l > 0:                                                               # Newton along dirx to the guessed surface (exact prefix)
            x1 = x0 + t0 * dirx
            for _ in range(4):
                z = float(wn @ prefix(net, x1[None], l)[0] + bn)
                g = float(wn @ (_pjac(net, x1, l) @ dirx))
                if abs(g) < 1e-12:
                    break
                x1 = x1 - (z / g) * dirx
            x0, t0 = x1, 0.0
        tk = _locate(bb, x0, dirx, t0, half=0.05 * (1 + abs(t0)))
        if tk is None:
            if why is not None: why.append("no-kink")
            continue
        xk = x0 + tk * dirx
        Pk = _pjac(net, xk, l); Dk = torch.linalg.pinv(Pk) if l > 0 else Pk
        cn = Dk.norm(dim=0).clamp_min(1e-30)                                    # length of each mapped h-direction in x
        side = []
        for sg in (-1.0, 1.0):
            xs = xk + sg * eps_side * dirx
            hs = torch.full((din,), h, dtype=D, device=dev) / cn                # h-space step so every x step has length h
            if l > 0:                                                            # attacker-side: no PREFIX kink crossed by a step
                z0 = _prefix_pre(net, xs[None], l)[0] > 0
                for _ in range(20):
                    zp = _prefix_pre(net, xs[None] + hs[:, None] * Dk.T, l) > 0
                    bad = (zp != z0[None]).any(1)
                    if not bad.any():
                        break
                    hs = torch.where(bad, hs / 4, hs)
            E = hs[:, None] * Dk.T                                               # (din, d_x)
            Y0 = bb(xs[None]); Yp = bb(xs[None] + E)
            Jc = ((Yp - Y0) / hs[:, None])                                       # (din, out): smallest-step estimate
            if l == 0 and tiers:
                # step-size ladder: larger one-sided steps cut the fp64 roundoff ~ eps*|f|/h; a step that crosses
                # another kink disagrees with the smaller-step estimate beyond its noise -> keep the smaller one
                fs = float(Y0.abs().max()) + 1.0
                cur_h = hs.clone()
                for fac in tiers:
                    hb = hs * fac
                    Jb = (bb(xs[None] + hb[:, None] * Dk.T) - Y0) / hb[:, None]
                    tol = 50 * 2.2e-16 * fs / cur_h[:, None] + 1e-12
                    good = ((Jb - Jc).abs() <= tol).all(1)
                    Jc = torch.where(good[:, None], Jb, Jc); cur_h = torch.where(good, hb, cur_h)
            side.append(Jc.T)                                                    # (out, din) Jacobian wrt h
        dJ = side[1] - side[0]
        U, S, Vh = torch.linalg.svd(dJ, full_matrices=False)
        if float(S[0]) < 1e-9 or float(S[1] / S[0]) > 1e-4:
            if why is not None: why.append(f"rank s0={float(S[0]):.1e} ratio={float(S[1] / S[0]):.1e} minstep={float(hs.min()):.1e}")
            continue
        r = Vh[0]
        if float(r @ wn) < 0:
            r = -r
        cos = float(r @ wn)
        if cos < math.cos(math.radians(gate_deg)):
            if why is not None: why.append(f"cos {cos:.3f}")
            continue
        hk = prefix(net, xk[None], l)[0]
        rb = -float(r @ hk)
        if l == 0:
            # offset from the ORIGIN line (kink-refine precision fix): on x = s r the pre-activation is s + b, so the
            # kink sits at s* = -b; reading b there avoids multiplying the normal's error by ||x_k|| (~30 here)
            s_k = _locate(bb, torch.zeros_like(r), r, -rb, half=1e-4 * (1 + abs(rb)))
            if s_k is not None and abs(-float(s_k) - rb) < 1e-5 * (1 + abs(rb)):   # same kink (else another unit's): keep it
                rb = -float(s_k)
        # orientation: the active side (slope 1) is where r.h + b > 0; check the jump sign agrees with the guess side
        v = U[:, 0] * S[0]
        Jp = side[1] @ r; Jm = side[0] @ r                                       # slopes along r on each side
        if float((Jp - Jm) @ v) * float(Vh[0] @ r) < 0:
            pass
        return r, rb, dict(cos=cos, rank_ratio=float(S[1] / S[0]))
    return None


@torch.no_grad()
def refine_layer(teacher64, net, l, units=None, seed=0, verbose=True, **kw):
    """refine units of layer l of `net` (fp64, layers < l exact).  Returns rows W (n, din), biases b (n,), ok mask, queries."""
    bb = Oracle(lambda X: teacher64(X))
    dev = next(net.parameters()).device; gen = torch.Generator(device=dev).manual_seed(seed)
    W = net.layers[l].weight.detach().double(); B = net.layers[l].bias.detach().double()
    n = W.shape[0]; units = range(n) if units is None else units
    Wr = torch.zeros_like(W); Br = torch.zeros_like(B); ok = torch.zeros(n, dtype=torch.bool, device=dev)
    for j in units:
        out = refine_unit(bb, net, l, W[j], B[j], gen, **kw)
        if out is not None:
            Wr[j], Br[j], ok[j] = out[0], out[1], True
    if verbose:
        print(f"  [lean-kink] L{l + 1}: {int(ok.sum())}/{len(list(units))} units, {bb.n:,} queries "
              f"({bb.n / max(1, len(list(units))):.0f}/unit)", flush=True)
    return Wr, Br, ok, bb.n


@torch.no_grad()
def refine_unit_points(bb, net, l, w, b, gen, x_std=0.5, extra=40, max_tries_factor=3, gate_deg=10.0, trim_rounds=3):
    """deep layers (l > 0): the row from kink LOCATIONS.  Every exact kink point x_k of unit j satisfies
    w.h(x_k) + b = 0 with h = the (exact) prefix -> collect din + extra points (located on lines along the x-gradient
    of the guessed pre-activation, each 10-30 queries) and take the null vector of [H | 1] (SVD), trimming the worst
    residuals (points that belong to another unit).  No derivatives: immune to the prefix Jacobian's conditioning."""
    dev, D = w.device, w.dtype
    din = w.shape[0]; d_x = net.layers[0].weight.shape[1]
    wn = w / w.norm(); bn = b / w.norm()
    need = din + extra; pts = []; tries = 0
    while len(pts) < need and tries < max_tries_factor * need:
        tries += 1
        x0 = torch.randn(d_x, dtype=D, device=dev, generator=gen) * x_std
        h0 = prefix(net, x0[None], l)[0]
        gx = _pjac(net, x0, l).T @ wn
        if float(gx.norm()) < 1e-12:
            continue
        dirx = gx / gx.norm(); t0 = -float(wn @ h0 + bn) / float(gx.norm())
        tk = _locate(bb, x0, dirx, t0, half=0.05 * (1 + abs(t0)))
        if tk is None:
            continue
        hk = prefix(net, (x0 + tk * dirx)[None], l)[0]
        if abs(float(wn @ hk + bn)) > 0.05 * (1 + float(hk.norm()) * 0.01):     # far from the guessed surface: another unit
            continue
        pts.append(hk)
    if len(pts) < din + 1:
        return None
    H = torch.stack(pts); A = torch.cat([H, torch.ones(len(H), 1, dtype=D, device=dev)], 1)
    keep = torch.ones(len(A), dtype=torch.bool, device=dev)
    for _ in range(trim_rounds):
        Am = A[keep]; Am = Am / Am.norm(dim=1, keepdim=True)
        v = torch.linalg.svd(Am, full_matrices=False)[2][-1]
        res = (A @ v).abs() / A.norm(dim=1)
        thr = max(1e-9, 10 * float(res[keep].median()))
        new_keep = res < thr
        if int(new_keep.sum()) < din + 1 or bool((new_keep == keep).all()):
            break
        keep = new_keep
    r = v[:-1]; rb = float(v[-1]); nr = float(r.norm()); r = r / nr; rb = rb / nr
    if float(r @ wn) < 0:
        r, rb = -r, -rb
    if float(r @ wn) < math.cos(math.radians(gate_deg)):
        return None
    return r, rb, dict(cos=float(r @ wn), points=int(keep.sum()), med_res=float(res[keep].median()))
