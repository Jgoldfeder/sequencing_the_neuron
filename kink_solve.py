"""Deep-layer refinement for the MLP peel (--loc-refine): TWO STAGES.

Setting: layers < l of `guess` are (near-)exact and layer l's rows are within
~1e-2. Goal: layer l to the fp64 floor, in seconds, black-box.

  stage 1  the existing seal refiner (method._mlp_refine_layer -> verify_layer1
           .refine_neuron): ONE kink per neuron + rank-1 gradient-jump normal,
           ~1.5k queries/neuron, lands at ~1e-8 (inverts the prefix, so the
           prefix error is amplified ~100x). It is a chain of ~500 tiny
           sequential oracle calls per neuron: latency-bound on a GPU, so it
           runs on a CPU fork pool (200 neurons in ~2.5s on 40 cores).
  stage 2  kink POLISH (polish_layer): with a ~1e-8 row the bracket around the
           guessed kink is ~1e-6 wide and free of foreign bends, so each kink
           point costs 8 queries (3+3 collinear side points + 2 verification
           points; no scan, no fingerprint). Din+40 points per neuron, all
           neurons in one batched oracle pass (~2.5k queries/neuron), then the
           null vector of [h 1] by leverage-aware trimming (solve_from_points).
           Prefix error enters x1 (forward eval only): exact prefix -> ~1e-14,
           1e-8 prefix -> ~3e-8. A polish inconsistent with its stage-1 row
           means stage 1 locked a wrong kink -> the channel falls back to:
  fallback solve_neuron: Carlini-style from the ~1e-2 guess. Aim at the guessed
           kink along its NORMAL (bracket ~sqrt(d) shorter than a random
           direction), scan the bracket for all bends, fingerprint each bend's
           normal against the guess with probes orthogonalized in h-space,
           ~25k queries/neuron.

Measured (7-layer 200-wide teacher, layer 1, 200 neurons): 200/200 in ~10s,
~4k queries/neuron, max err 3.8e-14 (exact prefix) / 7.6e-8 (1e-8 prefix).
"""
import torch
import torch.nn.functional as F


def _phi(net, X, l):
    a = X
    for i in range(l):
        a = net.act(net.layers[i](a))
    return a


@torch.no_grad()
def _preacts(net, X, l):
    out = []
    a = X
    for i in range(l + 1):
        z = net.layers[i](a)
        out.append(z)
        a = net.act(z)
    return out


class MLPUnits:
    """Unit view of hidden layer l of an MLP: unit = neuron (index j == channel),
    feat(x) = the layer's input (prefix output), row j = (W[j], b[j])."""
    def __init__(self, net, l):
        self.net, self.l = net, l
        L = net.layers[l]
        self.W, self.B = L.weight, L.bias
        self.n_channels = self.W.shape[0]; self.n_pos = 1; self.n_units = self.n_channels
        self.din = self.W.shape[1]; self.d = net.layers[0].weight.shape[1]
        self.device, self.dtype, self.act = self.W.device, self.W.dtype, net.act
        self.has_preacts = True
    def feat(self, X, jidx=None):
        return _phi(self.net, X, self.l)
    def rows(self, j):
        return self.W[j], self.B[j]
    def channel(self, j):
        return j
    def units_of(self, c, n, gen):
        return torch.full((n,), int(c), device=self.device, dtype=torch.long)
    def preacts(self, X):
        return _preacts(self.net, X, self.l)
    def weight_shape(self):
        return tuple(self.net.layers[self.l].weight.shape)


class ConvNetUnits:
    """Unit view of layer `frontier` of a nets.ConvNet.
    Conv frontier: unit = (channel c, output position p) encoded j = c*n_pos + p;
      feat(x, j) = the receptive-field patch (C_in*k*k) of the prefix image at p;
      row = flattened filter c. A kink of unit (c,p) gives w_c . patch + b_c = 0:
      one linear constraint on the shared filter, from ANY position.
    FC frontier: unit = neuron, feat = flattened prefix output."""
    def __init__(self, net, frontier):
        self.net, self.l = net, frontier
        L = net.layers[frontier]
        self.is_conv = frontier < net.n_conv
        self.device, self.dtype, self.act = L.weight.device, L.weight.dtype, net.act
        self.d = int(torch.tensor(net.input_shape).prod())
        self.has_preacts = False
        if self.is_conv:
            self.k = L.kernel_size[0]; self.s = L.stride[0]; self.pad = L.padding[0]
            with torch.no_grad():
                x = torch.zeros(1, self.d, device=self.device, dtype=self.dtype)
                self.hout = tuple(L(self._prefix(x)).shape[2:])
            self.n_pos = self.hout[0] * self.hout[1]
            self.n_channels = L.weight.shape[0]; self.din = L.weight[0].numel()
            self.W = L.weight.reshape(self.n_channels, -1); self.B = L.bias
        else:
            self.n_pos = 1; self.n_channels = L.weight.shape[0]; self.din = L.weight.shape[1]
            self.W, self.B = L.weight, L.bias
        self.n_units = self.n_channels * self.n_pos
    def _prefix(self, x):
        net = self.net
        x = x.view(x.shape[0], *net.input_shape)
        for i in range(net.n_conv):
            if i == self.l:
                return x
            x = net.act(net.layers[i](x))
            if net.pools[i] > 0:
                x = F.avg_pool2d(x, net.pools[i])
        x = torch.flatten(x, 1)
        for j in range(net.n_conv, len(net.layers) - 1):
            if j == self.l:
                return x
            x = net.act(net.layers[j](x))
        return x
    def feat(self, X, jidx=None):
        h = self._prefix(X)
        if not self.is_conv:
            return h
        cols = F.unfold(h, self.k, padding=self.pad, stride=self.s)         # (n, Din, L)
        p = (jidx % self.n_pos) if jidx is not None else torch.zeros(len(X), dtype=torch.long, device=X.device)
        return cols[torch.arange(len(X), device=X.device), :, p]           # (n, Din)
    def rows(self, j):
        c = j // self.n_pos if torch.is_tensor(j) else int(j) // self.n_pos
        return self.W[c], self.B[c]
    def channel(self, j):
        return j // self.n_pos
    def units_of(self, c, n, gen):
        p = torch.randint(self.n_pos, (n,), device=self.device, generator=gen)
        return int(c) * self.n_pos + p
    def weight_shape(self):
        return tuple(self.net.layers[self.l].weight.shape)


def _as_model(guess, l):
    if isinstance(guess, (MLPUnits, ConvNetUnits)):
        return guess
    if hasattr(guess, "n_conv"):
        return ConvNetUnits(guess, l)
    return MLPUnits(guess, l)


def _gval(guess, X, l, wg, bg, jidx=None):
    """wg.feat(X)+bg; wg may be one row (Din,) or one row per X row (n, Din)."""
    M = _as_model(guess, l)
    P = M.feat(X, jidx)
    return (P @ wg + bg) if wg.dim() == 1 else ((P * wg).sum(1) + bg)


def _g_and_normal(guess, X, l, wg, bg, jidx=None):
    M = _as_model(guess, l)
    with torch.enable_grad():
        Xg = X.detach().requires_grad_(True)
        g = _gval(M, Xg, l, wg, bg, jidx)
        n = torch.autograd.grad(g.sum(), Xg)[0]
    return g.detach(), n


@torch.no_grad()
def seed_brackets(guess, l, j, n_seed, gen, eps=0.02, margin=3.0, x_scale=1.0, isolate_prefix=True):
    """(X0, U, R[, jidx]): points on unit j's guessed kink, unit normal there,
    and half-width R of a bracket along U. j may be an int or a LongTensor of
    per-seed target units (then n_seed = len(j), eps may be per-seed, and the
    kept jidx is returned too). Works for MLP layers and ConvNet layers
    (units = (channel, position)) through the unit model."""
    M = _as_model(guess, l)
    dev, d = M.device, M.d
    multi = torch.is_tensor(j)
    jidx = j if multi else None
    if multi:
        n_seed = len(j)
    wg, bg = M.rows(j)
    X = x_scale * torch.randn(n_seed, d, device=dev, dtype=torch.float64, generator=gen)
    for _ in range(8):                                     # Newton on piecewise-linear g
        g, n = _g_and_normal(M, X, l, wg, bg, jidx)
        nn2 = (n * n).sum(1).clamp_min(1e-30)
        X = X - (g / nn2)[:, None] * n
    g, n = _g_and_normal(M, X, l, wg, bg, jidx)
    nrm = n.norm(dim=1)
    U = n / nrm.clamp_min(1e-30)[:, None]
    # finish with a model-only bisection along U so g(X0) = 0 to ~1e-15
    lo = torch.full_like(g, -1e-2); hi = torch.full_like(g, 1e-2)
    glo = _gval(M, X + lo[:, None] * U, l, wg, bg, jidx)
    ghi = _gval(M, X + hi[:, None] * U, l, wg, bg, jidx)
    ok = (glo * ghi < 0) & (nrm > 1e-8)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        gm = _gval(M, X + mid[:, None] * U, l, wg, bg, jidx)
        left = gm * glo > 0
        lo = torch.where(left, mid, lo); glo = torch.where(left, gm, glo)
        hi = torch.where(left, hi, mid)
    X0 = X + (0.5 * (lo + hi))[:, None] * U
    H = M.feat(X0, jidx)
    # |dg| = |dw.h + db| ~ eps*(|h|+1)/sqrt(Din) for a random-direction row
    # error of relative size eps; R = margin x that, divided by the slope |n|.
    Din = H.shape[1]
    R = margin * eps * (H.norm(dim=1) + 1.0) / (Din ** 0.5) / nrm.clamp_min(1e-30)
    # optional model-only isolation (MLP only): no other kink of layers <= l
    # inside the bracket. The scan + fingerprint handle every other bend, so
    # this is off in the direct/fallback paths.
    if isolate_prefix and M.has_preacts:
        for i in range(l + 1):
            m = R if i < l else 2 * R
            zm = M.preacts(X0 - m[:, None] * U)[i]
            zp = M.preacts(X0 + m[:, None] * U)[i]
            sflip = torch.sign(zm) * torch.sign(zp) < 0
            if i == l:
                if multi:
                    sflip[torch.arange(len(j), device=dev), j] = False
                else:
                    sflip[:, j] = False
            ok &= ~sflip.any(1)
    if multi:
        return X0[ok], U[ok], R[ok], j[ok]
    return X0[ok], U[ok], R[ok]


def _linefit(T, Y):
    """exact line fit per row: T (n,k), Y (n,k,o) -> a (n,o), s (n,o), resid (n,)"""
    tm = T.mean(1, keepdim=True); ym = Y.mean(1, keepdim=True)
    s = ((T - tm)[:, :, None] * (Y - ym)).sum(1) / ((T - tm) ** 2).sum(1)[:, None]
    a = ym[:, 0] - s * tm
    res = (Y - a[:, None, :] - s[:, None, :] * T[:, :, None]).abs().amax(dim=(1, 2))
    return a, s, res


def _probe_dirs(guess, l, X, U, wg, gen, n_probe=3, fd=1e-6, jidx=None):
    """n_probe unit x-directions v per row whose induced h-displacement
    dh_v = J_phi v is orthogonal to wg (v <- v - beta U with beta chosen so
    wg.dh_v = 0). Returns (V (n,p,d), |dh_v| (n,p), |dh_U| (n,)). Model only."""
    n, d = X.shape
    M = _as_model(guess, l)
    H0 = M.feat(X, jidx)
    dhU = (M.feat(X + fd * U, jidx) - H0) / fd
    V = torch.randn(n, n_probe, d, device=X.device, dtype=X.dtype, generator=gen)
    V = V / V.norm(dim=2, keepdim=True)
    jrep = jidx.repeat_interleave(n_probe) if jidx is not None else None
    dhV = (M.feat((X[:, None, :] + fd * V).reshape(-1, d), jrep).reshape(n, n_probe, -1) - H0[:, None, :]) / fd
    if wg.dim() == 2:                                       # one row per seed
        beta = (dhV * wg[:, None, :]).sum(2) / (dhU * wg).sum(1)[:, None]
    else:
        beta = (dhV @ wg) / (dhU @ wg)[:, None]
    V = V - beta[:, :, None] * U[:, None, :]
    dhV = dhV - beta[:, :, None] * dhU[:, None, :]
    nv = V.norm(dim=2, keepdim=True)
    return V / nv, dhV.norm(dim=2) / nv[:, :, 0], dhU.norm(dim=1)


@torch.no_grad()
def locate(oracle, X0, U, R, gen, guess, l, wg, K=15, span=0.6, tol=1e-9, fp_tol=0.03, n_probe=3, max_cand=8, debug=None, jidx=None):
    """Find neuron j's kink on the segment X0 + t U, |t| <= R, among the other
    (deeper-layer) bends on it.
      scan: K points; for every interval (i, i+1) fit the line through points
            (i-1, i) and through (i+1, i+2); if they differ, intersect -> t*.
      verify: query x* -+ e U (e = 1e-3 spacing). Both must lie EXACTLY on their
            side line: any second bend in [t_{i-1}, t*-e] or [t*+e, t_{i+2}]
            breaks that (the oracle is exact), so the interval holds one bend.
      fingerprint: in h-space (exact prefix) every kink of layers >= l is a
            hyperplane a.dh = 0; ours has a = w_true = w_g + O(eps). Probe the
            gradient jump along x-directions v whose dh_v is orthogonal to w_g:
            |jump_v|/|dh_v| over |jump_U|/|dh_U| is O(eps) for ours, O(1) for a
            foreign bend (the probes are orthogonalized in h-space, not x-space,
            because the prefix Jacobian is anisotropic and would otherwise
            correlate all normals). Candidates are tried nearest-to-0 first.
    Returns (X*, ok)."""
    n, d = X0.shape
    dev, dt = X0.device, X0.dtype
    if n == 0:
        return X0, torch.zeros(0, dtype=torch.bool, device=dev)
    grid = span * torch.linspace(-1.0, 1.0, K, device=dev, dtype=dt)
    T = R[:, None] * grid[None, :]                                   # (n,K)
    P = X0[:, None, :] + T[:, :, None] * U[:, None, :]
    Y = oracle(P.reshape(-1, d)).reshape(n, K, -1)                  # (n,K,o)
    scale = Y.abs().amax(dim=(1, 2)).clamp_min(1e-300)
    sp = T[:, 1] - T[:, 0]                                            # spacing
    cand = []
    for i in range(1, K - 2):                                        # interval (i, i+1)
        sL = (Y[:, i] - Y[:, i - 1]) / sp[:, None]; aL = Y[:, i] - sL * T[:, i, None]
        sR = (Y[:, i + 2] - Y[:, i + 1]) / sp[:, None]; aR = Y[:, i + 1] - sR * T[:, i + 1, None]
        ds = sR - sL
        dsn = ds.norm(dim=1)
        tstar = ((aL - aR) * ds).sum(1) / (dsn ** 2).clamp_min(1e-300)
        ok = (dsn * sp > 1e3 * tol * scale) & (tstar > T[:, i]) & (tstar < T[:, i + 1])
        cand.append((tstar, ok, aL, sL, aR, sR, ds))
    tolv_all = 1e-9 * torch.stack([c[6].norm(dim=1) for c in cand]) * sp[None, :]   # (I, n): per-bend tolerance
    cand_t = torch.stack([c[0] for c in cand], 1); cand_ok = torch.stack([c[1] for c in cand], 1)
    # (I, n, o) stacks so a candidate's line data is one advanced-indexing gather
    c_aL = torch.stack([c[2] for c in cand]); c_sL = torch.stack([c[3] for c in cand])
    c_aR = torch.stack([c[4] for c in cand]); c_sR = torch.stack([c[5] for c in cand])
    c_ds = torch.stack([c[6] for c in cand])
    key = torch.where(cand_ok, cand_t.abs(), torch.full_like(cand_t, float("inf")))
    order = key.argsort(1)
    Xstar = X0.clone(); found = torch.zeros(n, dtype=torch.bool, device=dev)
    ar = torch.arange(n, device=dev)
    for c in range(min(max_cand, order.shape[1])):
        idx = order[:, c]
        act = (~found) & torch.isfinite(key.gather(1, idx[:, None])[:, 0])
        if not act.any():
            break
        sub = act.nonzero()[:, 0]
        ks = idx[sub]
        ts = cand_t[sub, ks]
        aL, sL, aR, sR, ds = c_aL[ks, sub], c_sL[ks, sub], c_aR[ks, sub], c_sR[ks, sub], c_ds[ks, sub]
        Us, Xs0, sps = U[sub], X0[sub], sp[sub]
        xs = Xs0 + ts[:, None] * Us
        # probe directions at the CANDIDATE point (the prefix Jacobian differs
        # across any earlier-layer bend between X0 and x*)
        Vs, dhV_s, dhU_s = _probe_dirs(guess, l, xs, Us, wg[sub] if wg.dim() == 2 else wg, gen, n_probe,
                                       jidx=jidx[sub] if jidx is not None else None)
        e = 1e-3 * sps; dl = 5e-5 * sps        # dl << e: the perpendicular step must not re-cross the kink
        xm = xs - e[:, None] * Us; xp = xs + e[:, None] * Us
        # probe set: x-, x+, then each shifted by dl*U (reference jump) and dl*v_p
        Vall = torch.cat([Us[:, None, :], Vs], 1)                    # (m,1+p,d)
        Pf = torch.cat([xm[:, None], xp[:, None],
                        xm[:, None, :] + dl[:, None, None] * Vall, xp[:, None, :] + dl[:, None, None] * Vall], 1)
        Yf = oracle(Pf.reshape(-1, d)).reshape(len(sub), 2 + 2 * (1 + n_probe), -1)
        # single-bend verification: probes on their side lines
        resL = (Yf[:, 0] - (aL + sL * (ts - e)[:, None])).abs().amax(1)
        resR = (Yf[:, 1] - (aR + sR * (ts + e)[:, None])).abs().amax(1)
        tolv = torch.minimum(tol * scale[sub], tolv_all[ks, sub])
        single = (resL < tolv) & (resR < tolv)
        # gradient jumps across the probe pair along U and the p perpendicular dirs
        J = (Yf[:, 3 + n_probe:] - Yf[:, 1, None] - Yf[:, 2:3 + n_probe] + Yf[:, 0, None]) / dl[:, None, None]  # (m,1+p,o)
        JU, Jv = J[:, 0], J[:, 1:]
        # positive evidence: the probe pair straddles the bend the scan found
        straddle = (JU - ds).norm(dim=1) < 1e-3 * ds.norm(dim=1)
        # fingerprint: h-normalized perpendicular jump ratios
        r = (Jv.norm(dim=2) / dhV_s) / (JU.norm(dim=1) / dhU_s)[:, None]
        good = single & straddle & (r < fp_tol).all(1)
        if debug is not None:
            debug.append(dict(sub=sub, xs=xs, single=single, straddle=straddle, r=r, ds=ds, good=good))
        found[sub[good]] = True
        Xstar[sub[good]] = xs[good]
    return Xstar, found


@torch.no_grad()
def solve_from_points(H, w_guess, gen=None, rounds=8, drop=0.02):
    """(w, b) with w.h + b = 0 on the inlier points. Inliers are exact
    (residual ~ prefix error); a contaminated point is off by >~1e-7. A plain
    least-squares fit spreads an outlier's error over all rows -- and a HIGH-
    LEVERAGE outlier is absorbed almost entirely, so its raw residual can even
    be small. So rank points by their LEAVE-ONE-OUT residual r_i/(1-h_ii)
    (h_ii = hat-matrix leverage, from the QR of the regression form with the
    guess's largest coordinate pinned to 1): fit, drop the worst `drop`
    fraction by LOO residual, refit, for a fixed number of rounds (the
    median only collapses once the LAST outlier is out, and a few QRs are
    free); the lowest-median fit wins. Final row = null
    vector (SVD) of the kept points. RANSAC is NOT usable: a (Din+1)-subset
    of Din+40 rows is almost never outlier-free."""
    Din = H.shape[1]
    dev0 = H.device
    H = H.cpu(); w_guess = w_guess.cpu()                 # small dense SVD/QR: CPU is ~20x faster than cuSOLVER
    A = torch.cat([H, torch.ones(len(H), 1, device=H.device, dtype=H.dtype)], 1)
    n = len(A)
    k = int(w_guess.abs().argmax())
    cols = [m for m in range(Din + 1) if m != k]
    Ak, Ar = A[:, k], A[:, cols]
    keep = torch.ones(n, dtype=torch.bool, device=H.device)
    best = None                                        # (median, keep)
    for _ in range(rounds):
        Q, Rm = torch.linalg.qr(Ar[keep])
        coef = torch.linalg.solve_triangular(Rm, -(Q.T @ Ak[keep])[:, None], upper=True)[:, 0]
        res = (Ar @ coef + Ak).abs()
        lev = torch.zeros(n, device=H.device, dtype=H.dtype)
        lev[keep] = (Q * Q).sum(1)
        loo = res / (1.0 - lev).clamp_min(1e-3)
        med = loo[keep].median().item()
        if best is None or med < best[0]:              # lowest median wins
            best = (med, keep.clone())
        kd = max(1, int(drop * int(keep.sum())))
        if int(keep.sum()) - kd < Din + 4:
            break
        worst = torch.where(keep, loo, torch.full_like(loo, -1.0)).topk(kd).indices
        keep[worst] = False
    med, keep = best
    _, S, Vh = torch.linalg.svd(A[keep], full_matrices=False)
    v = Vh[-1]
    w, b = v[:Din], v[Din]
    nrm = w.norm(); w = w / nrm; b = b / nrm
    if torch.dot(w, w_guess) < 0:
        w, b = -w, -b
    return w.to(dev0), b.to(dev0), int(keep.sum()), (S[-1] / S[-2]).item()


@torch.no_grad()
def solve_neuron(oracle, guess, l, j, gen, need=None, eps=0.02, verbose=False, return_points=False):
    Din = guess.layers[l].weight.shape[1]
    need = need or Din + 16
    q0 = oracle.n
    Hs = []; Ts = []; got = 0; n_seed = 0; tried = 0; hit = 0
    while got < need:
        if n_seed > 400 * need:                        # give up: caller rejects
            if got < Din + 8:
                raise RuntimeError(f"neuron {j}: only {got} kinks from {n_seed} seeds")
            break
        X0, U, R = seed_brackets(guess, l, j, 8 * need, gen, eps=eps, isolate_prefix=False)
        n_seed += 8 * need
        if len(X0) == 0:
            tried += 1                                  # empty batch: bracket too wide
            if tried >= 5:
                eps *= 0.5; tried = 0
            continue
        X0, U, R = X0[:need - got], U[:need - got], R[:need - got]
        Xs, ok = locate(oracle, X0, U, R, gen, guess, l, guess.layers[l].weight[j],
                        K=25 + 12 * l)                     # more bends per bracket at depth
        Hs.append(_phi(guess, Xs[ok], l)); got += int(ok.sum())
        Ts.append((((Xs - X0) * U).sum(1) / R)[ok].abs())      # kink offset / R
        tried += len(ok); hit += int(ok.sum())
        if verbose > 1:
            print(f"    j={j} batch: {len(X0)} brackets, {int(ok.sum())}/{len(ok)} located, eps={eps:.3g}", flush=True)
        # adapt the bracket to the OBSERVED kink offsets: ours is Gaussian around
        # the guessed kink; if the 95th pct sits at the scan edge we are clipping
        # (widen), if it is deep inside we waste bends (narrow). And widen when
        # (almost) nothing is found at all.
        t_all = torch.cat(Ts)
        if len(t_all) >= 30:
            q95 = t_all.quantile(0.95).item()
            if q95 > 0.45:
                eps *= 1.5; Ts = []
            elif q95 < 0.12:
                eps *= 0.6; Ts = []
        elif tried >= 100 and hit < 0.05 * tried:
            eps *= 2; tried = 0; hit = 0
    H = torch.cat(Hs)
    w, b, kept, gap = solve_from_points(H, guess.layers[l].weight[j], gen=gen)
    if verbose:
        print(f"  neuron {j:4d}: {len(H)} kinks from {n_seed} seeds, "
              f"{oracle.n - q0} q, kept {kept}, s_min/s_next {gap:.1e}", flush=True)
    if return_points:
        return w, b, H
    return w, b


@torch.no_grad()
def solve_layer(oracle, guess, l, seed=0, neurons=None, eps=0.02, verbose=True):
    dev = guess.layers[0].weight.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    W = guess.layers[l].weight.clone(); B = guess.layers[l].bias.clone()
    neurons = range(W.shape[0]) if neurons is None else neurons
    for j in neurons:
        w, b = solve_neuron(oracle, guess, l, j, gen, eps=eps, verbose=verbose)
        W[j] = w; B[j] = b
    return W, B


@torch.no_grad()
def locate_light(oracle, X0, U, R, tol=1e-12):
    """Kink location for a bracket that is (almost surely) free of foreign
    bends: 3 points per side (t = -R, -3R/4, -R/2 and R/2, 3R/4, R); each side
    must be EXACTLY collinear (else a bend sits in that side, e.g. our kink
    when the bracket is too narrow -- a chord would then give a wrong
    intersection); intersect the two lines; then verify with 2 points at
    t* -+ R/8, which must lie exactly on their side lines (a foreign bend inside
    the central gap would shift the intersection and break this). 8 queries.
    Returns (X*, ok)."""
    n, d = X0.shape
    grid = torch.tensor([-1.0, -0.75, -0.5, 0.5, 0.75, 1.0], device=X0.device, dtype=X0.dtype)
    T = R[:, None] * grid[None, :]
    P = X0[:, None, :] + T[:, :, None] * U[:, None, :]
    Y = oracle(P.reshape(-1, d)).reshape(n, 6, -1)
    aL, sL, rL = _linefit(T[:, :3], Y[:, :3])
    aR, sR, rR = _linefit(T[:, 3:], Y[:, 3:])
    ds = sR - sL; dsn = ds.norm(dim=1)
    scale = Y.abs().amax(dim=(1, 2)).clamp_min(1e-300)
    tstar = ((aL - aR) * ds).sum(1) / (dsn ** 2).clamp_min(1e-300)
    # collinearity tolerance RELATIVE TO OUR BEND: an undetected foreign bend of
    # slope change ds' biases t* by ~ds'*R/dsn, so allow only ds'*R < 1e-9*dsn*R.
    # (1e-12*|Y| let the thousands of weak conv-unit bends through -> uniform
    # ~1e-9 point errors.) The bend-strength gate dsn*R > 1e-6*|Y| keeps this
    # tolerance above fp64 noise (~1e-16*|Y|).
    tolv = torch.minimum(tol * scale, 1e-9 * dsn * R)
    ok = ((dsn * R > 1e-6 * scale) & (rL < tolv) & (rR < tolv)
          & (tstar > T[:, 2]) & (tstar < T[:, 3]))
    # verification points hug t*: any second bend must lie inside (t*-e, t*+e) to
    # escape the check, and then the intersection error is < 2e. R/8 let ~2% of
    # points through with 1e-7..1e-5 errors; R/500 -> ~0.03%, < 1e-8.
    e = R / 500
    tm, tp = tstar - e, tstar + e
    Pv = torch.stack([X0 + tm[:, None] * U, X0 + tp[:, None] * U], 1)
    Yv = oracle(Pv.reshape(-1, d)).reshape(n, 2, -1)
    resL = (Yv[:, 0] - (aL + sL * tm[:, None])).abs().amax(1)
    resR = (Yv[:, 1] - (aR + sR * tp[:, None])).abs().amax(1)
    ok &= (resL < tolv) & (resR < tolv)
    return X0 + tstar[:, None] * U, ok


@torch.no_grad()
def polish_neuron(oracle, guess, l, j, gen, eps=1e-5, need=None, verbose=False,
                  max_seed_mult=200):
    """Stage 2: from a ~1e-8..1e-6 guess (e.g. the seal refiner's output), reach
    the fp64 floor. The bracket around the guessed kink is now ~1e-6 wide, so it
    is free of foreign bends (~1e-4 per bracket): every kink point costs 4
    queries, no scan, no fingerprint; a rare foreign point is dropped by the
    trimmed solve. ~8*(Din+40) queries per neuron. Prefix error enters x1
    (forward eval only, no inversion). Returns (w, b, H) or None."""
    Din = guess.layers[l].weight.shape[1]
    need = need or Din + 40
    Hs = []; Ts = []; got = 0; n_seed = 0; empty = 0
    while got < need:
        if n_seed > max_seed_mult * need:
            if got < Din + 8:
                return None
            break
        X0, U, R = seed_brackets(guess, l, j, 4 * need, gen, eps=eps)
        n_seed += 4 * need
        if len(X0) == 0:
            empty += 1
            if empty >= 5:                      # bracket too wide for isolation
                eps *= 0.5; empty = 0
            continue
        X0, U, R = X0[:need - got], U[:need - got], R[:need - got]
        Xs, ok = locate_light(oracle, X0, U, R)
        Hs.append(_phi(guess, Xs[ok], l)); got += int(ok.sum())
        Ts.append((((Xs - X0) * U).sum(1) / R)[ok].abs())
        if verbose:
            print(f"    j={j} polish batch: {len(X0)} brackets, {int(ok.sum())} located, eps={eps:.2g}", flush=True)
        # adapt the bracket to the observed offsets (ours is Gaussian about the
        # guessed kink): clipping at the edge -> widen; deep inside -> narrow
        t_all = torch.cat(Ts)
        if len(t_all) >= 30:
            q95 = t_all.quantile(0.95).item()
            if q95 > 0.4:
                eps *= 2.0; Ts = []
            elif q95 < 0.05:
                eps *= 0.3; Ts = []
        elif len(ok) >= 40 and ok.double().mean() < 0.1:
            eps *= 2.0
    H = torch.cat(Hs)
    w, b, kept, gap = solve_from_points(H, guess.layers[l].weight[j], gen=gen)
    return w, b, H


@torch.no_grad()
def polish_layer(oracle, guess, l, channels, gen, eps0=1e-5, need=None,
                 max_rows=200000, max_rounds=60, verbose=False, full=False, trace=None,
                 return_points=False, scales=(1.0,)):
    """Kink points for MANY channels at once: every round seeds brackets for
    all channels still short of `need` points (per-channel eps; for a conv
    layer each seed picks a random output position of the channel), locates
    them in ONE batched oracle pass (full: scan+fingerprint from a ~1e-2 guess;
    else the 8-query light locate from a ~1e-8 row), and adapts each channel's
    bracket width from its observed kink offsets. Returns {c: (w, b, H)} for
    the channels that solved, or with return_points {c: (X, units)}."""
    M = _as_model(guess, l)
    dev = M.device
    Din = M.din
    need = need or Din + 40
    chans = list(channels)
    got = {c: 0 for c in chans}; eps = {c: eps0 for c in chans}
    yld = {c: 0.5 for c in chans}                       # located / brackets, per channel
    Hs = {c: [] for c in chans}; Xp = {c: [] for c in chans}; Up = {c: [] for c in chans}
    Ts = {c: [] for c in chans}; tried = {c: 0 for c in chans}
    seeds = {c: 0 for c in chans}
    for rnd in range(max_rounds):
        todo = [c for c in chans if got[c] < need and seeds[c] <= 200 * need]
        if not todo:
            break
        # seeds per channel: what it still needs over its observed yield (x2 margin)
        per = {c: min(40000, int(2.0 * (need - got[c]) / max(yld[c], 0.02)) + 8) for c in todo}
        tot = sum(per.values())
        if tot > max_rows:                                   # cap the round
            f = max_rows / tot
            per = {c: max(8, int(v * f)) for c, v in per.items()}
        jidx = torch.cat([M.units_of(c, per[c], gen) for c in todo])
        erow = torch.cat([torch.full((per[c],), eps[c], device=dev, dtype=torch.float64) for c in todo])
        for c in todo:
            seeds[c] += per[c]
        X0, U, R, jk = seed_brackets(M, l, jidx, len(jidx), gen, eps=erow,
                                     isolate_prefix=not full,
                                     x_scale=scales[rnd % len(scales)])   # multiscale seeds
        if len(X0):
            # query only ~1.3x what each channel still needs (seeding overshoots)
            ck = M.channel(jk)
            order = torch.argsort(ck, stable=True)
            ck_s = ck[order]
            cnt = torch.bincount(ck_s, minlength=M.n_channels)
            first = torch.cumsum(cnt, 0) - cnt
            rank = torch.arange(len(ck_s), device=dev) - first[ck_s]
            lim = torch.zeros(M.n_channels, device=dev, dtype=torch.long)
            for c in todo:
                lim[c] = int(1.3 * (need - got[c])) + 2
            sel = order[rank < lim[ck_s]]
            X0, U, R, jk = X0[sel], U[sel], R[sel], jk[sel]
        if len(X0) == 0:
            for c in todo:
                tried[c] += 1
                if tried[c] >= 5:
                    eps[c] *= 0.5; tried[c] = 0
            continue
        Xs = torch.empty_like(X0); ok = torch.zeros(len(X0), dtype=torch.bool, device=dev)
        ch = 2048 if full else 16384
        for a in range(0, len(X0), ch):                        # oracle chunks
            if full:                                           # scan + fingerprint (from a ~1e-2 guess)
                xs_, ok_ = locate(oracle, X0[a:a + ch], U[a:a + ch], R[a:a + ch], gen, M, l,
                                  M.rows(jk[a:a + ch])[0], K=25 + 12 * l, jidx=jk[a:a + ch])
            else:                                              # 8-query light locate (from a ~1e-8 row)
                xs_, ok_ = locate_light(oracle, X0[a:a + ch], U[a:a + ch], R[a:a + ch])
            Xs[a:a + ch] = xs_; ok[a:a + ch] = ok_
        H_all = M.feat(Xs, jk)
        ck = M.channel(jk)
        toff = (((Xs - X0) * U).sum(1) / R).abs()
        for c in todo:
            m = (ck == c)
            mo = m & ok
            k = int(mo.sum())
            if trace is not None and c in trace:
                trace[c].append((rnd, per[c], int(m.sum()), k, eps[c], got[c]))
            if int(m.sum()) >= 8:
                yld[c] = 0.5 * yld[c] + 0.5 * k / int(m.sum())
            if k == 0:
                if int(m.sum()) >= 40 and eps[c] < 8 * eps0:
                    eps[c] *= 2.0
                continue
            take = min(k, need - got[c])
            Hs[c].append(H_all[mo][:take]); Xp[c].append(Xs[mo][:take]); Up[c].append(jk[mo][:take])
            got[c] += take
            Ts[c].append(toff[mo])
            t_all = torch.cat(Ts[c])
            if len(t_all) >= 30:
                q95 = t_all.quantile(0.95).item()
                if q95 > 0.4 and eps[c] < 8 * eps0:            # kinks piling at the scan edge
                    eps[c] *= 1.5; Ts[c] = []
                elif q95 < 0.05 and not full:                    # (polish only) bracket far too wide
                    eps[c] *= 0.3; Ts[c] = []
            elif int(m.sum()) >= 40 and k < 0.1 * int(m.sum()) and eps[c] < 8 * eps0:
                eps[c] *= 2.0
        if verbose:
            print(f"    [polish] round {rnd}: {len(todo)} channels, {len(X0)} brackets, "
                  f"{int(ok.sum())} located, {sum(1 for c in chans if got[c] >= need)}/{len(chans)} done",
                  flush=True)
    if return_points:
        return {c: ((torch.cat(Xp[c]), torch.cat(Up[c])) if Xp[c] else None) for c in chans}
    out = {}
    for c in chans:
        if got[c] < Din + 8:
            continue
        H = torch.cat(Hs[c])
        w, b, kept, gap = solve_from_points(H, M.W[c], gen=gen)
        out[c] = (w, b, H)
    return out


def _stage1_worker(task):
    """Pool task: (payload, chunk) -> ({c: (w, b)}, n_queries). CPU only and
    fully grad-DISABLED: the seal refiner is forward-only, and with no autograd
    engine use a forked child is safe even after the parent ran autograd (and
    fork, unlike spawn, never re-imports the caller's main module)."""
    payload, chunk = task
    import torch as _t
    _t.set_num_threads(1)
    _t.set_grad_enabled(False)
    import method
    from nets import MLP
    T = MLP(payload["dims"], act=payload["act"]).double(); T.load_state_dict(payload["t_state"]); T.eval()
    C = MLP(payload["dims"], act=payload["act"]).double(); C.load_state_dict(payload["c_state"]); C.eval()
    for p in list(T.parameters()) + list(C.parameters()):
        p.requires_grad_(False)
    W1, b1, m1, nq = method._mlp_refine_layer(T, C, payload["frontier"], "cpu", payload["act"],
                                              only_channels=chunk, angle_gate=payload["angle_gate"],
                                              xspace=False, loc=False, deep_tries=40)
    res = {}
    if W1 is not None:
        for c in chunk:
            if bool(m1[c]):
                res[c] = (W1[c].clone(), b1[c].clone())
    return res, nq


def stage1_parallel(teacher, cons, frontier, act, channels, angle_gate=12.0, n_workers=None):
    """The existing seal refiner (_mlp_refine_layer) on CPU, channels split
    over a fork pool: it is a chain of ~500 tiny sequential oracle calls per
    neuron, so it is latency-bound on a GPU and embarrassingly parallel on CPU
    cores. Returns ({c: (w, b)}, n_queries)."""
    import multiprocessing as mp, os
    n_workers = n_workers or max(1, min(len(channels), (os.cpu_count() or 8) - 2, 40))
    payload = dict(dims=list(teacher.dims), act=act, frontier=frontier, angle_gate=angle_gate,
                   t_state={k: v.detach().double().cpu() for k, v in teacher.state_dict().items()},
                   c_state={k: v.detach().double().cpu() for k, v in cons.state_dict().items()})
    chunks = [list(channels[i::n_workers]) for i in range(n_workers)]
    chunks = [ch for ch in chunks if ch]
    if len(chunks) == 1:
        return _stage1_worker((payload, chunks[0]))
    with mp.get_context("fork").Pool(len(chunks)) as pool:
        outs = pool.map(_stage1_worker, [(payload, ch) for ch in chunks])
    res = {}; nq = 0
    for r, q in outs:
        res.update(r); nq += q
    return res, nq


@torch.no_grad()
def track_layer(oracle, guess, l, anchors, gen, need=None, step=0.5, max_steps=None,
                verbose=False, par=4):
    """From >= 1 exact kink point per channel, collect `need` points per channel
    by TRACKING the kink surface: step along a random tangent (model), project
    onto the model's surface SHIFTED by the last known model-vs-true offset,
    re-locate the true kink along the normal in a bracket sized by how much
    that offset can change over one step (~|dw| |dh| / sqrt(Din) / |n|),
    8 queries, no scan, no fingerprint, no re-identification. `par` walks per
    channel, all channels in lockstep. anchors: {c: (X, units)}. Each tracked
    point stays on the surface of the SAME unit (for a conv layer: the same
    output position). Returns {c: H (n_c, Din)} for channels that reached need."""
    M = _as_model(guess, l)
    dev, d, Din = M.device, M.d, M.din
    need = need or Din + 40
    max_steps = max_steps or 4 * need
    chans = [c for c, a in anchors.items() if a is not None and len(a[0])]
    P = torch.zeros(len(chans), need, d, device=dev, dtype=torch.float64)   # points per channel
    PU = torch.zeros(len(chans), need, dtype=torch.long, device=dev)         # their units
    got = torch.zeros(len(chans), dtype=torch.long, device=dev)
    for i, c in enumerate(chans):
        Xa, Ua = anchors[c]
        k = min(len(Xa), need); P[i, :k] = Xa[:k]; PU[i, :k] = Ua[:k]; got[i] = k
    for it in range(max_steps):
        act = (got < need).nonzero()[:, 0]
        if len(act) == 0:
            break
        act = act.repeat_interleave(par)                    # `par` walks per channel per step
        r = (torch.rand(len(act), device=dev, generator=gen) * got[act]).long()
        X = P[act, r]; u = PU[act, r]
        wg, bg = M.rows(u)
        g0, n0 = _g_and_normal(M, X, l, wg, bg, u)          # model offset at a TRUE kink point
        nh = n0 / n0.norm(dim=1, keepdim=True).clamp_min(1e-30)
        V = torch.randn(X.shape, device=dev, dtype=X.dtype, generator=gen)
        V = V - (V * nh).sum(1, keepdim=True) * nh
        V = V / V.norm(dim=1, keepdim=True).clamp_min(1e-30)
        Xn = X + step * V
        for _ in range(6):                                  # project onto the SHIFTED model surface
            g, n = _g_and_normal(M, Xn, l, wg, bg, u)
            Xn = Xn - ((g - g0) / (n * n).sum(1).clamp_min(1e-30))[:, None] * n
        g, n = _g_and_normal(M, Xn, l, wg, bg, u)
        nrm = n.norm(dim=1).clamp_min(1e-30); U = n / nrm[:, None]
        dh = (M.feat(Xn, u) - M.feat(X, u)).norm(dim=1)
        R = 5.0 * (1e-2 * dh / (Din ** 0.5) + 1e-7) / nrm
        Xs, ok = locate_light(oracle, Xn, U, R)
        lane = torch.arange(len(act), device=dev) % par
        for i in range(par):                                # store walk i's hits, then count
            sel = ok & (lane == i)
            hit = act[sel]
            room = got[hit] < need
            hit = hit[room]
            P[hit, got[hit]] = Xs[sel][room]; PU[hit, got[hit]] = u[sel][room]
            got[hit] += 1
        if verbose and (it % 20 == 0):
            print(f"    [track] step {it}: {len(act) // par} channels, {int(ok.sum())}/{len(act)} located, "
                  f"{int((got >= need).sum())}/{len(chans)} done", flush=True)
    out = {}
    for i, c in enumerate(chans):
        k = int(got[i])
        if k >= Din + 8:
            out[c] = M.feat(P[i, :k], PU[i, :k])
    return out


def recover_layer(teacher, cons, frontier, device, only_channels=None, gen=None,
                  angle_gate=12.0, polish=True, fallback=True, verbose=True,
                  n_workers=None, direct=True, sampling="design", **_ignored):
    """Pipeline entry point: recover layer `frontier` of `cons` (layers <
    frontier = the near-exact prefix, layer `frontier` = the ~1e-2 guess) from
    black-box `teacher`. `cons` may be an MLP or a nets.ConvNet (conv or FC
    frontier; conv units are (channel, position), every kink point constrains
    the shared filter).
    sampling="design" (MLP default): peel/informative_kinks -- multiscale
      seeds, pivoted-QR information-directed choice of brackets, held-out gate
      (validated recursively on a 7x200 net, 200/200 every layer, worst 2.5e-11).
    sampling="track" (ConvNet default, MLP alternative): ONE scan+fingerprint
      round for anchor kinks, then surface tracking (8 queries/point), then the
      trimmed null-vector solve. Forward-only; no prefix inversion anywhere.
    direct=False: legacy two-stage path (seal refiner -> polish), MLP only.
    Returns (W_ref, b_ref, refined_mask, n_oracle_queries); W_ref has the
    layer's weight shape, refined rows are unit [w|b]."""
    is_conv = hasattr(cons, "n_conv")
    if is_conv and sampling == "design":
        if verbose:
            print("    [kink] ConvNet layer: design sampling is MLP-only -> tracking", flush=True)
        sampling = "track"
    if is_conv:
        direct = True
    if (sampling == "design" and frontier == 0 and len(cons.layers) == 2
            and cons.layers[0].weight.shape[1] >= 1024
            and cons.layers[0].weight.shape[0] <= cons.layers[0].weight.shape[1]):
        from peel.shallow_sweep import recover_layer as recover_shallow
        if verbose:
            print('[design-refine] wide-input shallow network: isolated affine sweeps', flush=True)
        attempts = 0 if _ignored.get("max_rounds") == 0 else _ignored.get("sweep_attempts", 6)
        return recover_shallow(teacher, cons, device, only_channels=only_channels,
                               gen=gen, verbose=verbose, angle_gate=angle_gate,
                               diagnostics=_ignored.get("diagnostics"), attempts=attempts)
    if sampling == "design":
        from peel.informative_kinks import recover_layer as recover_designed
        W, b, mask, nq = recover_designed(teacher, cons, frontier, device,
                                          only_channels=only_channels, gen=gen,
                                          angle_gate=angle_gate, verbose=verbose, **_ignored)
        norm = (W.square().sum(1) + b.square()).sqrt().clamp_min(1e-30)
        W[mask] = W[mask] / norm[mask, None]
        b[mask] = b[mask] / norm[mask]
        return W, b, mask, nq
    if sampling != "track":
        raise ValueError(f"Unknown kink sampling strategy: {sampling}")
    import copy, math, time
    from verify_layer1 import _Oracle
    act = getattr(cons, "act_name", "leaky_relu")
    pre = copy.deepcopy(cons).double().to(device).eval()
    orc = _Oracle(copy.deepcopy(teacher).double().to(device).eval())
    if gen is None:
        gen = torch.Generator(device=device).manual_seed(0)
    M = _as_model(pre, frontier)
    wshape = M.weight_shape()
    Wl = M.W.detach().clone(); bl = M.B.detach().clone()      # (n_channels, Din) rows
    Cout, Din = Wl.shape
    W_ref = Wl.clone(); b_ref = bl.clone()
    refined = torch.zeros(Cout, dtype=torch.bool, device=device)
    todo = list(range(Cout)) if only_channels is None else list(only_channels)
    t0 = time.time()

    def _set_rows(W, b):                                     # write rows into pre's layer
        with torch.no_grad():
            pre.layers[frontier].weight.copy_(W.reshape(wshape).to(pre.layers[frontier].weight.dtype))
            pre.layers[frontier].bias.copy_(b.to(pre.layers[frontier].bias.dtype))

    if direct:
        rows, nq1 = {}, 0
        polish = False
    else:
        rows, nq1 = stage1_parallel(teacher, cons, frontier, act, todo, angle_gate, n_workers)
    for c, (w, b) in rows.items():
        W_ref[c] = w.to(device=device, dtype=W_ref.dtype); b_ref[c] = b.to(device=device, dtype=b_ref.dtype)
        refined[c] = True
    n1 = len(rows)
    if verbose and not direct:
        print(f"    [kink] stage 1 (seal refiner, cpu pool): {n1}/{len(todo)} refined, "
              f"{nq1} q, {time.time() - t0:.1f}s", flush=True)
    _set_rows(W_ref, b_ref)

    def _gate(c, w, b, H, ref_w, ref_b, max_deg):
        A = torch.cat([H, torch.ones(len(H), 1, device=device, dtype=H.dtype)], 1)
        res = (A[:, :Din] @ w + b).abs()
        thr = max(100.0 * res.median().item(), 1e-12)
        n_in = int((res <= thr).sum())
        gv = torch.cat([ref_w, ref_b.reshape(1)]); gv = gv / gv.norm().clamp_min(1e-30)
        v = torch.cat([w, b.reshape(1)]); v = v / v.norm()
        ang = math.degrees(math.acos(abs(float((v @ gv).clamp(-1.0, 1.0)))))
        rmed = res.median().item()
        # a kink-point solve is either EXACT (residual ~ prefix error, <=1e-7 for a
        # 1e-8 prefix) or a wrong lock (points from several kinks: residual >=1e-4,
        # inliers a fraction of the points). The angle gate alone let a 3e-2-wrong
        # row through when the consensus guess was 1e-1 off.
        ok = (n_in >= Din + 8 and n_in >= 0.8 * len(H) and ang <= max_deg and rmed <= 1e-5)
        return ok, v, n_in, ang, rmed

    n2 = 0
    if polish and n1:                                        # legacy stage 2
        t1 = time.time(); q0 = orc.n
        out = polish_layer(orc, pre, frontier, sorted(rows), gen, verbose=verbose > 1)
        gates = {c: _gate(c, w, b, H, W_ref[c], b_ref[c], 1.0) for c, (w, b, H) in out.items()}
        floor = sorted(g[4] for g in gates.values())[len(gates) // 2] if gates else 0.0
        for c, (w, b, H) in out.items():
            ok, v, n_in, ang, rmed = gates[c]
            ok = ok and rmed <= max(100.0 * floor, 1e-13)
            if ok:
                W_ref[c] = v[:-1]; b_ref[c] = v[-1]; n2 += 1
            else:
                W_ref[c] = Wl[c]; b_ref[c] = bl[c]; refined[c] = False
                if verbose:
                    print(f"    [kink] c={c}: polish REJECT (inliers {n_in}/{len(H)} "
                          f"res_med {rmed:.1e} angle {ang:.1e}deg) -> fallback", flush=True)
        if verbose:
            print(f"    [kink] stage 2 (batched polish): {n2}/{n1} improved, {orc.n - q0} q, "
                  f"{time.time() - t1:.1f}s", flush=True)
    if fallback:
        rest = [c for c in todo if not bool(refined[c])]
        if rest:
            t2 = time.time(); q0 = orc.n
            _set_rows(Wl, bl)
            M = _as_model(pre, frontier)
            if direct:
                # (a) ONE scan+fingerprint round -> a few exact anchor kinks per channel
                anchors = polish_layer(orc, M, frontier, rest, gen, eps0=0.02, full=True,
                                       need=48, max_rounds=1, verbose=verbose > 1,
                                       return_points=True)
                empty = [c for c in rest if anchors.get(c) is None]
                if empty:                                  # a few more rounds, only for them
                    more = polish_layer(orc, M, frontier, empty, gen, eps0=0.02, full=True,
                                        need=48, max_rounds=12, verbose=verbose > 1,
                                        return_points=True, scales=(1.0, 4.0, 16.0))
                    anchors.update({c: v for c, v in more.items() if v is not None})
                if verbose:
                    na = [len(v[0]) for v in anchors.values() if v is not None]
                    print(f"    [kink] anchors: {sum(1 for v in anchors.values() if v is not None)}/{len(rest)} "
                          f"channels, median {sorted(na)[len(na) // 2] if na else 0} kinks, "
                          f"{orc.n - q0} q, {time.time() - t2:.1f}s", flush=True)
                # (b) TRACK the surface from the anchors to Din+40 points per channel
                # tracking step: 0.5 for MLP layers (validated); conv inputs are larger
                # (|x| ~ sqrt(d)) and patch features decorrelate only over bigger
                # moves, so step ~0.3 sqrt(d) (10x better conditioning at conv3)
                Hs = track_layer(orc, M, frontier, anchors, gen, need=Din + 40,
                                 verbose=verbose > 1,
                                 step=0.15 * (M.d ** 0.5) if is_conv else 0.5)
                out = {}
                for c, H in Hs.items():
                    w, b, kept, gap = solve_from_points(H, Wl[c], gen=gen)
                    out[c] = (w, b, H)
            else:
                out = polish_layer(orc, M, frontier, rest, gen, eps0=0.02, full=True,
                                   verbose=verbose > 1)
            nf = 0
            for c in rest:
                if c not in out:
                    if verbose:
                        print(f"    [kink] c={c}: fallback found too few kinks", flush=True)
                    continue
                w, b, H = out[c]
                ok, v, n_in, ang, rmed = _gate(c, w, b, H, Wl[c], bl[c], angle_gate)
                if ok:
                    W_ref[c] = v[:-1]; b_ref[c] = v[-1]; refined[c] = True; nf += 1
                elif verbose:
                    print(f"    [kink] c={c}: fallback reject inliers {n_in}/{len(H)} angle {ang:.2f}deg", flush=True)
            if verbose:
                print(f"    [kink] {'direct' if direct else 'fallback'} (batched scan+fingerprint): {nf}/{len(rest)} solved, "
                      f"{orc.n - q0} q, {time.time() - t2:.1f}s", flush=True)
    if verbose:
        print(f"    [kink] layer {frontier}: {int(refined[todo].sum())}/{len(todo)} refined, "
              f"{orc.n + nq1} q total, {time.time() - t0:.1f}s", flush=True)
    return (W_ref.reshape(wshape).to(cons.layers[frontier].weight.dtype),
            b_ref.to(cons.layers[frontier].bias.dtype), refined, orc.n + nq1)
