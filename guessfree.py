"""Guess-free frontier recovery for the peel pipeline (wraps diag_guessfree_layer.py).

recover_frontier_rows(teacher, prefix_net, K, known_channels, ...) -> (rows, n_queries, info)
  teacher      : the black box (any dtype; queried through an fp64 clone, forward only)
  prefix_net   : ConvNet whose layers < K are the EXACT recovered prefix (the fp64 partial record);
                 layer K's `known_channels` rows must be exact too (their kinks are excluded)
  rows         : list of unit [w|b] filter rows (frontier-input gauge of prefix_net), best-supported
                 first (cluster size), each a different hyperplane, none equal to a known channel
See diag_guessfree_layer.py for the method and the validation (LeNet C5 92/120 exact from 30 lines;
L2 straggler from one line; real-run snapshot: unsolved channel recovered to 1.9e-5).
"""
import math, time
import torch
import diag_guessfree_layer as G


def _extreme_base(net, K, d_in, x0_std, gen, device, steps=40, lr=0.1, box=3.0):
    """attacker-legal base point for a scan line: push a random image so that phi(x) is extreme along a random
    direction of the frontier-input space (visits the reachable set's boundary, where one-sided frontier units flip)."""
    phi = G.prefix_fn(net, K) if hasattr(G, "prefix_fn") else None
    D = G._phi_shape(net, K); D = int(torch.tensor(D).prod())
    u = torch.randn(D, device=device, dtype=torch.double, generator=gen); u = u / u.norm()
    sgn = 1.0 if torch.rand(1, generator=gen, device=device).item() < 0.5 else -1.0
    x = (torch.randn(1, d_in, device=device, dtype=torch.double, generator=gen) * x0_std).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        opt.zero_grad(); ph = G.prefix_feats(net, x, K)[0]; (-(sgn * (ph @ u))).sum().backward(); opt.step()
        with torch.no_grad(): x.clamp_(-box, box)
    return x.detach()[0]


def recover_frontier_rows(teacher, prefix_net, K, known_channels, lines=10, N=40001, T=3.0, x0_std=0.5,
                          max_kinks=400, device="cuda", seed=0, verbose=True, min_cluster=1, stop_at=None,
                          verify_tests=8, verify_min=0.85, extreme=False):
    t64 = teacher.clone().double()
    for p in t64.parameters():
        p.requires_grad_(False)
    net = prefix_net
    if next(net.parameters()).dtype != torch.float64:
        net = net.clone().double()
    for p in net.parameters():
        p.requires_grad_(False)
    gen = torch.Generator(device=device).manual_seed(seed)
    d_in = int(torch.tensor(net.input_shape).prod()) if hasattr(net, "input_shape") else net.layers[0].weight.shape[1]
    blocks = G.rf_blocks(net, K)
    q0 = G.QUERIES[0]
    found = []
    t0 = time.time()
    # known rows (unit, filter space) to reject duplicates of already-solved channels
    lay = net.layers[K]
    Wk = torch.cat([lay.weight.reshape(lay.weight.shape[0], -1), lay.bias[:, None]], 1).double()
    Wk = Wk / Wk.norm(dim=1, keepdim=True).clamp_min(1e-30)
    known_rows = Wk[list(known_channels)] if len(known_channels) else None
    for li in range(lines):
        if extreme:
            x0 = _extreme_base(net, K, d_in, x0_std, gen, device)
        else:
            x0 = torch.randn(d_in, device=device, dtype=torch.double, generator=gen) * x0_std
        d = torch.randn(d_in, device=device, dtype=torch.double, generator=gen)
        d = d / d.norm() * math.sqrt(d_in) * x0_std
        t_unk, ncand, dt_grid = G.scan_line(t64, net, K, x0, d, T, N, list(known_channels))
        n_ok = 0
        for t in t_unk[:max_kinks]:
            ts = G.bisect_kink(t64, x0, d, t - dt_grid, t + dt_grid)
            x_star = x0 + ts * d
            n, ratio, s0 = G.jump_normal(t64, x0, d, ts, 1e-5, 1e-6)
            if ratio < 1e3:
                continue
            row, rel = G.mapback(net, K, x_star, n, blocks)
            if rel > 1e-3:
                continue
            if known_rows is not None and bool(((known_rows @ row).abs() > 0.999).any()):
                continue                                   # a known channel's kink that slipped the exclusion
            found.append(row); n_ok += 1
        clusters = _cluster(found)
        if verbose:
            print(f"    [guess-free] L{K + 1} line {li + 1}/{lines}: {ncand} kinks, {len(t_unk)} unexplained, {n_ok} mapped back; "
                  f"{sum(1 for c in clusters if len(c) >= min_cluster)} recurring clusters | queries {G.QUERIES[0] - q0:,} | {time.time() - t0:.0f}s",
                  flush=True)
        # early stop counts RECURRING clusters only (singletons are mostly deeper-layer kinks that verification
        # will reject); stopping on all clusters ended a 72-channel attempt after 2 lines with 22 accepted rows
        if stop_at is not None and sum(1 for c in clusters if len(c) >= max(2, min_cluster)) >= stop_at:
            break
    clusters = _cluster(found)
    rows = []; n_cand = 0; n_rej = 0
    for m in sorted(clusters, key=len, reverse=True):
        if len(m) < min_cluster:
            continue
        R = torch.stack([found[i] for i in m])
        s = torch.sign(R @ R[0]); r = (R * s[:, None]).mean(0); r = r / r.norm()
        if any(bool((r @ q).abs() > 0.999) for q in rows):
            continue
        n_cand += 1
        # GLOBAL consistency: a deeper layer's kink (F6) also maps back cleanly and can recur inside one frontier
        # activation region, but its hyperplane is only valid there; a true frontier row has a black-box kink
        # wherever w.phi(x)+b = 0. Test that at fresh random points (cheap: ~4 black-box queries per test).
        frac = verify_row(t64, net, K, r, device, gen, n_tests=verify_tests, x0_std=x0_std)
        if frac < verify_min:
            n_rej += 1
            continue
        rows.append(r)
    info = dict(lines=li + 1, n_found=len(found), clusters=sorted([len(c) for c in clusters], reverse=True)[:20],
                candidates=n_cand, rejected_by_verification=n_rej, wall=time.time() - t0)
    if verbose:
        print(f"    [guess-free] L{K + 1}: {n_cand} candidate rows (clusters >= {min_cluster}), {n_rej} rejected by the global "
              f"kink test, {len(rows)} accepted | queries {G.QUERIES[0] - q0:,}", flush=True)
    return rows, G.QUERIES[0] - q0, info


def _frontier_preact(net, K, row, X):
    """pre-activation(s) of a hypothetical frontier unit with unit [w|b] = row on inputs X: (N, positions)."""
    import torch.nn.functional as F
    ph, _ = G.prefix_feats(net, X, K)
    lay = net.layers[K]
    w, b = row[:-1], row[-1]
    if lay.weight.dim() == 4:
        C, H, W = G._phi_shape(net, K)
        z = F.conv2d(ph.view(ph.shape[0], C, H, W), w.view(1, *lay.weight.shape[1:]), b.view(1), stride=lay.stride, padding=lay.padding)
        return z.reshape(z.shape[0], -1)
    return (ph @ w + b)[:, None]


@torch.no_grad()
def verify_row(t64, net, K, row, device, gen, n_tests=6, x0_std=0.5, h=1e-6, T=3.0, N=2001):
    """fraction of fresh points on the candidate hyperplane {w.phi(x)+b=0} at which the black box has a kink."""
    d_in = int(torch.tensor(net.input_shape).prod()) if hasattr(net, "input_shape") else net.layers[0].weight.shape[1]
    hits = 0; tried = 0; attempts = 0
    while tried < n_tests and attempts < 4 * n_tests:
        attempts += 1
        x0 = torch.randn(d_in, device=device, dtype=torch.double, generator=gen) * x0_std
        d = torch.randn(d_in, device=device, dtype=torch.double, generator=gen); d = d / d.norm() * math.sqrt(d_in) * x0_std
        ts = torch.linspace(-T, T, N, device=device, dtype=torch.double)
        g = _frontier_preact(net, K, row, x0 + ts[:, None] * d)          # (N, positions), exact prefix, no queries
        cross = ((torch.sign(g[1:]) != torch.sign(g[:-1])) & (g[1:] != 0)).nonzero()
        if cross.numel() == 0:
            continue
        i, p = cross[torch.randint(len(cross), (1,), device=device, generator=gen)][0].tolist()
        lo, hi = float(ts[i]), float(ts[i + 1])
        glo = float(g[i, p])
        for _ in range(60):                                                 # bisection on the exact prefix (no queries)
            mid = 0.5 * (lo + hi)
            gm = float(_frontier_preact(net, K, row, (x0 + mid * d)[None])[0, p])
            if (gm > 0) == (glo > 0):
                lo = mid; glo = gm
            else:
                hi = mid
        tstar = 0.5 * (lo + hi)
        # the candidate row is only ~1e-6 accurate, so its root is displaced ~1e-5 in t from the true kink:
        # scan a small window around t* and ask whether the black box has a kink within `tol` of it.
        # (prefix kinks are ~1e-3 apart, so a random one lands inside +-tol with prob ~0.25; requiring
        #  >= 85% of 8 tests makes a false row pass with prob ~1e-3.)
        w, npts, tol = 3e-4, 61, 1e-4
        tw = tstar + torch.linspace(-w, w, npts, device=device, dtype=torch.double)
        Y = G.bb(t64, x0 + tw[:, None] * d)
        s = (Y[1:] - Y[:-1]) / (tw[1] - tw[0])
        jump = (s[1:] - s[:-1]).norm(dim=1)
        thr = 1e-6 * max(1.0, float(s.norm(dim=1).median()))
        cand = (jump > thr).nonzero().flatten()
        if cand.numel():
            tk = tw[cand + 1]
            kink = bool((tk - tstar).abs().min() <= tol)
        else:
            kink = False
        hits += int(kink); tried += 1
    return hits / tried if tried else 0.0


def _cluster(found, thr=0.999):
    if not found:
        return []
    R = torch.stack(found)
    cos = (R @ R.T).abs()
    assigned = -torch.ones(len(R), dtype=torch.long)
    clusters = []
    for i in range(len(R)):
        if assigned[i] >= 0:
            continue
        members = ((cos[i] > thr).cpu() & (assigned < 0)).nonzero().flatten()
        assigned[members] = len(clusters)
        clusters.append(members.tolist())
    return clusters


@torch.no_grad()
def normal_consistency(t64, net, K, row, device, gen, n_points=3, x0_std=0.5, eps_jac=1e-6, eps_side=1e-5, base_points=None):
    """Second certificate for a frontier candidate (2026-10-16): a TRUE frontier row has the same kink normal in the
    frontier's input space at every point of its hyperplane; a deeper layer's kink surface (F6 seen from conv2 space)
    is only piecewise a hyperplane and its normal changes between activation regions of the frontier. For n_points
    roots of w.phi(x)+b = 0 found from different base points, measure the black-box Jacobian jump across the surface
    (central differences along the d_h prefix-mapped directions), take its leading right singular vector in h-space
    and return the min pairwise |cos| between those normals (and the min |cos| to the candidate row's w).
    Channel 4 of the lenet_residual record (an F6 surface that passed verify_row 1.00) -> inconsistent normals."""
    d_in = int(torch.tensor(net.input_shape).prod()) if hasattr(net, "input_shape") else net.layers[0].weight.shape[1]
    w = row[:-1].double(); b = row[-1].double(); d_h = w.numel()
    normals = []
    tries = 0
    while len(normals) < n_points and tries < 6 * n_points:
        tries += 1
        if base_points is not None:
            x0 = base_points[int(torch.randint(base_points.shape[0], (1,), generator=gen, device=device))].to(device).double()
        else:
            x0 = torch.randn(d_in, device=device, dtype=torch.double, generator=gen) * x0_std
        d = torch.randn(d_in, device=device, dtype=torch.double, generator=gen); d = d / d.norm() * math.sqrt(d_in) * x0_std
        ts = torch.linspace(-3.0, 3.0, 2001, device=device, dtype=torch.double)
        g = _frontier_preact(net, K, row, x0 + ts[:, None] * d)[:, 0]
        cross = ((torch.sign(g[1:]) != torch.sign(g[:-1])) & (g[1:] != 0)).nonzero().flatten()
        if cross.numel() == 0:
            continue
        i = int(cross[torch.randint(len(cross), (1,), device=device, generator=gen)])
        lo, hi = float(ts[i]), float(ts[i + 1]); glo = float(g[i])
        for _ in range(60):
            mid = 0.5 * (lo + hi); gm = float(_frontier_preact(net, K, row, (x0 + mid * d)[None])[0, 0])
            if (gm > 0) == (glo > 0): lo = mid; glo = gm
            else: hi = mid
        xs = x0 + 0.5 * (lo + hi) * d
        # prefix Jacobian at xs -> directions in x that move phi along unit vectors (first order, within the region)
        xr = xs.clone().requires_grad_(True)
        with torch.enable_grad():
            ph = G.prefix_feats(net, xr[None], K)[0].reshape(-1)
            Jp = torch.stack([torch.autograd.grad(ph[j], xr, retain_graph=True)[0] for j in range(d_h)])
        Dx = torch.linalg.pinv(Jp)                                 # (d_in, d_h)
        # the prefix Jacobian can be ill-conditioned (conv + pooling): keep every x-step at |dx| = eps so no prefix
        # kink is crossed, and divide by the phi-displacement each step actually produces (first order, in-region)
        cn = Dx.norm(dim=0).clamp_min(1e-30)                        # |Dx[:, i]|
        E = (Dx / cn[None, :]).T * eps_jac                          # (d_h, d_in): |row| = eps_jac, moves phi_i by eps_jac / cn_i
        dphi = eps_jac / cn                                         # (d_h,)
        def jac_h(x):
            Yp = G.bb(t64, x[None] + E); Ym = G.bb(t64, x[None] - E)
            return ((Yp - Ym) / (2 * dphi)[:, None]).T             # (d_out, d_h)
        dA = Dx @ w; dA = dA / dA.norm().clamp_min(1e-30) * eps_side
        M = jac_h(xs + dA) - jac_h(xs - dA)
        U_, S_, Vh = torch.linalg.svd(M, full_matrices=False)
        n = Vh[0]; n = n * torch.sign(n @ w)
        normals.append(n)
    if len(normals) < 2:
        return float("nan"), float("nan")
    N = torch.stack(normals)
    C = (N @ N.T).abs(); C.fill_diagonal_(1.0)
    wn = w / w.norm()
    return float(C.min()), float((N @ wn).abs().min())
