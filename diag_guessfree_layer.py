"""GUESS-FREE frontier-layer recovery from kinks (prototype). The committee cannot produce C5
guesses (GD has ~no signal for channels that rarely cross their kink), yet diag_c5_reach shows every
C5/F6 kink is reachable inside the unit cube. So skip the guess: with the prefix (conv1+conv2) EXACT,
  1. scan the black box along random lines x0 + t d, locate every kink (slope change of the
     piecewise-linear output), discard the kinks explained by prefix units (their pre-acts cross
     zero there -- computable from the exact prefix);
  2. at each remaining kink x*, measure the gradient-jump matrix  J_f(x*+) - J_f(x*-) = u n^T
     (rank 1; n = input-space normal of the kink surface) with finite differences;
  3. map n back to frontier-input space through the prefix Jacobian: n = J_phi(x*)^T w (784 eqs,
     D=400 unknowns -> least squares); b = -w . phi(x*). For a frontier CONV with weight sharing
     (L2 straggler case) the row is restricted to one receptive field: try every position.
  4. cluster the unit [w|b] rows across kinks: a C5 row recurs at every crossing of that channel;
     rows from deeper kinks (F6) depend on the activation pattern and do not recur.
Teacher weights are used only for the exact prefix (as a peel run has it), to label known
kinks, and to score. Everything else is forward queries of the black box.

Usage: python diag_guessfree_layer.py --frontier 2 --lines 20 --device cuda:1
       python diag_guessfree_layer.py --frontier 1 --known-channels all-but:6 --lines 10   (L2 straggler)
"""
import argparse, json, math, time
import torch
import torch.nn.functional as F
from data import make_teacher_cnn

CFGS = [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)]
FC = (84,)
QUERIES = [0]


def bb(teacher, x):
    """the black box: forward queries only (counted)."""
    QUERIES[0] += x.shape[0]
    with torch.no_grad():
        return teacher(x)


def prefix_feats(net, x, K):
    """phi(x): flattened input of layer K; plus the pre-activations of layers < K (for kink labelling)."""
    if not hasattr(net, "n_conv"):                      # MLP prefix
        pre = []
        for i in range(K):
            z = net.layers[i](x); pre.append(z); x = net.act(z)
        return x, pre
    x = x.view(x.shape[0], *net.input_shape)
    pre = []
    for i in range(K):
        z = net.layers[i](x); pre.append(z.reshape(z.shape[0], -1))
        x = net.act(z)
        if net.pools[i] > 0:
            x = F.avg_pool2d(x, net.pools[i])
    return x.reshape(x.shape[0], -1), pre


def _phi_shape(net, K):
    """(C, H, W) of the frontier's input map."""
    if not hasattr(net, "n_conv"):
        return (net.layers[K].weight.shape[1],)
    with torch.no_grad():
        x = torch.zeros(1, *net.input_shape, device=net.layers[0].weight.device, dtype=net.layers[0].weight.dtype)
        for i in range(K):
            x = net.act(net.layers[i](x))
            if net.pools[i] > 0:
                x = F.avg_pool2d(x, net.pools[i])
    return tuple(x.shape[1:])


def scan_line(teacher, net, K, x0, d, T, N, known_channels=None):
    """kinks of f along x0 + t d, t in [-T, T]: returns (t_unknown, n_candidates, grid step). Kinks explained by
    prefix units -- and by the frontier channels already solved (`known_channels`, as a partial record) -- are dropped."""
    ts = torch.linspace(-T, T, N, device=x0.device, dtype=x0.dtype)
    X = x0 + ts[:, None] * d
    Y = torch.cat([bb(teacher, X[i:i + 4096]) for i in range(0, N, 4096)])     # (N, out), chunked (GPU memory)
    g = (Y[1:] - Y[:-1]) / (ts[1] - ts[0])               # slopes per interval
    jump = (g[1:] - g[:-1]).norm(dim=1)                  # slope change at interior grid points
    thr = 1e-6 * max(1.0, float(g.norm(dim=1).median()))
    cand = (jump > thr).nonzero().flatten() + 1          # grid index of the kink's grid point
    # known kinks: zero crossings of prefix pre-activations along the line
    with torch.no_grad():
        parts = [prefix_feats(net, X[i:i + 4096], K) for i in range(0, N, 4096)]
        ph = torch.cat([p[0] for p in parts]); pre = [torch.cat([p[1][l] for p in parts]) for l in range(len(parts[0][1]))]
        if known_channels is not None and len(known_channels):
            lay = net.layers[K]
            if lay.weight.dim() == 4:
                zf = lay(ph.view(ph.shape[0], *_phi_shape(net, K)))
            else:
                zf = lay(ph)
            zf = zf[:, known_channels].reshape(zf.shape[0], -1)
            pre = pre + [zf]
    known = torch.zeros(N, dtype=torch.bool, device=x0.device)
    for z in pre:
        cross = (torch.sign(z[1:]) != torch.sign(z[:-1])).any(1)     # interval i crosses
        idx = cross.nonzero().flatten()
        for off in (0, 1, -1):
            known[(idx + off).clamp(0, N - 1)] = True
    unk = [int(i) for i in cand if not bool(known[i])]
    # merge adjacent grid points
    merged = []
    for i in unk:
        if merged and i - merged[-1][-1] <= 1:
            merged[-1].append(i)
        else:
            merged.append([i])
    t_unk = [float(ts[m[len(m) // 2]]) for m in merged]
    return t_unk, int(cand.numel()), ts[1] - ts[0]


def bisect_kink(teacher, x0, d, t_lo, t_hi, iters=40):
    """locate the kink between t_lo and t_hi (slope change) by bisection on the second difference sign."""
    def slope(t, h):
        X = torch.stack([x0 + t * d, x0 + (t + h) * d])
        Y = bb(teacher, X)
        return (Y[1] - Y[0]) / h
    h = (t_hi - t_lo) * 1e-3
    s_lo = slope(t_lo - h, h)                           # slope just below the bracket
    for _ in range(iters):
        tm = 0.5 * (t_lo + t_hi)
        sm = slope(tm, h)
        if (sm - s_lo).norm() > 1e-7 * max(1.0, float(s_lo.norm())):
            t_hi = tm
        else:
            t_lo = tm
        if t_hi - t_lo < 1e-11:
            break
    return 0.5 * (t_lo + t_hi)


def jump_normal(teacher, x0, d, t_star, dt, eps):
    """n: unit input-space normal of the kink at x0 + t* d from the gradient jump (fd on both sides)."""
    D = x0.numel()
    out = []
    for side in (-1.0, 1.0):
        xc = x0 + (t_star + side * dt) * d
        X = torch.cat([xc[None], xc[None] + eps * torch.eye(D, device=x0.device, dtype=x0.dtype)])
        Y = bb(teacher, X)
        out.append(((Y[1:] - Y[0]) / eps).T)             # (out, D) Jacobian
    Jd = out[1] - out[0]
    U, S, Vh = torch.linalg.svd(Jd, full_matrices=False)
    n = Vh[0]
    return n, float(S[0] / max(float(S[1]), 1e-300)), float(S[0])


def rf_blocks(net, K):
    """for a conv frontier with weight sharing: list of index sets into flattened phi (C_in, H, W), one per
    output position, each covering the kernel's receptive field in raster order (c, kh, kw)."""
    if not hasattr(net, "n_conv"):
        return None
    lay = net.layers[K]
    if lay.weight.dim() != 4:
        return None
    cin, kh, kw = lay.weight.shape[1], lay.weight.shape[2], lay.weight.shape[3]
    s, pad = lay.stride[0], lay.padding[0]
    # phi shape: run a dummy forward
    with torch.no_grad():
        x = torch.zeros(1, *net.input_shape, device=lay.weight.device, dtype=lay.weight.dtype)
        for i in range(K):
            x = net.act(net.layers[i](x))
            if net.pools[i] > 0:
                x = F.avg_pool2d(x, net.pools[i])
    _, C, H, W = x.shape
    assert C == cin and pad == 0, "prototype handles unpadded conv frontiers"
    Ho, Wo = (H - kh) // s + 1, (W - kw) // s + 1
    if Ho * Wo == 1:
        return None                                            # single position = dense
    blocks = []
    for i in range(Ho):
        for j in range(Wo):
            idx = [c * H * W + (i * s + a) * W + (j * s + b) for c in range(cin) for a in range(kh) for b in range(kw)]
            blocks.append(torch.tensor(idx, device=lay.weight.device))
    return blocks


def mapback(net, K, x_star, n, blocks=None):
    """solve n ~ J_phi(x*)^T w ; returns unit [w|b] row and the relative residual. With `blocks` (conv
    weight sharing) w is confined to one receptive field: every position is tried, best residual wins,
    and the returned row is the FILTER (kernel + bias), comparable across positions."""
    def phi(x):
        return prefix_feats(net, x[None], K)[0][0]
    J = torch.autograd.functional.jacobian(phi, x_star)   # (D_phi, 784)
    with torch.no_grad():
        ph = phi(x_star)
    if blocks is None:
        w = torch.linalg.pinv(J.T, rtol=1e-10) @ n             # (CUDA lstsq only supports gels; pinv is fine at 784x400)
        rel = float((J.T @ w - n).norm() / n.norm())
        b = -float(ph @ w)
    else:
        best = None
        for idx in blocks:
            A = J[idx].T                                       # (784, |RF|)
            wb = torch.linalg.pinv(A, rtol=1e-10) @ n
            r = float((A @ wb - n).norm() / n.norm())
            if best is None or r < best[0]:
                best = (r, wb, idx)
        rel, w, idx = best
        b = -float(ph[idx] @ w)
    row = torch.cat([w, torch.tensor([b], device=w.device, dtype=w.dtype)])
    return row / row.norm(), rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", type=int, default=2, help="layer index to recover (prefix = layers < frontier)")
    ap.add_argument("--lines", type=int, default=20)
    ap.add_argument("--N", type=int, default=4001, help="scan points per line")
    ap.add_argument("--T", type=float, default=3.0, help="half-length of the scan segment (x0 ~ N(0,0.5^2) + t d, |d|=1 scaled by sqrt(784))")
    ap.add_argument("--max-kinks", type=int, default=400, help="per line, cap on unknown kinks processed")
    ap.add_argument("--x0-std", type=float, default=0.5, help="std of the Gaussian base point x0 (and |d| scale)")
    ap.add_argument("--snapshot", default="", help="REAL-RUN mode: a _pop_ snapshot; prefix + known frontier channels come from its fp64 partial record (student gauge), the black box stays the teacher; the recovered row is scored after cnn canonicalisation+alignment")
    ap.add_argument("--known-channels", default="", help="frontier channels already solved, e.g. 'all-but:6' or '0,1,2' (their kinks are excluded)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/diag/guessfree_layer.json")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(a.seed)
    teacher = make_teacher_cnn((1, 28, 28), CFGS, FC, 10, epochs=25, seed=0, device=dev, act="leaky_relu").double()
    for p in teacher.parameters():
        p.requires_grad_(False)
    K = a.frontier
    Wt = teacher.layers[K].weight.reshape(teacher.layers[K].weight.shape[0], -1)
    rows_t = torch.cat([Wt, teacher.layers[K].bias[:, None]], 1)
    rows_t = rows_t / rows_t.norm(dim=1, keepdim=True)
    C = rows_t.shape[0]
    known = []
    net = teacher
    snap = None
    if a.snapshot:
        from nets import ConvNet
        snap = torch.load(a.snapshot, map_location=dev, weights_only=False)
        net = ConvNet(snap["input_shape"], snap["conv_cfgs"], snap["fc_dims"], snap["out_dim"], snap["act"]).to(dev).double()
        net.load_state_dict({k: v.to(dev).double() for k, v in snap["partial_state"].items()})
        for p_ in net.parameters():
            p_.requires_grad_(False)
        pm = snap["partial_mask"][K].to(dev)
        known = pm.nonzero().flatten().tolist()
        print(f"snapshot {a.snapshot}: iter {snap['iter']}, prefix layers {[int(snap['partial_mask'][l].sum()) for l in range(K)]} solved rows, "
              f"frontier solved {len(known)}/{C}, unsolved {(~pm).nonzero().flatten().tolist()}")
    if a.known_channels.startswith("all-but:"):
        excl = {int(v) for v in a.known_channels[8:].split(",")}
        known = [c for c in range(C) if c not in excl]
    elif a.known_channels:
        known = [int(v) for v in a.known_channels.split(",")]
    blocks = rf_blocks(net, K)
    print(f"frontier layer {K + 1}: {C} channels, row dim {rows_t.shape[1] - 1}; prefix exact (layers 1..{K}); "
          f"known channels {len(known)}; conv positions {len(blocks) if blocks else 1}")
    found = []                 # list of (unit row, rel residual, jump ratio)
    t0 = time.time()
    for li in range(a.lines):
        x0 = torch.randn(784, device=dev, dtype=torch.double) * a.x0_std
        d = torch.randn(784, device=dev, dtype=torch.double); d = d / d.norm() * math.sqrt(784) * a.x0_std   # |d| ~ RMS of x0
        t_unk, ncand, dt_grid = scan_line(teacher, net, K, x0, d, a.T, a.N, known)
        t_unk = t_unk[:a.max_kinks]
        n_ok = 0
        for t in t_unk:
            ts = bisect_kink(teacher, x0, d, t - dt_grid, t + dt_grid)
            x_star = x0 + ts * d
            n, ratio, s0 = jump_normal(teacher, x0, d, ts, 1e-5, 1e-6)
            if ratio < 1e3:                                  # not a clean rank-1 jump (two kinks too close / numerical)
                continue
            row, rel = mapback(net, K, x_star, n, blocks)
            if rel > 1e-3:                                   # normal not in the prefix Jacobian's row space -> deeper-layer kink
                continue
            found.append((row, rel, ratio)); n_ok += 1
        # progress: how many distinct teacher channels have a found row within 1e-3?
        if found and snap is None:
            R = torch.stack([f[0] for f in found])
            cos = (R @ rows_t.T).abs()
            best = cos.max(0).values
            print(f"  line {li + 1}: {ncand} kinks on the line, {len(t_unk)} unexplained, {n_ok} mapped back | channels with a row at cos>0.999: "
                  f"{int((best > 0.999).sum())}/{C}, >0.99: {int((best > 0.99).sum())} | queries {QUERIES[0]:,} | {time.time() - t0:.0f}s", flush=True)
    # cluster the found rows (sign-invariant) and score clusters against the teacher
    R = torch.stack([f[0] for f in found])
    cos = (R @ R.T).abs()
    assigned = -torch.ones(len(R), dtype=torch.long)
    clusters = []
    for i in range(len(R)):
        if assigned[i] >= 0:
            continue
        members = ((cos[i] > 0.999) & (assigned < 0).to(cos.device)).nonzero().flatten()
        assigned[members.cpu()] = len(clusters)
        clusters.append(members)
    sizes = sorted([len(m) for m in clusters], reverse=True)
    print(f"{len(R)} rows -> {len(clusters)} clusters; sizes (top 20) {sizes[:20]}; singletons {sum(1 for s in sizes if s == 1)}")
    good = 0; errs = []
    hits = torch.zeros(C, dtype=torch.bool)
    for m in clusters:
        if len(m) < 2:
            continue
        r = R[m].mean(0); r = r / r.norm()
        c = (rows_t @ r).abs(); j = int(c.argmax())
        e = float((rows_t[j] - torch.sign(rows_t[j] @ r) * r).abs().max())
        if c[j] > 0.999:
            good += 1; hits[j] = True; errs.append(e)
    print(f"recurring clusters matching a teacher channel at cos>0.999: {good}; distinct channels hit {int(hits.sum())}/{C}; "
          f"cluster-mean row error median {float(torch.tensor(errs).median()) if errs else float('nan'):.2e} | total queries {QUERIES[0]:,}")
    if snap is not None and clusters:
        # inject the biggest cluster's filter into the unsolved channel and score after canonicalisation + alignment
        from align import cnn_canonicalize_, cnn_align_to_
        big = max(clusters, key=len)
        r = R[big].mean(0); r = r / r.norm()
        uns = (~snap["partial_mask"][K].to(dev)).nonzero().flatten().tolist()
        lay = net.layers[K]
        for u in uns:
            with torch.no_grad():
                lay.weight[u] = r[:-1].view_as(lay.weight[u]); lay.bias[u] = r[-1]
        t_ = teacher.clone(); r_ = net.clone()
        cnn_canonicalize_(t_); cnn_canonicalize_(r_); cnn_align_to_(r_, t_)
        A = torch.cat([t_.layers[K].weight.reshape(C, -1), t_.layers[K].bias[:, None]], 1)
        B = torch.cat([r_.layers[K].weight.reshape(C, -1), r_.layers[K].bias[:, None]], 1)
        e = (A - B).abs().max(1).values
        print(f"REAL-RUN check: injected cluster (size {len(big)}) into unsolved channel(s) {uns}; after canonicalisation+alignment the frontier's "
              f"per-channel errors: median {float(e.median()):.1e}, max {float(e.max()):.1e}, #<1e-3: {int((e < 1e-3).sum())}/{C}, "
              f"worst channel {int(e.argmax())}")
    json.dump(dict(frontier=K, lines=a.lines, queries=QUERIES[0], n_rows=len(R), cluster_sizes=sizes,
                   channels_hit=int(hits.sum()), hits=hits.tolist(), errs=errs), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
