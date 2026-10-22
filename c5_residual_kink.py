"""Residual-Jacobian directed kink search for the LAST missing rows of a frontier layer (black-box, 2026-10-19).

Setting: prefix (conv1, conv2) exact, most frontier rows (C5) already exact (committee + certificate); the leftovers are
channels that no random/mined query ever drives positive ("silent"), so the committee never learns them and random
lines never cross their kinks.  Leaky ReLU keeps them in the output Jacobian at slope 0.01, so:

  1. J_h(x) = d y / d h  (10 x 400) by central differences along the 400 prefix-mapped directions  Dx = pinv(d h/d x)
     (800 queries per point), at K random points;
  2. project every Jacobian row off the span of the KNOWN rows: the residual is spanned by the unknown rows' complement
     parts; its numerical rank = number of missing rows (completeness certificate), its top right-singular vectors are
     the search directions;
  3. straight lines in x along Dx v for each direction v, from several starts; the black box is scanned along the line
     (second differences of a piecewise-linear function), kinks are bisected to ~1e-9, kinks explained by KNOWN units
     (conv1/conv2 positions, known C5 rows) are discarded attacker-side;
  4. at each remaining kink the Jacobian jump  dJ_h = 0.99 u w^T  gives the row direction w (top right singular vector)
     and the bias from the kink point (w.h* + b = 0); rows are deduped and verified with the hyperplane test
     (random points on {w.h+b=0}, realised by inversion through the prefix, must all be kinks of the black box);
  5. accepted rows join the known set and the residual is recomputed; stop when its rank is 0.

Scale of a row is gauge (positive homogeneity); rows are reported unit-norm with b/|w|.  Oracle (teacher weights) is
used ONLY for scoring in the printout."""
import argparse, math, time, torch, torch.nn.functional as F
from data import make_teacher_cnn

p = argparse.ArgumentParser()
p.add_argument("--points", type=int, default=48, help="Jacobian sample points for the residual subspace")
p.add_argument("--starts", type=int, default=6, help="line starts per direction")
p.add_argument("--half", type=float, default=6.0, help="half-length of each x-line (in units of the mapped direction)")
p.add_argument("--scan", type=int, default=1501, help="scan points per line")
p.add_argument("--rounds", type=int, default=4)
p.add_argument("--known-frac", type=float, default=-1, help="<0: known = all channels with pos-frac>=1e-3 on gaussian x (the committee's set); else drop this fraction of rows at random")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--verbose", action="store_true")
a = p.parse_args()
dev = "cuda"; torch.manual_seed(a.seed); g = torch.Generator(device=dev).manual_seed(a.seed); D = torch.double
t = make_teacher_cnn((1, 28, 28), [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)], (84,), 10, epochs=25, seed=0, device=dev, act="leaky_relu", verbose=False).double()
for q in t.parameters(): q.requires_grad_(False)
NQ = [0]
def bb(x):                                   # the black box (fp64 forward), counts queries
    NQ[0] += x.shape[0]
    with torch.no_grad(): return t(x)
# ---- attacker side: the exact prefix (conv1, conv2) ----
def prefix(x):
    h = x.view(-1, 1, 28, 28)
    for i in range(2): h = F.avg_pool2d(t.act(t.layers[i](h)), t.pools[i])
    return h.reshape(h.shape[0], -1)
def prefix_preacts(x):                       # all conv1/conv2 pre-activations (for kink attribution)
    h = x.view(-1, 1, 28, 28); out = []
    for i in range(2):
        z = t.layers[i](h); out.append(z.reshape(z.shape[0], -1)); h = F.avg_pool2d(t.act(z), t.pools[i])
    return torch.cat(out, 1)
def prefix_jac(x0):                          # (400, 784); works under a caller's no_grad too
    with torch.enable_grad():
        return torch.autograd.functional.jacobian(lambda x: prefix(x[None])[0], x0.detach(), vectorize=True)
Wt = t.layers[2].weight.reshape(120, -1); bt = t.layers[2].bias     # ORACLE, scoring only
with torch.no_grad():
    Xg = torch.randn(20000, 784, device=dev, generator=g, dtype=D) * 0.5
    pos = ((prefix(Xg) @ Wt.T + bt) > 0).double().mean(0)
if a.known_frac < 0:
    known_idx = (pos >= 1e-3).nonzero().flatten()
else:
    perm = torch.randperm(120, generator=torch.Generator().manual_seed(a.seed)); known_idx = perm[int(a.known_frac * 120):].to(dev).sort().values
unknown_idx = torch.tensor([i for i in range(120) if i not in set(known_idx.tolist())], device=dev)
Wk = Wt[known_idx].clone(); bk = bt[known_idx].clone()           # "known" rows as the committee would hold them (exact)
print(f"known rows {len(known_idx)}, unknown {len(unknown_idx)}: {unknown_idx.tolist()}  (unknown pos-frac on gaussian x: {[f'{v:.0e}' for v in pos[unknown_idx].tolist()]})", flush=True)
EPS = 1e-6
def safe_steps(x0, Dx, eps0=EPS, halvings=14):
    """attacker-side: per-direction step eps_i such that x0 +- eps_i Dx e_i crosses NO conv1/conv2 kink (pinv columns have
    norms up to ~1e3, so a fixed eps crosses prefix kinks and ruins the Jacobian).  Returns eps (400,) and #unsafe."""
    with torch.no_grad():
        z0 = prefix_preacts(x0[None])[0] > 0; eps = torch.full((400,), eps0, device=dev, dtype=D); bad = torch.ones(400, dtype=torch.bool, device=dev)
        for _ in range(halvings):
            E = eps[:, None] * Dx.T
            zp = prefix_preacts(x0[None] + E) > 0; zm = prefix_preacts(x0[None] - E) > 0
            bad = ((zp != z0[None]).any(1) | (zm != z0[None]).any(1))
            if not bad.any(): break
            eps = torch.where(bad, eps / 2, eps)
    return eps, int(bad.sum())
def jac_h(x0, Dx, eps=None):                  # black-box Jacobian wrt h at x0 (10, 400): 800 queries
    if eps is None: eps, _ = safe_steps(x0, Dx)
    E = eps[:, None] * Dx.T                   # (400, 784)
    return ((bb(x0[None] + E) - bb(x0[None] - E)) / (2 * eps[:, None])).T
def unit(v): return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-300)

def invert_to(ht, x_init, steps=60, lr=0.02):
    """attacker-side: x with prefix(x) ~ ht, warm-started from x_init (no queries)"""
    x = x_init.detach().clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        loss = ((prefix(x[None])[0] - ht) ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): x.clamp_(-3, 3)
    return x.detach()
def certify(w, b, hk, tries=6, verbose=False):
    """hyperplane certificate: random points on {w.h+b=0} must be kinks of the black box.  A point is placed by inverting
    towards hk's neighbourhood, then an EXACT attacker-side bisection of w.prefix(x)+b along the mapped normal dr; the
    kink test compares the second difference at the point with its neighbours (piecewise-linear black box)."""
    ok = 0; fails = []; placed = 0
    for _ in range(6 * tries):
        if placed >= tries: break
        if placed >= 2 and ok < placed: break                   # early exit: a wrong hyperplane fails immediately
        xr = (torch.randn(784, device=dev, generator=g, dtype=D) * 0.5).requires_grad_(True); opt = torch.optim.Adam([xr], lr=0.05)
        for _ in range(150):
            hr = prefix(xr[None])[0]; loss = (w @ hr + b) ** 2 + 1e-3 * ((hr - hk) ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
        xr = xr.detach(); dr = torch.linalg.pinv(prefix_jac(xr)) @ w; dr = dr / dr.norm()
        with torch.no_grad():
            f = lambda s: float(w @ prefix((xr + s * dr)[None])[0] + b)
            lo, hi = -0.5, 0.5
            for _ in range(8):                                   # bracket a sign change
                if f(lo) * f(hi) < 0: break
                lo *= 2; hi *= 2
            if f(lo) * f(hi) >= 0: fails.append("nobracket"); continue
            for _ in range(80):
                m = 0.5 * (lo + hi)
                if f(lo) * f(m) <= 0: hi = m
                else: lo = m
            sk = 0.5 * (lo + hi); xk = xr + sk * dr
            if abs(f(sk)) > 1e-9: fails.append(f"res{abs(f(sk)):.0e}"); continue
            e = 1e-5
            pts = xk[None] + torch.tensor([-2 * e, -e, 0.0, e, 2 * e], device=dev, dtype=D)[:, None] * dr[None]
            y = bb(pts); dd = (y[2:] - 2 * y[1:-1] + y[:-2]).abs().amax(1)
            passed = float(dd[1]) > 100 * max(float(dd[0]), float(dd[2]), 1e-13)
            ok += int(passed); placed += 1
            if not passed: fails.append(f"dd{float(dd[1]):.0e}/{max(float(dd[0]), float(dd[2])):.0e}")
    if verbose and fails: print(f"      certificate: placed {placed}/{tries}, failures: {fails}")
    return ok, placed

found = []                                    # accepted (w_unit, b_over_norm, oracle_match)
def score(w, b):                              # oracle: best cosine with an unknown teacher row (sign-aware) and bias error
    wn = unit(Wt); cos = (wn @ w); j = int(cos.argmax()); bn = bt[j] / Wt[j].norm()
    return j, float(cos[j]), float(abs(bn - b))
t0 = time.time()
with torch.no_grad(): hk0 = prefix(Xg[:1])[0]
for j in known_idx[:6].tolist():
    ok, tr = certify(unit(Wt[j]), bt[j] / Wt[j].norm(), hk0, verbose=True); print(f"[self-test] certificate on known row {j}: {ok}/{tr}", flush=True)
for _ in range(2):                                               # negative control: a random hyperplane must FAIL
    wr = unit(torch.randn(400, device=dev, generator=g, dtype=D)); ok, tr = certify(wr, -(wr @ hk0), hk0); print(f"[self-test] certificate on a RANDOM hyperplane: {ok}/{tr} (must be ~0)", flush=True)
for rnd in range(a.rounds):
    Wall = torch.cat([Wk] + [f[0][None] for f in found]) if found else Wk
    Q, _ = torch.linalg.qr(Wall.T); Pp = torch.eye(400, device=dev, dtype=D) - Q @ Q.T
    # ---- (1)-(2) residual Jacobian subspace ----
    rows = []; nbad = []
    X0 = torch.randn(2 * a.points, 784, device=dev, generator=g, dtype=D) * 0.5
    for k in range(2 * a.points):
        if len(rows) >= a.points: break
        P = prefix_jac(X0[k]); Dx = torch.linalg.pinv(P); eps, nb = safe_steps(X0[k], Dx)
        if nb > 0: nbad.append(nb); continue                     # a prefix unit sits too close to this point: skip it
        rows.append(jac_h(X0[k], Dx, eps) @ Pp)
    R = torch.cat(rows); S = torch.linalg.svdvals(R)
    r = int((S > 1e-4 * S[0]).sum()) if S[0] > 1e-9 else 0
    print(f"[round {rnd}] Jacobian points used {len(rows)} (skipped {len(nbad)} with unsafe directions)", flush=True)
    print(f"\n[round {rnd}] known {Wall.shape[0]}: residual singular values {[f'{v:.1e}' for v in S[:max(r + 3, 4)].tolist()]} -> rank {r} (missing rows)  | {NQ[0]:,} queries", flush=True)
    if r == 0: break
    _, _, Vh = torch.linalg.svd(R, full_matrices=False); dirs = Vh[:r]
    # ---- (3) directed line scans: the h-line h0 + s v is realised as a dense POLYLINE of warm-started inversions.  On each
    #      x-segment the black box is piecewise linear; the KNOWN units' flips (conv1/conv2/known C5, attacker-side, no
    #      queries) cut the segment into flip-free pieces, and a piece holds an UNKNOWN kink iff its midpoint leaves the chord ----
    new = 0; rejected = 0
    bias_all = torch.cat([bk] + [torch.tensor([f[1]], device=dev, dtype=D) for f in found])
    def known_pattern(Xq):
        with torch.no_grad():
            return torch.cat([prefix_preacts(Xq), prefix(Xq) @ Wall.T + bias_all], 1) > 0
    for di, v in enumerate(dirs):
        for s0 in range(a.starts):
            tline = time.time()
            x0 = torch.randn(784, device=dev, generator=g, dtype=D) * 0.5
            with torch.no_grad(): h0 = prefix(x0[None])[0]
            sgrid = torch.linspace(-a.half, a.half, a.scan, device=dev, dtype=D)
            xs = [None] * a.scan; mid = a.scan // 2; xs[mid] = x0
            for order in (range(mid + 1, a.scan), range(mid - 1, -1, -1)):      # walk outwards, warm-started
                prev = x0
                for i in order:
                    xi = invert_to(h0 + sgrid[i] * v, prev, steps=25); xs[i] = xi; prev = xi
            X = torch.stack(xs)
            with torch.no_grad():
                Hs = prefix(X); follow = float(((Hs - Hs[mid]) @ v / (sgrid - sgrid[mid]).clamp_min(1e-9))[mid + 1:].median())
            # known flips per segment, located attacker-side by bisection -> flip-free pieces [ta, tb] in segment coordinates
            pieces = []                                                           # (seg index, ta, tb)
            Z0 = known_pattern(X)                                                 # (M, U)
            flip_seg = (Z0[1:] != Z0[:-1]).any(1)
            for c in range(a.scan - 1):
                if not flip_seg[c]:
                    pieces.append((c, 0.0, 1.0)); continue
                xa, xb_ = X[c], X[c + 1]; cuts = []
                units = (Z0[c] != Z0[c + 1]).nonzero().flatten()
                for u_ in units.tolist():                                         # bisect each flipping unit's root on the segment (no queries)
                    lo, hi = 0.0, 1.0; za = bool(Z0[c, u_])
                    for _ in range(30):
                        m_ = 0.5 * (lo + hi); zm = bool(known_pattern((xa + m_ * (xb_ - xa))[None])[0, u_])
                        if zm == za: lo = m_
                        else: hi = m_
                    cuts.append(0.5 * (lo + hi))
                cuts = sorted(cuts); bounds = [0.0] + cuts + [1.0]; marg = 1e-7
                for ta, tb in zip(bounds[:-1], bounds[1:]):
                    if tb - ta > 4 * marg: pieces.append((c, ta + marg, tb - marg))
            # chord test on every flip-free piece (3 queries per piece: ends + midpoint)
            PA = torch.stack([X[c] + ta * (X[c + 1] - X[c]) for c, ta, tb in pieces]); PB = torch.stack([X[c] + tb * (X[c + 1] - X[c]) for c, ta, tb in pieces])
            ya = bb(PA); yb = bb(PB); ym = bb(0.5 * (PA + PB))
            dev_ = (ym - 0.5 * (ya + yb)).abs().amax(1); scale = ya.abs().amax() + 1
            cand = (dev_ > 1e-9 * scale).nonzero().flatten().tolist()
            for ci in cand:
                xa, xb_ = PA[ci], PB[ci]
                for _ in range(45):                                               # exact bisection: no known kink inside the piece
                    xm = 0.5 * (xa + xb_); q1 = 0.5 * (xa + xm); q2 = 0.5 * (xm + xb_)
                    y = bb(torch.stack([xa, q1, xm, q2, xb_]))
                    dl = (y[1] - 0.5 * (y[0] + y[2])).abs().max(); dr_ = (y[3] - 0.5 * (y[2] + y[4])).abs().max()
                    if dl > dr_: xb_ = xm
                    else: xa = xm
                xk = 0.5 * (xa + xb_); dseg = PB[ci] - PA[ci]; dseg = dseg / dseg.norm()
                Dxk = torch.linalg.pinv(prefix_jac(xk)); step = 1e-6 * dseg
                dJ = jac_h(xk + step, Dxk) - jac_h(xk - step, Dxk)
                Uj, Sj, Vj = torch.linalg.svd(dJ)
                if Sj[0] < 1e-6 or Sj[1] / Sj[0] > 1e-3: rejected += 1; continue
                w = Vj[0]
                with torch.no_grad(): hk = prefix(xk[None])[0]
                b = -(w @ hk)
                if (unit(Wall) @ w).abs().max() > 0.999: continue
                if found and (torch.stack([f[0] for f in found]) @ w).abs().max() > 0.999: continue
                ok, tries = certify(w, b, hk)
                j, cos, berr = score(w, b); acc = tries >= 4 and ok == tries
                if acc or a.verbose:
                    print(f"  dir {di} start {s0}: rank-1 ratio {float(Sj[1] / Sj[0]):.1e}  certificate {ok}/{tries} -> {'ACCEPT' if acc else 'reject'}"
                          f"   [oracle: teacher row {j} ({'unknown' if j in unknown_idx.tolist() else 'known'}) |cos| {abs(cos):.6f}  bias err {berr:.1e}]", flush=True)
                if acc:
                    found.append((w if cos >= 0 else -w, b if cos >= 0 else -b, j)); new += 1
                    bias_all = torch.cat([bk] + [torch.tensor([f[1]], device=dev, dtype=D) for f in found]); Wall = torch.cat([Wk] + [f[0][None] for f in found])
                else: rejected += 1
            print(f"    dir {di} start {s0}: {int(flip_seg.sum())}/{a.scan - 1} segments with known flips -> {len(pieces)} flip-free pieces, {len(cand)} unknown-kink candidates, {time.time() - tline:.0f}s | found {len(found)} | {NQ[0]:,} q", flush=True)
        print(f"  dir {di}: h-path follows v at rate {follow:.2f} (1 = exact); found so far {len(found)}, rejected {rejected} | {NQ[0]:,} q", flush=True)
    print(f"[round {rnd}] +{new} rows; found so far {len(found)}: teacher rows {sorted(set(f[2] for f in found))} | {NQ[0]:,} queries, {time.time() - t0:.0f}s", flush=True)
    if new == 0 and rnd > 0: break
got = sorted(set(f[2] for f in found)); miss = [i for i in unknown_idx.tolist() if i not in got]
print(f"\nDONE: recovered {len(got)}/{len(unknown_idx)} unknown rows {got}; missing {miss}; total {NQ[0]:,} queries, {time.time() - t0:.0f}s")
