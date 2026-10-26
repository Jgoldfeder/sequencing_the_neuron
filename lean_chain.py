"""lean_chain.py -- exact recovery of a deep (leaky) ReLU MLP from a committee guess with the lean kink refiner.

    python lean_chain.py <pop snapshot .pt> [--member best|IDX] [--h 1e-5] [--eps-side 1e-4] [--tries 6] [--out out.pt]

Layer by layer: refine every unit of layer l with lean_kink (prefix = the already refined layers), install the refined
row at the guess's magnitude (keeps the next layer's guess in its own gauge), then the next layer.  Units the refiner
rejects keep their guess and are retried once with fresh base points.  Output layer: closed-form least squares on
fresh Gaussian queries.  Black box = fp64 teacher forward only; the teacher's parameters are read only for scoring.
"""
import argparse, time, torch
from nets import MLP
from method import param_errors
import lean_kink as lk

ap = argparse.ArgumentParser()
ap.add_argument("snapshot"); ap.add_argument("--member", default="best")
ap.add_argument("--h", type=float, default=1e-5); ap.add_argument("--eps-side", type=float, default=1e-4)
ap.add_argument("--tries", type=int, default=6); ap.add_argument("--head-points", type=int, default=20000)
ap.add_argument("--out", default=""); ap.add_argument("--device", default="cuda")
ap.add_argument("--vote", action="store_true", help="replace committee-outvoted guess rows BEFORE refining (hurt L3: "
                "45 track failures vs 0); without it other members' rows are only fallback guesses for failed units")
ap.add_argument("--polish-l1", action="store_true", help="kink-point polish of layer 1 after the lean refine (~1e-14, expensive)")
a = ap.parse_args(); dev = a.device
d = torch.load(a.snapshot, map_location=dev, weights_only=False)
dims, act = d["dims"], d["act"]
t64 = MLP(dims, act=act).to(dev); t64.load_state_dict(d["teacher_state"]); t64 = t64.double()
for p in t64.parameters():
    p.requires_grad_(False)
Q = {"n": 0}
class _CountedTeacher(torch.nn.Module):                     # the black box as a module (kink_solve calls it like a net)
    def __init__(s, m): super().__init__(); s.m = m; s.dims = m.dims if hasattr(m, "dims") else None; s.layers = m.layers; s.act = m.act
    def forward(s, X):
        Q["n"] += X.shape[0]; return s.m(X)
def bb(X):
    Q["n"] += X.shape[0]
    with torch.no_grad():
        return t64(X)
X = torch.randn(4096, dims[0], device=dev, dtype=torch.float64) * 0.5
mems = []
for s in d["pop_states"]:
    m = MLP(dims, act=act).to(dev); m.load_state_dict(s); mems.append(m.double())
if a.member == "best":
    errs = [float((m(X) - bb(X)).abs().mean()) for m in mems]; bi = min(range(len(mems)), key=lambda i: errs[i])
else:
    bi = int(a.member)
net = mems[bi]
for p in net.parameters():
    p.requires_grad_(False)
# committee row consensus (teacher-free): the best-output member can hold a wrong copy of a unit that most other
# members got right (9-layer run: L4 unit learned by members 0/1/2/4 at cos ~1, member 6 = the guess at 0.17).
# Normalise the guess's gauge, align every member to it, and replace each row the committee outvotes.
from align import align_clone_to, scale_normalize_
# The vote runs on a scale-normalised COPY: the refiners' step sizes and acceptance tolerances are tuned to the
# guess's own gauge (normalising the guess itself broke the layer-1 bias: 7.7e-14 -> 8.5e-5).
with torch.no_grad():
    nz = MLP(dims, act=act).to(dev).double(); nz.load_state_dict(net.state_dict())
    scale_normalize_(nz)
    col_s, s_prev = [], torch.ones(dims[0], device=dev, dtype=torch.float64)
    for l in range(len(dims) - 2):                      # col_s[l] = column scales nz applies to layer l's inputs
        col_s.append(s_prev); s_prev = (net.layers[l].weight.double() * s_prev[None]).norm(dim=1)
    alt = [align_clone_to(m, nz)[1] for k, m in enumerate(mems) if k != bi]
def _unit_rows(n, l):
    R = torch.cat([n.layers[l].weight, n.layers[l].bias[:, None]], 1).double()
    return R / R.norm(dim=1, keepdim=True)
def _candidates(l, j, cos=0.99):
    """distinct candidate rows for unit j of layer l, most committee support first: [(support, row)]"""
    rows = [_unit_rows(nz, l)[j]] + [_unit_rows(m, l)[j] for m in alt]
    Rm = torch.stack(rows); sup = ((Rm @ Rm.T) > cos).sum(1)
    out, seen = [], []
    for i in sorted(range(len(rows)), key=lambda i: -int(sup[i])):
        if all(float(rows[i] @ r_) < cos for r_ in seen):
            seen.append(rows[i]); out.append((int(sup[i]), i))
    return out, Rm
def _install_guess(l, j, src):
    """put member src's row for unit j (0 = the guess itself) into net at the guess's magnitude"""
    if src == 0:
        return
    m = alt[src - 1]; W = net.layers[l].weight; B = net.layers[l].bias
    w_ = m.layers[l].weight[j].double() / col_s[l]; b_ = m.layers[l].bias[j].double()   # nz gauge -> guess gauge
    mag = float(torch.cat([W[j], B[j:j + 1]]).norm()) / float(torch.cat([w_, b_[None]]).norm())
    W[j] = (mag * w_).to(W.dtype); B[j] = mag * b_
def _install_row(l, j, row):
    """install a row given in nz gauge ([w|b], any scale) as unit j's guess, at the guess's magnitude"""
    W = net.layers[l].weight; B = net.layers[l].bias
    w_ = row[:-1] / col_s[l]; b_ = row[-1]
    mag = float(torch.cat([W[j], B[j:j + 1]]).norm()) / float(torch.cat([w_, b_[None]]).norm())
    W[j] = (mag * w_).to(W.dtype); B[j] = mag * b_
def _orphans(l, Wk, bk, mk, cos=0.99):
    """teacher-free candidates for layer l's unsolved units: every member's rows (nz gauge) that match NONE of the
    refined rows, clustered across members; [(support, row)] most-supported first.  (Member alignment to a guess
    that holds a garbage row puts the members' good copy of that unit in an arbitrary slot, so no slot lookup.)"""
    if Wk is not None and bool(mk.any()):
        Rr = torch.cat([Wk[mk].double() * col_s[l][None], bk[mk].double()[:, None]], 1)
        Rr = Rr / Rr.norm(dim=1, keepdim=True)
    else:
        Rr = torch.zeros(0, dims[l] + 1, device=dev, dtype=torch.float64)
    pool = []
    for m in [nz] + alt:
        R = _unit_rows(m, l)
        far = (R @ Rr.T).abs().amax(1) < cos if len(Rr) else torch.ones(len(R), dtype=torch.bool, device=dev)
        pool += [(r_, id(m)) for r_ in R[far]]
    out, used = [], [False] * len(pool)
    for i, (ri, mi) in enumerate(pool):
        if used[i]:
            continue
        grp = [k for k, (rk, mk_) in enumerate(pool) if not used[k] and float(rk @ ri) > cos]
        for k in grp:
            used[k] = True
        sup = len({pool[k][1] for k in grp})
        out.append((sup, torch.stack([pool[k][0] for k in grp]).median(0).values))
    return sorted(out, key=lambda t: -t[0])
with torch.no_grad():
    for l in (range(len(dims) - 2) if a.vote else ()):
        sw = 0
        for j in range(net.layers[l].weight.shape[0]):
            cands, Rm = _candidates(l, j)
            sup0 = int(((Rm @ Rm[0]) > 0.99).sum())          # committee support of the guess's own row
            if cands[0][0] > sup0:
                _install_guess(l, j, cands[0][1]); sw += 1
        print(f"[lean-chain] committee row vote: L{l + 1} {sw} guess rows replaced", flush=True)
q0 = Q["n"]; t0 = time.time()
oracle_net = _CountedTeacher(t64)
print(f"[lean-chain] guess = member {bi} of {len(mems)} | dims {dims}", flush=True)
e = param_errors(net, t64); print(f"[lean-chain] start: max_eps {e['max_eps']:.3e}", flush=True)
Lh = len(dims) - 2
for l in range(Lh):
    tl = time.time()
    oracle = lk.Oracle(bb)
    W = net.layers[l].weight; B = net.layers[l].bias
    n = W.shape[0]; ok = torch.zeros(n, dtype=torch.bool, device=dev)
    if l >= 1:
        # deep layers: forward-only kink solver (never inverts the prefix: its Jacobian is ill-conditioned, cond 1e6-1e19)
        import kink_solve
        _g = torch.Generator(device=dev).manual_seed(1234 + l)
        Wk, bk, mk, nqk = kink_solve.recover_layer(oracle_net, net, l, dev, gen=_g, angle_gate=12.0, sampling="track")
        oracle.n += nqk
        # retries for the units track left unsolved: fresh-seed track, then the design sampler (one unsolved unit
        # here poisons every layer above it)
        for samp, sd in (("track", 5678 + l), ("design", 9012 + l), ("design", 3456 + l)):
            left = [j for j in range(n) if mk is None or not bool(mk[j])]
            if not left:
                break
            kw = dict(need=max(400, 2 * W.shape[1]), pool=max(768, 2 * W.shape[1])) if samp == "design" else {}
            W2, b2, m2, nq2 = kink_solve.recover_layer(oracle_net, net, l, dev, only_channels=left,
                                                       gen=torch.Generator(device=dev).manual_seed(sd),
                                                       angle_gate=12.0, sampling=samp, **kw)
            oracle.n += nq2
            if W2 is not None:
                if mk is None:
                    Wk, bk, mk = W2, b2, m2
                else:
                    for j in left:
                        if bool(m2[j]):
                            Wk[j] = W2[j]; bk[j] = b2[j]; mk[j] = True
            print(f"[lean-chain]   L{l + 1} retry ({samp}): {sum(1 for j in left if W2 is not None and bool(m2[j]))}/{len(left)} "
                  f"recovered, {nq2:,} queries", flush=True)
        left = [j for j in range(n) if mk is None or not bool(mk[j])]
        orph = _orphans(l, Wk, bk, mk) if left else []
        if left:
            print(f"[lean-chain]   L{l + 1}: {len(left)} unsolved, committee orphan rows (support): "
                  f"{[sup for sup, _ in orph[:8]]}", flush=True)
        for j in left:
            for sup, row in orph:
                if sup < 2:
                    break
                g0 = (net.layers[l].weight[j].clone(), net.layers[l].bias[j].clone())
                with torch.no_grad():
                    _install_row(l, j, row)
                W2, b2, m2, nq2 = kink_solve.recover_layer(oracle_net, net, l, dev, only_channels=[j],
                                                           gen=torch.Generator(device=dev).manual_seed(7 + j),
                                                           angle_gate=12.0, sampling="track")
                oracle.n += nq2
                print(f"[lean-chain]   L{l + 1} unit {j}: orphan guess (support {sup}) -> "
                      f"{'OK' if W2 is not None and bool(m2[j]) else 'fail'}, {nq2:,} queries", flush=True)
                if W2 is not None and bool(m2[j]):
                    # the orphan guess is only ~1e-2 and rarely-firing units sit on badly spread kink surfaces:
                    # re-track from the refined row until it stops moving (one bad unit poisons every layer above)
                    v = torch.cat([W2[j], b2[j:j + 1]]).double()
                    for it in range(6):
                        with torch.no_grad():
                            _install_row(l, j, torch.cat([v[:-1] * col_s[l], v[-1:]]))
                        W3, b3, m3, nq3 = kink_solve.recover_layer(oracle_net, net, l, dev, only_channels=[j],
                                                                   gen=torch.Generator(device=dev).manual_seed(99 + it),
                                                                   angle_gate=12.0, sampling="design", verbose=False,
                                                                   need=max(400, 2 * W.shape[1]), pool=max(768, 2 * W.shape[1]))
                        oracle.n += nq3
                        if W3 is None or not bool(m3[j]):
                            break
                        v3 = torch.cat([W3[j], b3[j:j + 1]]).double(); v3 = v3 * torch.sign(v3 @ v)
                        dv = float((v3 / v3.norm() - v / v.norm()).norm()); v = v3
                        print(f"[lean-chain]     re-solve (design) {it + 1}: row moved {dv:.2e}, {nq3:,} queries", flush=True)
                        if dv < 1e-9:
                            break
                    if Wk is None:
                        Wk, bk, mk = W2, b2, m2
                    Wk[j] = v[:-1].to(Wk.dtype); bk[j] = v[-1].to(bk.dtype); mk[j] = True
                    break
                with torch.no_grad():
                    net.layers[l].weight[j] = g0[0]; net.layers[l].bias[j] = g0[1]
        if Wk is not None:
            for j in mk.nonzero().flatten().tolist():
                r = Wk[j].double(); rb = float(bk[j])
                mag = float(torch.cat([W[j], B[j:j + 1]]).norm()) / float(torch.cat([r, torch.tensor([rb], device=dev, dtype=r.dtype)]).norm())
                if float(r @ W[j]) < 0:
                    r, rb = -r, -rb
                W[j] = (mag * r).to(W.dtype); B[j] = mag * rb; ok[j] = True
    for rnd, seed in enumerate((l, 1000 + l, 2000 + l, 3000 + l) if l < 1 else ()):
        gen = torch.Generator(device=dev).manual_seed(seed)
        for j in [j for j in range(n) if not ok[j]]:
            out = lk.refine_unit(oracle, net, l, W[j].double(), B[j].double(), gen, h=a.h, eps_side=a.eps_side, tries=a.tries)
            if out is not None:
                r, rb, _ = out
                mag = float(torch.cat([W[j], B[j:j + 1]]).norm()) / float(torch.cat([r, torch.tensor([rb], device=dev, dtype=r.dtype)]).norm())
                W[j] = (mag * r).to(W.dtype); B[j] = mag * rb; ok[j] = True
    if l == 0 and a.polish_l1:
        # kink-point polish of the lean rows (kink_solve.polish_layer): 1e-10 -> ~1e-14
        import copy as _cp, kink_solve
        from verify_layer1 import _Oracle
        _orc = _Oracle(_cp.deepcopy(t64).eval()); _q = _orc.n
        _out = kink_solve.polish_layer(_orc, net, 0, ok.nonzero().flatten().tolist(),
                                       torch.Generator(device=dev).manual_seed(77), max_rows=20000)
        for j, (wp, bp, _H) in _out.items():
            r = wp.double(); rb = float(bp)
            if float(r @ W[j]) < 0:
                r, rb = -r, -rb
            mag = float(torch.cat([W[j], B[j:j + 1]]).norm()) / float(torch.cat([r, torch.tensor([rb], device=dev, dtype=r.dtype)]).norm())
            W[j] = (mag * r).to(W.dtype); B[j] = mag * rb
        Q["n"] += _orc.n - _q; oracle.n += _orc.n - _q
        print(f"[lean-chain] L1 polish: {len(_out)}/{int(ok.sum())} rows polished, {_orc.n - _q:,} queries", flush=True)
    ep = param_errors(net, t64)["max_eps_per_matrix"]
    print(f"[lean-chain] L{l + 1}: {int(ok.sum())}/{n} refined | {oracle.n:,} queries ({oracle.n / n:.0f}/unit) | "
          f"{time.time() - tl:.0f}s | L{l + 1} max eps W {ep[2 * l]:.2e} b {ep[2 * l + 1]:.2e}", flush=True)
with torch.no_grad():
    Xh = torch.randn(a.head_points, dims[0], device=dev, dtype=torch.float64)
    Y = bb(Xh); H = Xh
    for lay in net.layers[:-1]:
        H = net.act(lay(H))
    A = torch.cat([H, torch.ones(len(H), 1, device=dev, dtype=H.dtype)], 1)
    sol = torch.linalg.lstsq(A, Y).solution
    net.layers[-1].weight.copy_(sol[:-1].T); net.layers[-1].bias.copy_(sol[-1])
e = param_errors(net, t64)
print(f"[lean-chain] head: closed-form LSQ on {a.head_points:,} queries", flush=True)
print(f"[lean-chain] FINAL max_eps {e['max_eps']:.3e} mean {sum(e['mean_eps_per_matrix']) / len(e['mean_eps_per_matrix']):.3e} | "
      f"per-matrix max {' '.join(f'{v:.1e}' for v in e['max_eps_per_matrix'])}", flush=True)
print(f"[lean-chain] refiner + head queries {Q['n'] - q0:,} | {time.time() - t0:.0f}s", flush=True)
if a.out:
    torch.save({"dims": dims, "act": act, "state_dict": net.state_dict(), "queries": Q["n"] - q0}, a.out)
    print("[lean-chain] saved", a.out, flush=True)
