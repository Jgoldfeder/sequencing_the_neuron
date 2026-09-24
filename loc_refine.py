"""Location-based layer refinement, GUIDED (no inversion, no oracle-side sweep).

Setting (the ONLY one this is called in): layers < frontier of `cons` are
essentially exact (forward-evaluable to ~1e-5 or better) and layer `frontier`
holds a guess within ~1e-2. Then everything about a target neuron's kink is
PREDICTABLE from the model, and oracle queries are spent only LOCATING it:

  * AIM (model-only): Newton-project seeds onto the guessed kink surface
    wg.phi(x)+bg=0; the true kink lies within |t| <= eps*||wb||*||h1||/nu of
    that point along the local normal (eps = the guess's relative error).
  * ISOLATE (model-only): reject any bracket a PREFIX neuron's kink enters
    (checked on the near-exact prefix) or a SIBLING's predicted kink comes
    within 3x of. No ~2d-query normal sweep per point (the pre-rewrite version
    burned ~2d queries/point on a refine_neuron sweep it then discarded).
  * LOCATE (oracle): 6 points per bracket, collinearity scan over the 4 point
    triples -> zoom onto the deviating interval until the kink sits in the
    central gap -> intersect the two affine side lines -> t* to ~1e-13.
  * FINGERPRINT (oracle, d-independent): the gradient jump across a kink is
    rank-1 along its normal; 3 perpendicular jump probes read the angle to the
    guess normal. Ours: <~1deg. An unrelated deeper-layer kink: ~60-90deg.
    Rejects wrong kinks per point (~16 queries).
  * SOLVE: h = phi(x*) over >= Din spanning kink points; LTS + trimmed SVD of
    w.h + b = 0, with a CROSS-VALIDATED error certificate per neuron.

The engine is batched ACROSS THE WHOLE LAYER: every neuron's seeds advance in
lockstep waves, so one wave = ONE oracle call for all in-flight brackets of
all neurons (plus one batched fingerprint call), and all state transitions are
tensor ops -- no per-seed Python, no per-seed GPU syncs.

Cost: ~10-25 queries per accepted point, ~1.3*Din points/neuron, independent
of input dim -> ~5-15k q/neuron; a 200-wide layer solves in seconds-to-a-
minute of GPU wall. Accuracy = prefix error x1 (forward only, no
amplification): exact prefix -> ~1e-13..1e-15; 1e-5 prefix -> ~1e-5 (true
frame; tighter in the recovered frame, which is what the peel chain uses).
"""
import torch
from verify_layer1 import _Oracle

_BASE6 = (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0)


def _phi(pre, X, frontier):
    a = X
    for i in range(frontier):
        a = pre.act(pre.layers[i](a))
    return a


def _project_rows(pre, X, Wg, bg, frontier, iters=12):
    """MODEL-ONLY batched Newton so wg_row.phi(x)+bg_row = 0 per row (each row
    may target a DIFFERENT neuron). Returns (X*, unit normals, nu, ok)."""
    xcur = X
    for _ in range(iters):
        with torch.enable_grad():
            xg = xcur.detach().clone().requires_grad_(True)
            g = (_phi(pre, xg, frontier) * Wg).sum(1) + bg           # (N,)
            u = torch.autograd.grad(g.sum(), xg)[0]                  # (N, d)
        nu = u.norm(dim=1).clamp_min(1e-30)
        step = (g.detach() / nu).unsqueeze(1) * (u / nu.unsqueeze(1))
        xcur = xg.detach() - torch.where((nu > 1e-14).unsqueeze(1), step,
                                         torch.zeros_like(step))
    with torch.enable_grad():
        xg = xcur.detach().clone().requires_grad_(True)
        g = (_phi(pre, xg, frontier) * Wg).sum(1) + bg
        u = torch.autograd.grad(g.sum(), xg)[0]
    nu = u.norm(dim=1)
    uh = u / nu.clamp_min(1e-30).unsqueeze(1)
    wbn = torch.cat([Wg, bg.unsqueeze(1)], 1).norm(dim=1).clamp_min(1e-30)
    ones = torch.ones(len(xcur), 1, dtype=X.dtype, device=X.device)
    hn = torch.cat([_phi(pre, xcur, frontier), ones], 1).norm(dim=1)
    ok = (nu > 1e-12) & (g.detach().abs() < 1e-9 * wbn * hn)
    return xcur.detach(), uh.detach(), nu.detach(), ok


@torch.no_grad()
def _iso_rows(pre, X0, Uh, ch, delta, frontier, Wl, bl, chunk=16384):
    """MODEL-ONLY batched isolation: for rows (x0, uh, channel, delta), True
    iff the bracket x0 +- delta*uh is clear of (a) prefix-neuron kinks
    (sign-constant with margin over 7 samples) and (b) sibling kinks predicted
    from their guesses (crossing distance > 3*delta). The sibling gate is
    purely geometric -- an absolute guess-error slack rejects half of all
    seeds (some sibling of 200 always sits below it); anything that sneaks in
    is caught by the dirty-bracket scan, the fingerprint, and the solve trim."""
    dev, dt = X0.device, X0.dtype
    offs = torch.linspace(-1.0, 1.0, 7, dtype=dt, device=dev)
    out = torch.zeros(len(X0), dtype=torch.bool, device=dev)
    for s0 in range(0, len(X0), chunk):
        s1 = min(s0 + chunk, len(X0))
        B = s1 - s0
        D = delta[s0:s1]
        P = (X0[s0:s1, None, :]
             + (offs[None, :, None] * D[:, None, None]) * Uh[s0:s1, None, :])
        a = P.reshape(B * 7, -1)
        ok = torch.ones(B, dtype=torch.bool, device=dev)
        for i in range(frontier):
            z = pre.layers[i](a)
            zv = z.reshape(B, 7, -1)
            sgn = (zv > 0).all(1) | (zv < 0).all(1)                # (B, n)
            margin = zv.abs().amin(1) > (zv[:, 1:] - zv[:, :-1]).abs().amax(1)
            ok &= (sgn & margin).all(-1)
            a = pre.act(z)
        H = a.reshape(B, 7, -1)
        zs = H @ Wl.t() + bl                                       # (B, 7, C)
        slope = (zs[:, 6] - zs[:, 0]) / (2.0 * D[:, None]).clamp_min(1e-300)
        tj = zs[:, 3].abs() / slope.abs().clamp_min(1e-30)
        tj.scatter_(1, ch[s0:s1].unsqueeze(1), float("inf"))
        ok &= (tj > 3.0 * D[:, None]).all(-1)
        out[s0:s1] = ok
    return out


def _side_fit(U, Y):
    """Batched 3-point affine fit y ~ a*u + c over all output dims.
    U: (3,), Y: (A, 3, O). Returns a (A, O), c (A, O), res (A,)."""
    um = U.mean()
    ym = Y.mean(1)
    a = ((U - um)[None, :, None] * (Y - ym[:, None, :])).sum(1) / ((U - um) ** 2).sum()
    c0 = ym - a * um
    res = (Y - (U[None, :, None] * a[:, None, :] + c0[:, None, :])).abs().amax((1, 2))
    return a, c0, res


@torch.no_grad()
def _collect_layer(orc, pre, frontier, Wl, bl, channels, gen, want, xcap,
                   pool_mult, eps_ladder, q_budget, match_deg=20.0,
                   fly_cap=96, max_waves=200, orc_chunk=65536):
    """Collect up to `want` clean kink points for EVERY channel in `channels`
    simultaneously. Returns {c: (list_of_h, list_of_obs)}. One wave = one
    oracle call over all in-flight brackets + one batched fingerprint call;
    all per-seed logic is tensorized (see module docstring for the per-seed
    semantics, which are unchanged from the serial version)."""
    dev, dt = Wl.device, Wl.dtype
    d = pre.layers[0].weight.shape[1]
    C = len(channels)
    R = len(eps_ladder)
    lad = torch.tensor(eps_ladder, dtype=dt, device=dev)
    base = torch.tensor(_BASE6, dtype=dt, device=dev)
    q0 = orc.n
    wbn = torch.cat([Wl, bl.unsqueeze(1)], 1).norm(dim=1).clamp_min(1e-30)

    # ---------------- per-channel bookkeeping (CPU), shared across rounds
    stats = {k: 0 for k in ("pool_reject", "launch_iso", "nonconsec",
                            "subnoise", "fit_dirty", "fit_outspan", "zoomcap",
                            "grow_overflow", "grow_iso", "fp_reject",
                            "accepted")}
    acc = {c: 0 for c in channels}          # accepted count
    fly = {c: 0 for c in channels}          # in-flight rows
    obs = {c: [] for c in channels}
    hsl = {c: [] for c in channels}
    start = {c: 0 for c in channels}
    P_X0 = P_Uh = P_Nu = P_Hn = P_D = P_ch = None
    queue = {}

    def _pool(needy):
        nonlocal P_X0, P_Uh, P_Nu, P_Hn, P_D, P_ch, queue
        # size each channel's pool to its SHORTFALL at the observed acceptance
        # (~30% in dense deeper fields), not a flat multiple of want
        ns = [max(64, int(pool_mult * (want - acc[c]) / 0.3)) for c in needy]
        P_ch = torch.cat([torch.full((n,), c, device=dev, dtype=torch.long)
                          for c, n in zip(needy, ns)])          # (N,)
        N = len(P_ch)
        sc = 0.3 + 3.5 * torch.rand(N, 1, generator=gen, device=dev, dtype=dt)
        seeds = torch.randn(N, d, generator=gen, device=dev, dtype=dt) * sc
        P_X0, P_Uh, P_Nu, okp = _project_rows(pre, seeds, Wl[P_ch], bl[P_ch],
                                              frontier)
        keep = okp & (P_X0.norm(dim=1) <= xcap)
        stats["pool_reject"] += int((~keep).sum())
        P_X0, P_Uh, P_Nu, P_ch = P_X0[keep], P_Uh[keep], P_Nu[keep], P_ch[keep]
        ones = torch.ones(len(P_X0), 1, dtype=dt, device=dev)
        P_Hn = torch.cat([_phi(pre, P_X0, frontier), ones], 1).norm(dim=1)
        P_D = (3.0 * lad[None, :]
               * (wbn[P_ch] * P_Hn / P_Nu.clamp_min(1e-30))[:, None])
        queue = {c: [] for c in needy}
        for i, c in enumerate(P_ch.tolist()):
            queue[c].append(i)
        for c in needy:
            fly[c] = 0

    def _launch():
        pids, rungs = [], []
        for c in queue:
            need = want - acc[c]
            if need <= 0:
                continue
            # stagger: until the rung floor is bootstrapped (9 obs), keep few
            # rows in flight -- a full launch would walk every ladder from 0
            cap = fly_cap if len(obs[c]) >= 9 else 16
            room = min(cap, need + 8) - fly[c]
            take = min(room, len(queue[c])) if room > 0 else 0
            for _ in range(take):
                pids.append(queue[c].pop())
                rungs.append(start[c])
                fly[c] += 1
        if not pids:
            return None
        pid = torch.tensor(pids, device=dev, dtype=torch.long)
        rung = torch.tensor(rungs, device=dev, dtype=torch.long)
        # isolation at the start rung; failures retry at the ladder floor
        ok = _iso_rows(pre, P_X0[pid], P_Uh[pid], P_ch[pid],
                       P_D[pid, rung], frontier, Wl, bl)
        low = ~ok & (rung > 0)
        if bool(low.any()):
            ok2 = _iso_rows(pre, P_X0[pid[low]], P_Uh[pid[low]], P_ch[pid[low]],
                            P_D[pid[low], torch.zeros(int(low.sum()),
                                                      device=dev,
                                                      dtype=torch.long)],
                            frontier, Wl, bl)
            rung[low] = 0
            ok[low] = ok2
        stats["launch_iso"] += int((~ok).sum())
        for c_ in P_ch[pid[~ok]].tolist():
            fly[c_] -= 1                                     # no capacity: drop
        pid, rung = pid[ok], rung[ok]
        return {"pid": pid, "rung": rung,
                "center": torch.zeros(len(pid), dtype=dt, device=dev),
                "half": P_D[pid, rung],
                "zoomed": torch.zeros(len(pid), dtype=torch.bool, device=dev),
                "zi": torch.zeros(len(pid), dtype=torch.long, device=dev)}

    def _cat(a, b):
        return {k: torch.cat([a[k], b[k]]) for k in a}

    for _round in range(6):
      needy = [c for c in channels if acc[c] < want]
      if not needy or (orc.n - q0) > q_budget:
          break
      _pool(needy)
      st = _launch()
      for _wave in range(max_waves):
        if st is None or len(st["pid"]) == 0:
            st = _launch()
            if st is None or len(st["pid"]) == 0:
                break
        if (orc.n - q0) > q_budget:
            break
        pid = st["pid"]
        A = len(pid)
        ts = st["center"][:, None] + base[None, :] * st["half"][:, None]
        rows = (P_X0[pid][:, None, :]
                + ts[:, :, None] * P_Uh[pid][:, None, :]).reshape(A * 6, d)
        Y = torch.cat([orc(rows[i:i + orc_chunk])
                       for i in range(0, len(rows), orc_chunk)]).reshape(A, 6, -1)
        noise = (1e-12 * Y.abs().amax((1, 2))).clamp_min(1e-300)
        devs = (Y[:, 1:5] - 0.5 * (Y[:, 0:4] + Y[:, 2:6])).abs().amax(-1)
        bad = devs > 30.0 * noise[:, None]                       # (A, 4)
        n = bad.sum(1)
        bf = bad.float()
        f = bf.argmax(1)
        l = 3 - bf.flip(1).argmax(1)
        consec = (l - f + 1) == n
        case_none = n == 0
        case_drop = (n > 2) | ((n == 2) & ~consec)
        pairc = (n == 2) & consec
        single = n == 1
        central = pairc & (f == 1)
        zoomable = (pairc | single) & ~central
        lo = torch.where(pairc, f + 1, f).clamp(0, 5)
        hi = (f + 2).clamp(0, 5)

        stats["nonconsec"] += int(case_drop.sum())
        stats["subnoise"] += int((case_none & st["zoomed"]).sum())
        grow = case_none.clone()          # kink outside bracket -> next rung
        drop = case_drop.clone()
        pend_mask = torch.zeros(A, dtype=torch.bool, device=dev)
        t_abs = torch.zeros(A, dtype=dt, device=dev)
        ci = central.nonzero(as_tuple=True)[0]
        if len(ci):
            aL, cL, rL = _side_fit(base[:3], Y[ci][:, :3])
            aR, cR, rR = _side_fit(base[3:], Y[ci][:, 3:])
            da = aL - aR
            sep = (cR - cL).abs().amax(1) + da.abs().amax(1)
            tu = ((cR - cL) * da).sum(1) / (da * da).sum(1).clamp_min(1e-300)
            nz = noise[ci]
            dirty = (rL > 30.0 * nz) | (rR > 30.0 * nz)
            nokink = ~dirty & (sep < torch.maximum(30.0 * (rL + rR), 300.0 * nz))
            okf = ~dirty & ~nokink & (tu.abs() < 0.6)
            stats["fit_dirty"] += int(dirty.sum())
            stats["fit_outspan"] += int((~dirty & ~nokink & ~okf).sum())
            stats["subnoise"] += int((nokink & st["zoomed"][ci]).sum())
            drop[ci[dirty | (~nokink & ~okf & ~dirty)]] = True
            grow[ci[nokink]] = True
            pend_mask[ci[okf]] = True
            t_abs[ci[okf]] = (st["center"][ci] + tu * st["half"][ci])[okf]
        # a sub-noise kink after zoom is unusable; a fresh bracket regrows
        drop |= (grow & st["zoomed"])
        grow &= ~st["zoomed"]

        # ---- accepted candidates: batched fingerprint, then store
        pi = pend_mask.nonzero(as_tuple=True)[0]
        if len(pi):
            P = len(pi)
            pp = pid[pi]
            ui = P_Uh[pp]
            xs = P_X0[pp] + t_abs[pi][:, None] * ui
            # probe at the FINAL (zoomed) bracket scale: it is verified
            # clean; the rung-delta scale straddles the neighbour kinks that
            # forced the zoom and false-rejects our own kink.
            dd = st["half"][pi]
            ss, hh = 0.25 * dd, 0.05 * dd
            U = torch.randn(P, 3, d, generator=gen, device=dev, dtype=dt)
            U = U - (U @ ui.unsqueeze(2)) * ui.unsqueeze(1)
            U = U / U.norm(dim=2, keepdim=True).clamp_min(1e-30)
            dirs = torch.cat([ui.unsqueeze(1), U], 1)            # (P, 4, d)
            bases = torch.stack([xs + ss[:, None] * ui,
                                 xs - ss[:, None] * ui], 1)      # (P, 2, d)
            pm = torch.tensor([1.0, -1.0], dtype=dt, device=dev)
            probes = (bases[:, :, None, None, :]
                      + pm[None, None, None, :, None]
                      * hh[:, None, None, None, None]
                      * dirs[:, None, :, None, :]).reshape(P * 16, d)
            Yp = torch.cat([orc(probes[i:i + orc_chunk])
                            for i in range(0, len(probes), orc_chunk)]
                           ).reshape(P, 2, 4, 2, -1)
            sl = (Yp[:, :, :, 0] - Yp[:, :, :, 1]) / (2.0 * hh[:, None, None, None])
            J = sl[:, 0] - sl[:, 1]                              # (P, 4, O)
            aw = J[:, 0].norm(dim=1).clamp_min(1e-30)
            Ch = J[:, 0] / aw[:, None]
            rat = (J[:, 1:] @ Ch.unsqueeze(2)).squeeze(2) / aw[:, None]
            ang = torch.atan(((d - 1) * (rat ** 2).mean(1)).clamp_min(0.0)
                             .sqrt()) * (180.0 / 3.141592653589793)
            good = ang <= match_deg
            stats["fp_reject"] += int((~good).sum())
            stats["accepted"] += int(good.sum())
            gi = good.nonzero(as_tuple=True)[0]
            if len(gi):
                Hg = _phi(pre, xs[gi], frontier)
                och = P_ch[pp[gi]].tolist()
                oob = (t_abs[pi[gi]].abs() * P_Nu[pp[gi]]
                       / (wbn[P_ch[pp[gi]]] * P_Hn[pp[gi]])).tolist()
                for k, c_ in enumerate(och):
                    hsl[c_].append(Hg[k])
                    obs[c_].append(oob[k])
                    acc[c_] += 1
                    if len(obs[c_]) >= 9 and len(obs[c_]) % 3 == 0:
                        med = sorted(obs[c_])[len(obs[c_]) // 2]
                        s0 = 0
                        while s0 + 1 < R and eps_ladder[s0] < 2.0 * med:
                            s0 += 1
                        start[c_] = s0
            for c_ in P_ch[pp].tolist():
                fly[c_] -= 1                    # pend rows leave flight either way

        # ---- rung growth (with isolation re-check at the new rung)
        gi = grow.nonzero(as_tuple=True)[0]
        grown_keep = None
        if len(gi):
            nr = st["rung"][gi] + 1
            inr = nr < R
            stats["grow_overflow"] += int((~inr).sum())
            gi2 = gi[inr]
            nr = nr[inr]
            if len(gi2):
                iso = _iso_rows(pre, P_X0[pid[gi2]], P_Uh[pid[gi2]],
                                P_ch[pid[gi2]], P_D[pid[gi2], nr],
                                frontier, Wl, bl)
                stats["grow_iso"] += int((~iso).sum())
                grown_keep = (gi2[iso], nr[iso])
        # ---- zoom updates
        zi_ = zoomable & ~drop
        zidx = zi_.nonzero(as_tuple=True)[0]
        zn = st["zi"][zidx] + 1
        stats["zoomcap"] += int((zn > 8).sum())
        zok = zidx[zn <= 8]
        u_lo, u_hi = base[lo[zok]], base[hi[zok]]
        # ---- assemble surviving state
        parts = []
        if grown_keep is not None and len(grown_keep[0]):
            g2, nr = grown_keep
            parts.append({"pid": pid[g2], "rung": nr,
                          "center": torch.zeros(len(g2), dtype=dt, device=dev),
                          "half": P_D[pid[g2], nr],
                          "zoomed": torch.zeros(len(g2), dtype=torch.bool,
                                                device=dev),
                          "zi": torch.zeros(len(g2), dtype=torch.long,
                                            device=dev)})
        if len(zok):
            parts.append({"pid": pid[zok], "rung": st["rung"][zok],
                          "center": st["center"][zok]
                          + 0.5 * (u_lo + u_hi) * st["half"][zok],
                          "half": 0.75 * (u_hi - u_lo) * st["half"][zok],
                          "zoomed": torch.ones(len(zok), dtype=torch.bool,
                                               device=dev),
                          "zi": st["zi"][zok] + 1})
        # rows that vanished from state (drop, failed growth, zoom-cap) leave
        # flight; pend rows were already decremented above
        stay = torch.zeros(A, dtype=torch.bool, device=dev)
        if grown_keep is not None and len(grown_keep[0]):
            stay[grown_keep[0]] = True
        stay[zok] = True
        gone = (~stay & ~pend_mask).nonzero(as_tuple=True)[0]
        for c_ in P_ch[pid[gone]].tolist():
            fly[c_] -= 1
        if len(parts) == 0:
            st = None
        elif len(parts) == 2:
            st = _cat(parts[0], parts[1])
        else:
            st = parts[0]
        # top-up with fresh launches
        nl = _launch()
        if nl is not None and len(nl["pid"]):
            st = nl if st is None or len(st["pid"]) == 0 else _cat(st, nl)
        if st is None or len(st["pid"]) == 0:
            break
    return {c: (hsl[c], obs[c]) for c in channels}, stats


def _solve(hs, Din, gen=None):
    """Robust homogeneous solve for [w|b] (unit): LTS (refit on the best 70%)
    -> trimmed SVD polish; inliers must stay the majority. Certificate:
    cross-validated -- solve on a 70% split of the kept rows, report the
    held-out median plane distance as est. Returns (v, est) or None."""
    M = torch.cat([torch.stack(hs),
                   torch.ones(len(hs), 1, dtype=hs[0].dtype, device=hs[0].device)], 1)
    Mn = M / M.norm(dim=1, keepdim=True)        # residual = plane distance
    N = len(Mn)
    _, _, Vh = torch.linalg.svd(Mn, full_matrices=False)
    v = Vh[-1]
    for _ in range(6):                          # LTS: refit on the best rows
        r = (Mn @ v).abs()
        keep_n = max(int(0.7 * len(r)), Din + 23)   # never trim below solvable
        if keep_n > len(r):
            return None
        keep = torch.argsort(r)[:keep_n]
        _, _, Vh = torch.linalg.svd(Mn[keep], full_matrices=False)
        v = Vh[-1]
    r = (Mn @ v).abs()
    keep = (r < 30.0 * r.median().clamp_min(1e-15)).nonzero(as_tuple=True)[0]
    if len(keep) < max(Din + 23, N // 2 + 1):   # inliers must be the majority
        return None
    _, _, Vh = torch.linalg.svd(Mn[keep], full_matrices=False)
    v = Vh[-1]
    perm = torch.argsort(torch.rand(len(keep), generator=gen,
                                    device=Mn.device, dtype=Mn.dtype))
    m = max(Din + 3, int(0.7 * len(keep)))
    A, B = keep[perm[:m]], keep[perm[m:]]
    _, _, Vh = torch.linalg.svd(Mn[A], full_matrices=False)
    est = float((Mn[B] @ Vh[-1]).abs().median())
    return v, est


@torch.no_grad()
def recover_layer(teacher, cons, frontier, device, only_channels=None,
                  gen=None, xcap=None, pool_mult=3, angle_gate=12.0, retries=1,
                  eps_ladder=(1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
                              1e-2, 3e-2),
                  want_mult=1.4, q_cap=150000, match_deg=20.0, verbose=False):
    """Recover layer `frontier` of `cons` (layers < frontier = the essentially
    exact prefix; layer `frontier` = the within ~1e-2 guess) from black-box
    `teacher` via guided kink location, ALL channels batched through one wave
    engine. Returns (W_ref [unit [w|b] rows], b_ref, refined_mask,
    n_oracle_queries). angle_gate: final sanity -- reject a solve further than
    this (deg) from the guess. q_cap: oracle budget PER CHANNEL (the engine's
    total budget is q_cap * n_channels)."""
    import copy, math
    pre = copy.deepcopy(cons).double().to(device).eval()
    orc = _Oracle(copy.deepcopy(teacher).double().to(device).eval())
    if gen is None:
        gen = torch.Generator(device=device).manual_seed(0)
    if xcap is None:                    # reach ~4x the typical input norm
        xcap = 4.0 * (pre.layers[0].weight.shape[1] ** 0.5)
    Wl = pre.layers[frontier].weight.detach()
    bl = pre.layers[frontier].bias.detach()
    Cout, Din = Wl.shape
    W_ref = Wl.clone(); b_ref = bl.clone()
    refined = torch.zeros(Cout, dtype=torch.bool, device=device)
    todo = list(range(Cout)) if only_channels is None else list(only_channels)
    for attempt in range(retries + 1):
        if not todo:
            break
        want = int(want_mult * Din) + attempt * (Din // 2)
        got, cst = _collect_layer(orc, pre, frontier, Wl, bl, todo, gen, want,
                                  xcap, pool_mult * (attempt + 1), eps_ladder,
                                  q_cap * len(todo), match_deg=match_deg)
        if verbose:
            print(f"    [loc] attempt {attempt} seed outcomes: {cst}",
                  flush=True)
        still = []
        for c in todo:
            hs, obs = got[c]
            sol = None
            if len(hs) >= Din + 23:
                # a kink decades below this neuron's typical error level is a
                # stray deeper-layer kink caught before the walk floor kicked
                # in -- drop it up front
                med = sorted(obs)[len(obs) // 2]
                hs = [h for h, o in zip(hs, obs) if o > 0.1 * med]
                if len(hs) >= Din + 23:
                    sol = _solve(hs, Din, gen=gen)
            if sol is None:
                still.append(c)
                if verbose:
                    print(f"    [loc] c={c}: no solve ({len(hs)} pts)",
                          flush=True)
                continue
            v, est = sol
            gv = torch.cat([Wl[c], bl[c].reshape(1)])
            gv = gv / gv.norm().clamp_min(1e-30)
            cos = float((v @ gv).clamp(-1.0, 1.0))
            if cos < 0:
                v, cos = -v, -cos
            dist_g = float((v - gv).abs().max())
            # accept only a solve that is (cross-validated) tighter than the
            # guess it replaces and that stayed near it (a wrong lock lands
            # far away; a correct one from a ~1e-2 guess sits at ~0.5deg).
            if est > max(1e-7, 0.05 * dist_g):
                still.append(c)
                if verbose:
                    print(f"    [loc] c={c}: cert reject (est {est:.2e} "
                          f"vs dist {dist_g:.2e})", flush=True)
                continue
            if math.degrees(math.acos(abs(cos))) > angle_gate:
                still.append(c)
                if verbose:
                    print(f"    [loc] c={c}: angle-gate reject", flush=True)
                continue
            W_ref[c] = v[:-1]; b_ref[c] = v[-1]
            refined[c] = True
        todo = still
    return (W_ref.to(cons.layers[frontier].weight.dtype),
            b_ref.to(cons.layers[frontier].bias.dtype), refined, orc.n)
