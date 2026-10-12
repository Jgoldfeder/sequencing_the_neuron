"""Black-box HEAD refiner for (leaky-)ReLU MLPs: exact head-column DIRECTIONS from
kink jumps, with no knowledge of the layers below.

Across the kink surface of a last-hidden unit j the output Jacobian changes by a
rank-1 matrix,  J_after - J_before = (1 - slope) * W_head[:, j] * g_j^T ,  whose
left factor is the head column itself. So: take a committee member's (approximate)
unit j, find where its pre-activation crosses zero along a line, snap to the TRUE
kink on the black box (two-sided collinear line fits, as kink_solve.locate_light),
and the difference of the two side slopes IS the column, up to the free positive
scale (the last-hidden unit's gauge) and a sign that the consensus column fixes.
~10 queries per attempt per unit, independent of the input dimension.

The consensus columns (head_consensus_columns) are only used for IDENTITY: a kink
found along the line may belong to a different unit (another last-hidden unit, or
a unit of an earlier layer, whose jump column is a mixture); a recovered column is
accepted only if it matches the consensus column of unit j (|cos| > match_thr).
"""
import math
import torch
from scipy.optimize import linear_sum_assignment
from align import scale_normalize_, _preacts
from kink_solve import _linefit


@torch.no_grad()
def _act_perm(ref_Z, Zr):
    """perm[i] = unit of the other net matched to ref unit i (pre-activation correlation, Hungarian)."""
    A = ref_Z - ref_Z.mean(0); B = Zr - Zr.mean(0)
    A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12); B = B / B.norm(dim=0, keepdim=True).clamp_min(1e-12)
    C = (A.T @ B)
    rows, cols = linear_sum_assignment((-C).cpu().numpy())
    perm = [0] * C.shape[0]
    for i, j in zip(rows, cols):
        perm[i] = int(j)
    return torch.tensor(perm, device=ref_Z.device), C[rows, cols]


@torch.no_grad()
def head_consensus_columns(pop, probe, ref_idx=0, eps=0.02, quorum_ratio=0.625):
    """Teacher-free head-column consensus in member ref_idx's last-hidden frame.
    Members' last-hidden units are paired to the reference by activation matching
    on `probe`; head columns are unit-normalized (magnitude = gauge); per column,
    the code's consensus rule (members within `eps` max-element of a seed, quorum)
    -> cluster mean. Returns cols (out, n3) with NaN columns where no quorum,
    the per-member perms, and the per-column cluster sizes."""
    mems = [m.clone() for m in pop]
    for m in mems:
        scale_normalize_(m)
    Z = [_preacts(m, probe)[-1] for m in mems]
    P = len(mems); quorum = math.ceil(quorum_ratio * P)
    perms, U = [], []
    for i, m in enumerate(mems):
        p, _ = _act_perm(Z[ref_idx], Z[i])
        perms.append(p)
        H = m.layers[-1].weight[:, p]
        U.append((H / H.norm(dim=0, keepdim=True).clamp_min(1e-12)).T)   # (n3, out)
    U = torch.stack(U)                                                     # (P, n3, out)
    n3, out = U.shape[1], U.shape[2]
    cols = torch.full((out, n3), float("nan"), device=U.device, dtype=U.dtype)
    sizes = torch.zeros(n3, dtype=torch.long, device=U.device)
    for j in range(n3):
        best = None
        for r in range(P):
            mem = (U[:, j] - U[r, j]).abs().amax(-1) <= eps
            k = int(mem.sum())
            if k >= quorum and (best is None or k > best[0]):
                best = (k, mem)
        if best is not None:
            c = U[best[1], j].mean(0)
            cols[:, j] = c / c.norm()
            sizes[j] = best[0]
    return cols, perms, sizes


@torch.no_grad()
def _kink_jump(oracle, X0, U, R, tol=1e-12):
    """Batched: for lines X0 + t U, t in [-R, R], fit a line to 3 points per side,
    require each side exactly collinear (no foreign bend inside a side), intersect
    -> kink t*, and return the slope JUMP ds = sR - sL (the Jacobian jump along U).
    Mirrors kink_solve.locate_light (incl. its verification points), but returns ds."""
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
    tolv = torch.minimum(tol * scale, 1e-9 * dsn * R)
    ok = ((dsn * R > 1e-6 * scale) & (rL < tolv) & (rR < tolv)
          & (tstar > T[:, 2]) & (tstar < T[:, 3]))
    e = R / 500
    tm, tp = tstar - e, tstar + e
    Pv = torch.stack([X0 + tm[:, None] * U, X0 + tp[:, None] * U], 1)
    Yv = oracle(Pv.reshape(-1, d)).reshape(n, 2, -1)
    resL = (Yv[:, 0] - (aL + sL * tm[:, None])).abs().amax(1)
    resR = (Yv[:, 1] - (aR + sR * tp[:, None])).abs().amax(1)
    ok &= (resL < tolv) & (resR < tolv)
    return tstar, ds, ok, 8 * n


def _preact_last(net, X):
    """Last-hidden PRE-activations, differentiable (align._preacts is no_grad)."""
    h = X
    for layer in net.layers[:-2]:
        h = net.act(layer(h))
    return net.layers[-2](h)


def refine_head(oracle, ref_net, cons_cols, X_pool, R_ladder=(0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4, 3e-5, 1e-5),
                n_tries=12, match_thr=0.9, seed=0, verbose=False,
                R_polish=(3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5),
                units=None, require_cons=True, verify_tol=1e-6):
    """Exact unit head columns for every last-hidden unit of `ref_net` (its frame).

    oracle: callable (n, d) -> (n, out), fp64 black box.
    ref_net: fp64 member net giving approximate last-hidden units (+ the frame).
    cons_cols: (out, n3) consensus unit columns in ref frame (NaN = none); used
        for identity/sign only. Units with NaN consensus fall back to ref_net's
        own head column as the identity reference.
    X_pool: (N, d) inputs from the query distribution (base points).
    Per unit, per try: a base point x0 with ref pre-activation of unit j, the line
    direction = ref's input gradient of that unit at x0 (its kink normal estimate),
    the ref's zero crossing along it (member-only, no queries), then the black-box
    kink jump on a bracket ladder (largest first). Accept the first bracket whose
    jump column matches the unit's reference column.
    VERIFICATION: a column is accepted only when two INDEPENDENT measurements
    (different base points / lines) agree to verify_tol (max-abs). A single bend
    contaminated by a second nearby kink can pass the collinearity checks and
    still match the consensus column at cos 0.99; two contaminated bends on
    different lines never agree to 1e-6. Each try measures every unit twice.
    Returns dict(cols (out, n3) unit columns [NaN where failed], ok, cos_ref,
    tstar_err (ref zero vs true kink), queries, tries)."""
    dev = X_pool.device; dt = X_pool.dtype
    n3 = ref_net.layers[-2].weight.shape[0]
    out = ref_net.layers[-1].weight.shape[0]
    refH = ref_net.layers[-1].weight
    refU = refH / refH.norm(dim=0, keepdim=True).clamp_min(1e-12)
    ident = cons_cols.clone().to(dev, dt) if cons_cols is not None else refU.clone()
    nanc = torch.isnan(ident).any(0)
    ident[:, nanc] = refU[:, nanc]
    cols = torch.full((out, n3), float("nan"), device=dev, dtype=dt)
    ok_all = torch.zeros(n3, dtype=torch.bool, device=dev)
    cos_all = torch.zeros(n3, device=dev, dtype=dt)
    terr_all = torch.full((n3,), float("nan"), device=dev, dtype=dt)
    tries_all = torch.zeros(n3, dtype=torch.long, device=dev)
    queries = 0
    gen = torch.Generator(device="cpu").manual_seed(seed)
    todo = torch.arange(n3, device=dev) if units is None else torch.as_tensor(units, device=dev).long()
    if require_cons and cons_cols is not None:
        todo = todo[~nanc[todo]]                  # identity needs a consensus column
    pending = {}                                  # unit -> list of measured unit columns (unverified)
    for attempt in range(n_tries):
        if todo.numel() == 0:
            break
        todo = torch.cat([todo, todo])            # two independent measurements per unit per try
        m = todo.numel()
        # --- base points + line directions from the reference member (no queries)
        idx = torch.randint(0, len(X_pool), (m,), generator=gen).to(dev)
        x0 = X_pool[idx].clone().requires_grad_(True)
        with torch.enable_grad():
            z = _preact_last(ref_net, x0)                           # (m, n3)
            zj = z[torch.arange(m, device=dev), todo]
            g, = torch.autograd.grad(zj.sum(), x0)
        x0 = x0.detach(); g = g.detach()
        Udir = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
        # ref zero crossing along Udir: a few secant/bisection steps on the member
        t = -zj.detach() / (g * Udir).sum(1).clamp_min(1e-12)      # linear estimate
        for _ in range(8):                                          # Newton on the member's unit (exact per region)
            with torch.enable_grad():
                xt = (x0 + t[:, None] * Udir).detach().requires_grad_(True)
                zt = _preact_last(ref_net, xt)[torch.arange(m, device=dev), todo]
                gt, = torch.autograd.grad(zt.sum(), xt)
            t = t - zt.detach() / (gt.detach() * Udir).sum(1).clamp_min(1e-12)
        X0 = (x0 + t[:, None] * Udir).detach()
        with torch.no_grad():
            zchk = _preact_last(ref_net, X0)[torch.arange(m, device=dev), todo].abs()
        # --- black-box kink jump on a ZOOMING bracket ladder: each bracket's chord
        #     intersection re-centers the next (smaller) bracket, so the strict
        #     collinearity test eventually runs on a tight bracket around the true
        #     kink, free of foreign bends; accept on ok + identity match.
        found = torch.zeros(m, dtype=torch.bool, device=dev)
        Xc = X0.clone()
        for R0 in R_ladder:
            rem = ~found
            if not bool(rem.any()):
                break
            ridx = rem.nonzero().flatten()
            R = torch.full((len(ridx),), R0, device=dev, dtype=dt)
            tstar, ds, ok, q = _kink_jump(oracle, Xc[ridx], Udir[ridx], R)
            queries += q
            u = ds / ds.norm(dim=1, keepdim=True).clamp_min(1e-300)   # (k, out)
            units = todo[ridx]
            c = (u * ident[:, units].T).sum(1)
            acc = ok & (c.abs() > match_thr)
            if bool(acc.any()):
                ui = units[acc]
                ucol = u[acc] * torch.sign(c[acc])[:, None]
                acc_cos = c[acc].abs()
                # POLISH: re-measure the jump on a tight bracket centred on the kink
                # (free of foreign bends, fd error ~1e-16*|Y|/R): use it when it
                # passes the strict checks and agrees with the coarse column.
                if R_polish:
                    # fd slope error ~ eps*|Y|/R, foreign-bend risk ~ R: take the LARGEST
                    # bracket centred on the kink that passes the strict checks
                    Xk = Xc[ridx][acc] + tstar[acc][:, None] * Udir[ridx][acc]
                    done = torch.zeros(len(Xk), dtype=torch.bool, device=dev)
                    for Rp0 in R_polish:
                        if Rp0 >= R0 or bool(done.all()):
                            continue
                        Rp = torch.full((len(Xk),), Rp0, device=dev, dtype=dt)
                        _, ds2, ok2, q2 = _kink_jump(oracle, Xk, Udir[ridx][acc], Rp)
                        queries += q2
                        u2 = ds2 / ds2.norm(dim=1, keepdim=True).clamp_min(1e-300)
                        u2 = u2 * torch.sign((u2 * ucol).sum(1))[:, None]
                        good = ok2 & ((u2 * ucol).sum(1) > 0.999) & ~done
                        ucol = torch.where(good[:, None], u2, ucol)
                        done |= good
                terr = ((Xc[ridx][acc] - X0[ridx][acc]) * Udir[ridx][acc]).sum(1) + tstar[acc]
                for k, uj in enumerate(ui.tolist()):
                    pending.setdefault(uj, []).append((ucol[k].clone(), float(acc_cos[k]), float(terr[k])))
                found[ridx[acc]] = True
            # re-center the rest on the estimated kink when the estimate is sane
            # (inside the bracket); otherwise keep the center and just shrink
            keep = ~acc & torch.isfinite(tstar) & (tstar.abs() < R)
            if bool(keep.any()):
                ki = ridx[keep]
                Xc[ki] = Xc[ki] + tstar[keep][:, None] * Udir[ki]
        # VERIFY: accept a unit when two independent measurements agree
        n_new = 0
        for uj, meas in list(pending.items()):
            if ok_all[uj] or len(meas) < 2:
                continue
            done_u = False
            for a in range(len(meas)):
                for b in range(a + 1, len(meas)):
                    if float((meas[a][0] - meas[b][0]).abs().max()) < verify_tol:
                        cols[:, uj] = meas[a][0]; ok_all[uj] = True
                        cos_all[uj] = meas[a][1]; terr_all[uj] = meas[a][2]
                        done_u = True; n_new += 1
                        break
                if done_u:
                    break
            if done_u:
                del pending[uj]
            elif len(meas) > 6:                   # keep the pending list bounded
                pending[uj] = meas[-4:]
        todo = torch.unique(todo)
        tries_all[todo] += 1
        if verbose:
            print(f"  [head-refine] try {attempt + 1}: +{n_new} verified ({int(ok_all.sum())}/{n3} total), "
                  f"{len(pending)} pending | queries so far {queries}", flush=True)
        todo = todo[~ok_all[todo]]
    return dict(cols=cols, ok=ok_all, cos_ref=cos_all, tstar_err=terr_all,
                queries=queries, tries=tries_all)
