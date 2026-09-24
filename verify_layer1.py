"""Efficient cryptanalytic verify/refine for layer-1 neurons (black box, queries only).

Two operations, both exploiting that we already hold a GOOD GUESS (w, b) for the
neuron -- a consensus hyperplane -- so we never reconstruct anything blind:

  1. verify_neuron(guess, eps): is the guess within eps of the true neuron?
       COST ~ O(k) queries, INDEPENDENT of input dimension d.
       A layer-1 kink's gradient jump is rank-1 along the true normal w*:
       crossing it, the directional-derivative jump along any direction u is
       (w*.u) * c0 for a fixed output vector c0. So probing a handful of random
       directions PERPENDICULAR to the guess normal detects any true-normal
       component the guess is missing -- if the guess is exact those all read 0.
       k directional probes (2 queries each) estimate the angle error; a short
       line search along the normal gives the offset error. No d-sized sweep.

  2. refine_neuron(guess): sharpen a within-eps guess to high precision.
       Offset: 1D binary search along the normal -> machine precision, O(1) queries.
       Normal: recover w* from the rank-1 jump. A dense d-vector needs ~d linear
       measurements (information-theoretic floor), done here at that floor via one
       forward-difference sweep projected onto c0 -- ~2d queries, vs 4d blind.

Gauge: LeakyReLU is positively homogeneous, so (w, b) is defined only up to a
positive scale. Everything is in the normalized frame (unit normal w/||w||, offset
-b/||w||); only the hyperplane is observable. Probing runs in float64.
"""
import math
import time

import torch


class _Oracle:
    """Wraps the (double) teacher and counts every input row queried -- so the
    reported query cost is the true number of black-box samples used."""

    def __init__(self, td):
        self.td = td
        self.n = 0

    @torch.no_grad()
    def __call__(self, X):
        self.n += int(X.shape[0])
        return self.td(X)


class _FastOracle:
    """fp32 TRIAGE oracle: same counting contract, but queries run through the
    float teacher (~30-60x faster than fp64 conv on consumer GPUs) and are cast
    back to double. Used only to DETECT/RANK kinks -- every accept still goes
    through the fp64 oracle's bisect + sweep + gates, so fp32 noise can waste a
    probe but never poison a result."""

    def __init__(self, tf):
        self.tf = tf
        self.n = 0

    @torch.no_grad()
    def __call__(self, X):
        self.n += int(X.shape[0])
        return self.tf(X.float()).double()


@torch.no_grad()
def _double_teacher(teacher):
    return teacher.clone().double()


@torch.no_grad()
def _plane(w, b):
    n = w.norm().clamp_min(1e-30)
    return w / n, b / n


@torch.no_grad()
def _slope(orc, x, direction, h):
    """Directional derivative of the network at x along `direction` (central)."""
    return (orc(x.unsqueeze(0) + h * direction) - orc(x.unsqueeze(0) - h * direction))[0] / (2 * h)


@torch.no_grad()
def _refine_node(orc, x0, w_hat, ts, i):
    """Sub-grid kink location at scan node i via piecewise-linear line intersection."""
    a, bb = ts[i - 1].item(), ts[i + 1].item()
    h = (bb - a) * 1e-3
    ga = orc((x0 + a * w_hat).unsqueeze(0))[0]
    sL = (orc((x0 + (a + h) * w_hat).unsqueeze(0))[0] - ga) / h
    gb = orc((x0 + bb * w_hat).unsqueeze(0))[0]
    sR = (gb - orc((x0 + (bb - h) * w_hat).unsqueeze(0))[0]) / h
    dslope = sL - sR
    rhs = (gb - ga) - sR * bb + sL * a
    t = float((dslope * rhs).sum() / (dslope * dslope).sum().clamp_min(1e-30))
    return t if a <= t <= bb else ts[i].item()


@torch.no_grad()
def _find_kinks(orc, x0, w_hat, window, n_scan, kink_tol, n_cand):
    """Kinks along w_hat within the window, refined and sorted by |t| (nearest the
    guess plane first). Returns up to n_cand candidate offsets, or [] if none."""
    ts = torch.linspace(-window, window, n_scan, device=x0.device, dtype=torch.float64)
    g = orc(x0.unsqueeze(0) + ts.unsqueeze(1) * w_hat.unsqueeze(0))          # (n_scan, O)
    kink = ((g[2:] - g[1:-1]) - (g[1:-1] - g[:-2])).norm(dim=1)              # slope change
    nodes = torch.arange(1, n_scan - 1, device=x0.device)[kink > kink_tol]
    if nodes.numel() == 0:
        return []
    order = ts[nodes].abs().argsort()                                       # nearest first
    return [_refine_node(orc, x0, w_hat, ts, int(nodes[o]))
            for o in order[:max(1, n_cand)]]


# --------------------------------------------------------------------------- #
#  1. VERIFY  --  is the guess within eps?   O(k) queries, d-independent.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _angle_estimate(orc, x_star, w_hat, k, s, h, gen):
    """Randomized tilt estimate (degrees) between the true normal at kink x_star
    and the guess normal w_hat, from k perpendicular directional-jump probes.
    O(k) queries, independent of input dimension."""
    dev = x_star.device
    d = w_hat.numel()
    xl, xr = x_star - s * w_hat, x_star + s * w_hat
    C = _slope(orc, xr, w_hat, h) - _slope(orc, xl, w_hat, h)   # dJ.w_hat = (w*.w_hat)c0
    a_w = C.norm().clamp_min(1e-30)
    C_hat = C / a_w
    ss = 0.0
    for _ in range(k):
        u = torch.randn(d, device=dev, dtype=torch.float64, generator=gen)
        u = u - (u @ w_hat) * w_hat
        u = u / u.norm().clamp_min(1e-30)
        Ju = _slope(orc, xr, u, h) - _slope(orc, xl, u, h)      # dJ.u = (w*.u)c0
        ss += (float(Ju @ C_hat) / float(a_w)) ** 2
    tan2 = max((d - 1) * ss / max(k, 1), 0.0)
    return math.degrees(math.atan(math.sqrt(tan2)))


@torch.no_grad()
def verify_neuron(orc, w, b, base_x, eps_offset=1e-2, eps_angle=1.0,
                  k=16, window=0.25, n_scan=21, s=1e-3, h=1e-4, gen=None,
                  n_cand=3):
    """Efficiently test whether hyperplane (w, b) is within (eps_offset, eps_angle)
    of the teacher's true layer-1 neuron. Returns dict:
      within (bool), offset (signed distance guess->true plane, None if no kink),
      angle_deg (estimated tilt between guess and true normal), queries.
    THIS neuron is identified by its normal, not proximity: among the n_cand kinks
    nearest the guess plane we keep the one whose tilt is smallest (so a nearer
    *other* neuron's plane can't force a false 'not within'). Cost is O(k), NOT O(d)."""
    dev = base_x.device
    w_hat, b_hat = _plane(w.to(dev).double(), b.to(dev).double())
    x0 = base_x.double()
    x0 = x0 - (w_hat @ x0 + b_hat) * w_hat                       # onto guess plane

    cands = _find_kinks(orc, x0, w_hat, window, n_scan, kink_tol=1e-9, n_cand=n_cand)
    if not cands:
        return {"within": False, "offset": None, "angle_deg": None,
                "reason": "no_kink_in_window", "queries": orc.n}

    best = None
    for t_star in cands:                                        # nearest first
        angle = _angle_estimate(orc, x0 + t_star * w_hat, w_hat, k, s, h, gen)
        within = (abs(t_star) <= eps_offset) and (angle <= eps_angle)
        rec = {"within": within, "offset": t_star, "angle_deg": angle,
               "queries": orc.n}
        if best is None or angle < best["angle_deg"]:           # most parallel = this neuron
            best = rec
        if angle <= eps_angle:                                  # found this neuron's kink
            best = rec
            break
    return best


@torch.no_grad()
def _offset_from_origin(orc, n_hat, s_exp, win, n_scan, bisect_iters):
    """Recover this neuron's offset by locating its kink along n_hat measured
    FROM THE ORIGIN, x(t) = t * n_hat.  Then n_hat . x_kink = t_kink exactly
    (||n_hat||=1), so b = -t_kink with NO (n* - n_hat).x_star cross term -- the
    off-origin projection in `b = -n_hat . x_star` otherwise amplifies the
    normal's sqrt(2(1-cos)) vector error (~1e-6) up by ||x_star|| into b.
    Scans a narrow window around the current estimate s_exp = -b_est and bisects
    on the slope change; returns t_kink, or None if no kink is bracketed."""
    dev = n_hat.device
    ts = torch.linspace(s_exp - win, s_exp + win, n_scan, device=dev, dtype=torch.float64)
    g = orc(ts.unsqueeze(1) * n_hat.unsqueeze(0))                        # (n_scan, O)
    kink = ((g[2:] - g[1:-1]) - (g[1:-1] - g[:-2])).norm(dim=1)
    nodes = torch.arange(1, n_scan - 1, device=dev)[kink > 1e-9]
    if nodes.numel() == 0:
        return None
    i = int(nodes[(ts[nodes] - s_exp).abs().argmin()])                  # kink nearest estimate
    lo, hi = ts[i - 1].item(), ts[i + 1].item()
    hh = (hi - lo) * 1e-4

    def slope_at(t):
        p = t * n_hat
        return (orc((p + hh * n_hat).unsqueeze(0))[0]
                - orc((p - hh * n_hat).unsqueeze(0))[0]) / (2 * hh)
    sL0, sR0 = slope_at(lo), slope_at(hi)
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        sm = slope_at(mid)
        if float((sm - sL0).norm()) < float((sm - sR0).norm()):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
#  2. REFINE  --  sharpen a within-eps guess.  offset O(1), normal ~2d (floor).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def refine_neuron(orc, w, b, base_x, window=0.25, n_scan=21, s=1e-3,
                  fd=1e-3, bisect_iters=40, normal=True, n_cand=3, gen=None,
                  support=None, chunk=4096, orc_fast=None, triage_deg=None):
    """Sharpen a within-eps guess (w, b) against the teacher.
    NB fd is the forward-difference step for the normal sweep. The net is
    piecewise-LINEAR, so the difference has no truncation error -- only rounding,
    ~eps/fd -- and a LARGER fd is better (fd=1e-3 -> normal err ~1e-9, vs ~1e-7
    at 1e-6). Cap it below the spacing to neighbouring kinks: fd>~3e-3 starts
    crossing them and corrupts columns (watch jump_ratio; it drops when this bites).
    Offset: binary-search the kink along the normal to ~machine precision (cheap).
    Normal (if normal=True): recover the true normal from the rank-1 gradient jump
    via one forward-difference sweep projected onto c0 (~2d queries -- the floor
    for a dense d-vector, vs 4d blind).
    support: optional index tensor -- sweep ONLY these coordinates and treat
    the normal as exactly zero elsewhere (valid when the normal's support is
    known a priori, e.g. a conv LAYER-1 receptive field in image space).
    chunk: sweep batch rows per oracle call (eye(d) at large d is O(d^2)
    memory; chunking is numerically identical).
    orc_fast: optional fp32 triage oracle -- used ONLY for kink detection and
    candidate ranking; the accepted kink's bisect + normal sweep stay on `orc`.
    triage_deg: if set, abstain BEFORE the expensive sweep when the best
    candidate's (cheap, randomized) angle estimate exceeds this many degrees.
    Returns dict: w_refined (unit, cpu), b_refined, offset, angle_deg, queries."""
    dev = base_x.device
    w_hat, b_hat = _plane(w.to(dev).double(), b.to(dev).double())
    d = w_hat.numel()
    x0 = base_x.double()
    x0 = x0 - (w_hat @ x0 + b_hat) * w_hat

    of = orc_fast if orc_fast is not None else orc
    # fp32 second differences carry ~1e-6-scale noise -- raise the kink tol
    # accordingly; real kinks at these window/node spacings sit well above it.
    ktol = 1e-9 if orc_fast is None else 1e-4
    cands = _find_kinks(of, x0, w_hat, window, n_scan, kink_tol=ktol, n_cand=n_cand)
    if not cands:
        return None
    # identify THIS neuron's kink cheaply (most parallel), then refine only that one
    if len(cands) == 1 and triage_deg is None:
        t0 = cands[0]
    else:
        est = [(t, _angle_estimate(of, x0 + t * w_hat, w_hat, 8, s, 1e-4, gen))
               for t in cands]
        t0, a0 = min(est, key=lambda te: te[1])
        if triage_deg is not None and a0 > triage_deg:
            return None                     # hopeless kink: skip bisect + sweep

    # --- offset: bracket the kink and bisect on the activation-pattern boundary.
    #     f is piecewise-linear along the ray; the kink is where the local slope
    #     changes. Bisect by comparing slopes of the two half-brackets. ---
    lo, hi = t0 - window / (n_scan - 1), t0 + window / (n_scan - 1)
    hh = (hi - lo) * 1e-4

    def slope_at(t):
        p = x0 + t * w_hat
        return (orc((p + hh * w_hat).unsqueeze(0))[0]
                - orc((p - hh * w_hat).unsqueeze(0))[0]) / (2 * hh)
    sL0, sR0 = slope_at(lo), slope_at(hi)
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        sm = slope_at(mid)
        # which side's slope does the midpoint match? move the opposite bound in.
        if float((sm - sL0).norm()) < float((sm - sR0).norm()):
            lo = mid
        else:
            hi = mid
    t_star = 0.5 * (lo + hi)
    x_star = x0 + t_star * w_hat
    out = {"offset": t_star, "queries": orc.n,
           "w_refined": w_hat.cpu(), "b_refined": -float(w_hat @ x_star),
           "angle_deg": None}
    if not normal:
        return out

    # --- normal: dJ = J_right - J_left is rank-1 = c0 (x) w*.  One forward-diff
    #     sweep on each side (share the baseline) gives dJ; project each column
    #     onto c0_hat to read w*[j].  ~2d queries. ---
    #     Larger fd = less rounding (~eps/fd), hence a better normal -- but too
    #     large and some column's step crosses a NEIGHBOURING kink, corrupting it
    #     and collapsing the rank-1 signal.  So start at fd and back it off until
    #     jump_ratio S0/S1 shows a clean rank-1 jump; keep the largest clean fd.
    idxs = (torch.arange(d, device=dev) if support is None
            else support.to(dev).long())
    m = idxs.numel()
    xl, xr = x_star - s * w_hat, x_star + s * w_hat
    fL, fR = orc(xl.unsqueeze(0)), orc(xr.unsqueeze(0))                       # shared baselines

    def _sweep(xb, fd_try, base):
        """(m, O) forward-diff rows over the swept coordinates, chunked."""
        rows = []
        for i in range(0, m, chunk):
            ii = idxs[i:i + chunk]
            X = xb.unsqueeze(0).repeat(len(ii), 1)
            X[torch.arange(len(ii), device=dev), ii] += fd_try
            rows.append((orc(X) - base) / fd_try)
        return torch.cat(rows, 0)

    best = None
    for fd_try in (fd, fd / 3, fd / 10, fd / 30, fd / 100):
        JL = _sweep(xl, fd_try, fL)                                           # (m, O)
        JR = _sweep(xr, fd_try, fR)                                           # (m, O)
        dJ = JR - JL                                                          # (m, O), rank-1
        U, S, Vh = torch.linalg.svd(dJ.t(), full_matrices=False)              # c0 = top left-sing vec
        jr = float(S[0] / S[1].clamp_min(1e-30)) if S.numel() > 1 else float("inf")
        w_raw = torch.zeros(d, device=dev, dtype=torch.float64)
        w_raw[idxs] = dJ @ U[:, 0]                                            # (d,) ~ (c0.c0) w*
        n_try = w_raw / w_raw.norm().clamp_min(1e-30)
        if float(n_try @ w_hat) < 0:
            n_try = -n_try
        if best is None or jr > best[0]:
            best = (jr, n_try, S)
        if jr > 1e7:                                                          # clean rank-1: take it
            break
    _, n_hat, S = best
    cos = float((n_hat @ w_hat).clamp(-1.0, 1.0))
    out["angle_deg"] = math.degrees(math.acos(cos))
    out["w_refined"] = n_hat.cpu()
    # offset on refined normal: measure the kink along n_hat FROM THE ORIGIN so
    # b = -t_kink, free of the ||x_star||-amplified normal error in -n_hat.x_star.
    # s_exp already pins the offset to ~1e-5, so scan a TIGHT window around it
    # (a wide one grabs a neighbouring neuron's kink) and reject any pick that
    # lands too far from s_exp -- then b is never worse than -n_hat.x_star.
    s_exp = float(n_hat @ x_star)
    t_kink = _offset_from_origin(orc, n_hat, s_exp, win=2e-3, n_scan=n_scan,
                                 bisect_iters=bisect_iters)
    if t_kink is not None and abs(t_kink - s_exp) < 1e-3:
        out["b_refined"] = -t_kink
    else:
        out["b_refined"] = -s_exp                                          # fallback: old estimate
    out["jump_ratio"] = float(S[0] / S[1].clamp_min(1e-30)) if S.numel() > 1 else float("inf")
    out["queries"] = orc.n
    return out


# --------------------------------------------------------------------------- #
#  Batch driver over a list of consensus neurons.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def verify_layer1(teacher, neurons, X, device, eps_offset=1e-2, eps_angle=1.0,
                  k=16, refine=False, max_neurons=0, seed=0, n_base=3):
    """Verify (and optionally refine) hypothesized layer-1 neurons against the
    teacher. neurons: list of (w, b). Returns {"per": [...], "summary": {...}}.
    Black-box only; 'queries' fields report actual teacher samples used (NOT
    charged to the extraction budget)."""
    t0 = time.time()
    td = _double_teacher(teacher)
    gen = torch.Generator(device=device).manual_seed(seed)
    if X is None or len(X) == 0:
        pool = torch.randn(max(n_base, 1), teacher.layers[0].weight.shape[1],
                           device=device, dtype=torch.float64)
    else:
        cpu_gen = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randint(0, len(X), (max(n_base, 1),), generator=cpu_gen)
        pool = X[idx].to(device).double()

    capped = 0
    items = list(enumerate(neurons))
    if max_neurons and len(items) > max_neurons:
        capped = len(items) - max_neurons
        items = items[:max_neurons]

    per, total_q = [], 0
    for kk, (w, b) in items:
        orc = _Oracle(td)
        best = None
        for j in range(pool.shape[0]):
            r = verify_neuron(orc, w, b, pool[j], eps_offset, eps_angle, k=k, gen=gen)
            if r["offset"] is None:
                best = best or r
                continue
            if best is None or (r["within"] and not best["within"]) or \
               (r["offset"] is not None and best.get("offset") is not None
                    and abs(r["offset"]) < abs(best["offset"])):
                best = r
            if r["within"]:
                break
        rec = {"neuron": kk, **best}
        if refine and best.get("offset") is not None:
            rf = refine_neuron(orc, w, b, pool[0], gen=gen)
            if rf is not None:
                rec["refined"] = {"offset": rf["offset"], "angle_deg": rf["angle_deg"],
                                  "b": rf["b_refined"]}
                rec["w_refined"] = rf["w_refined"]
        total_q += orc.n
        per.append(rec)

    within = [r for r in per if r.get("within")]
    got = [r for r in per if r.get("offset") is not None]

    def med(vals):
        v = sorted(x for x in vals if x is not None)
        return v[len(v) // 2] if v else None
    summary = {
        "n": len(per), "n_within": len(within), "n_kink": len(got),
        "capped": capped, "refined": refine,
        "med_offset": med([abs(r["offset"]) for r in got]),
        "max_offset": max([abs(r["offset"]) for r in got], default=None),
        "med_angle": med([r.get("angle_deg") for r in got]),
        "max_angle": max([r["angle_deg"] for r in got
                          if r.get("angle_deg") is not None], default=None),
        "queries_total": total_q,
        "queries_per_neuron": round(total_q / max(len(per), 1)),
        "wall_s": round(time.time() - t0, 2),
    }
    return {"per": per, "summary": summary}


@torch.no_grad()
def extract_neuron_exact(orc, w, b, base_x, s=1e-2, r=1e-3, m_mult=1.3,
                         window=0.25, n_scan=21, min_jump_ratio=50.0,
                         max_angle_deg=5.0, base_scale=1.0, gen=None):
    """EXACTLY extract a layer-1 neuron by fitting the affine region on each side of
    its kink and reading the rank-1 gradient jump. The net is piecewise-linear, so the
    per-side affine fit has NO truncation error -> ~float precision (vs the fd-limited
    refine_neuron). ~2d queries, one shot, no per-digit cost.

    GUARDED so it can never silently corrupt a good neuron: returns None unless the
    recovered normal is a clean rank-1 jump (jump_ratio >= min_jump_ratio) AND within
    max_angle_deg of the guess (rejects locking onto a neighbouring neuron's kink).
    Returns {w (unit, cpu), b, angle_deg, jump_ratio, queries} on success, else None."""
    what, bhat = _plane(w.double(), float(b))            # b may be a python float
    dev = base_x.device
    d = what.numel()
    # Seed NEAR the foot of the guess plane (small norm), not far out: then the guessed
    # neuron's own kink sits at t~=0 (only it toggles in the window; other neurons have
    # O(1) pre-activation here), so the sweep can't lock onto a neighbour.
    tang = base_x.double()
    tang = tang - (what @ tang) * what                          # in-plane direction
    tang = tang / tang.norm().clamp_min(1e-30)
    x0 = -bhat * what + base_scale * tang                       # foot + small tangential
    cands = _find_kinks(orc, x0, what, window, n_scan, 1e-9, 3)
    if not cands:
        return {"ok": False, "reason": "no_kink"}
    t0 = min(cands, key=lambda t:                                # nearest, most parallel
             _angle_estimate(orc, x0 + t * what, what, 8, s, 1e-4, gen))
    xs = x0 + t0 * what

    def region_jac(xc):                                          # exact affine gradient
        M = int(m_mult * d)
        D = torch.randn(M, d, device=dev, dtype=torch.float64, generator=gen)
        D = D / D.norm(dim=1, keepdim=True)
        P = xc.unsqueeze(0) + r * D
        f = orc(P)
        A = torch.cat([P, torch.ones(M, 1, device=dev, dtype=torch.float64)], 1)
        sol = torch.linalg.lstsq(A, f).solution
        res = (A @ sol - f).norm(dim=1)                          # drop kink-crossing pts
        keep = res <= 3 * res.median().clamp_min(1e-30)
        if int(keep.sum()) >= d + 1:                             # refit only if still overdetermined
            sol = torch.linalg.lstsq(A[keep], f[keep]).solution
        return sol[:d]                                           # (d, O)

    dJ = region_jac(xs + s * what) - region_jac(xs - s * what)   # rank-1 = c0 (x) w*
    U, S, Vh = torch.linalg.svd(dJ, full_matrices=False)
    jr = float(S[0] / S[1].clamp_min(1e-30)) if S.numel() > 1 else float("inf")
    n = U[:, 0]
    if float(n @ what) < 0:
        n = -n
    ang = math.degrees(math.acos(float((n @ what).clamp(-1.0, 1.0))))
    if jr < min_jump_ratio:                                      # jump not cleanly rank-1
        return {"ok": False, "reason": "low_jump", "jump_ratio": jr, "angle_deg": ang}
    if ang > max_angle_deg:                                      # locked onto a neighbour
        return {"ok": False, "reason": "high_angle", "jump_ratio": jr, "angle_deg": ang}
    return {"ok": True, "w": n.cpu(), "b": -float(n @ xs), "angle_deg": ang,
            "jump_ratio": jr, "queries": orc.n}


def format_summary(s):
    def f(x, spec=".2e"):
        return format(x, spec) if x is not None else "n/a"
    out = (f"{s['n_within']}/{s['n']} within-eps ({s['n_kink']} kink) | "
           f"off med {f(s['med_offset'])} max {f(s['max_offset'])} | "
           f"angle med {f(s['med_angle'], '.3f')}° max {f(s['max_angle'], '.3f')}°")
    if s.get("refined"):
        out += " | refined"
    out += (f" | ~{s['queries_per_neuron']} q/neuron ({s['queries_total']} q) "
            f"| {s['wall_s']}s")
    if s.get("capped"):
        out += f" | (+{s['capped']} not probed)"
    return out
