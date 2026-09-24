"""Why does --peeltry stall on L3 (C5, 120 ch) after L1/L2 peel exactly?

Loads the cheat LeNet checkpoint (L1/L2 frozen exact, L3 = trained guess),
aligns to the teacher, and answers with measurements:

Phase A (cheap, batched, teacher-side knowledge -- DIAGNOSTIC ONLY):
  per channel, over the SAME probe distribution the refiner uses:
    - Newton projection success onto the guess plane
    - safe window = min(0.9 * safe_r(L1+L2 kinks), 1.0)  [abstain if < 1e-4]
    - dist_ray  = |t| to the TRUE channel-c kink along the scan ray
    - target_in = dist_ray <= window  (can the scan even see the true kink?)
    - n_nearer  = teacher kinks strictly nearer the plane than the target
      (the scan keeps the 15 nearest; target beyond #15 is never angle-tested)

Phase B (instrumented replica of _cnn_refine_layer, attacker-faithful):
  per channel, per probe: which gate kills it --
    newton_fail / window_small / no_kink / angle_fail(angle) / accept;
  per channel: solved / few_kinks / null_bad / runaway, + post-refine eps.

Usage: python diag_l3_stall.py [--ck ...] [--probes-a 100] [--probes-b 40]
       [--chans all|bad|N] [--device cuda]
"""
import argparse
import math
import time
from collections import Counter

import torch
import torch.nn.functional as F

from align import cnn_canonicalize_, cnn_align_to_
from data import make_teacher_cnn
from method import _cnn_prefix_input
from nets import ConvNet
from verify_layer1 import _Oracle, refine_neuron

CFGS = [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)]
FC = (84,)
FRONTIER = 2
CK = ("recon/mergedbest_cnn_cheat__1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_"
      "16-120-5-1-0-0_fc84__s0_final.pt")


def rows(net, li):
    W, b = net.layers[li].weight, net.layers[li].bias
    return torch.cat([W.view(W.shape[0], -1), b[:, None]], 1)


def batched_safe_r(pre, x0, uh, frontier):
    """_cnn_safe_r vectorized over a batch: distance along +/-uh from x0 before
    ANY earlier-layer (< frontier) preact flips. x0, uh: (P, idim)."""
    eps = 1e-4
    P = x0.shape[0]
    xa = x0.view(P, *pre.input_shape)
    xb = (x0 + eps * uh).view(P, *pre.input_shape)
    safe = torch.full((P,), float("inf"), dtype=x0.dtype, device=x0.device)
    for i in range(min(frontier, pre.n_conv)):
        pa, pb = pre.layers[i](xa), pre.layers[i](xb)
        dp = (pb - pa) / eps
        t = torch.where(dp.abs() > 1e-12, -(pa / dp),
                        torch.full_like(pa, 1e30)).abs()
        safe = torch.minimum(safe, t.view(P, -1).min(1).values)
        xa, xb = pre.act(pa), pre.act(pb)
        if pre.pools[i] > 0:
            xa, xb = F.avg_pool2d(xa, pre.pools[i]), F.avg_pool2d(xb, pre.pools[i])
    return safe


def newton_project(pre, wg, bg, base, rf, iters=6):
    """Batched Newton projection of base (P, idim) onto the guess plane
    {x : wg . phi(x) + bg = 0}. Returns x0, uh, nu, ok."""
    xcur = base
    uh = torch.zeros_like(base)
    nu = torch.zeros(base.shape[0], dtype=base.dtype, device=base.device)
    for _ in range(iters):
        xg = xcur.detach().clone().requires_grad_(True)
        s0 = _cnn_prefix_input(pre, xg, FRONTIER, rf) @ wg + bg
        u = torch.autograd.grad(s0.sum(), xg)[0]
        nu = u.norm(dim=1)
        ok = nu > 1e-12
        uh = torch.where(ok[:, None], u / nu.clamp_min(1e-30)[:, None], uh)
        xcur = torch.where((ok & (s0.abs() >= 1e-9))[:, None],
                           (xg - (s0 / nu.clamp_min(1e-30) ** 2)[:, None] * u),
                           xg).detach()
    with torch.no_grad():
        s_fin = _cnn_prefix_input(pre, xcur, FRONTIER, rf) @ wg + bg
    return xcur.detach(), uh.detach(), nu.detach(), s_fin.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default=CK)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--probes-a", type=int, default=100)
    ap.add_argument("--probes-b", type=int, default=40)
    ap.add_argument("--chans", default="all",
                    help="'all', 'bad' (guess eps > 5e-2), or an int cap")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()
    dev = args.device

    teacher = make_teacher_cnn((1, 28, 28), CFGS, FC, 10, epochs=25, seed=0,
                               device=dev, act="leaky_relu")
    ck = torch.load(args.ck, map_location=dev, weights_only=False)
    student = ConvNet((1, 28, 28), CFGS, FC, 10, "leaky_relu").to(dev)
    student.load_state_dict(ck["state_dict"])

    t, r = teacher.clone(), student.clone()
    cnn_canonicalize_(t); cnn_canonicalize_(r)
    cnn_align_to_(r, t)

    # sanity: prefix must be exact for the correspondence to mean anything
    for li in (0, 1):
        Dt, Dg = rows(t, li), rows(r, li)
        gn = Dg / Dg.norm(dim=1, keepdim=True).clamp_min(1e-12)
        print(f"[sanity] L{li+1} aligned eps max "
              f"{(Dt - gn).abs().max().item():.2e}")

    Dt = rows(t, FRONTIER).double()
    Dg = rows(r, FRONTIER).double()
    gn = Dg / Dg.norm(dim=1, keepdim=True).clamp_min(1e-12)
    e_as = (Dt - gn).abs().max(1).values
    e_fl = (Dt + gn).abs().max(1).values
    cat = ["good" if a <= 0.05 else ("flip" if f < a else "bad")
           for a, f in zip(e_as, e_fl)]
    print(f"\n[guess] L3 eps: med {e_as.median():.2e} mean {e_as.mean():.2e} "
          f"max {e_as.max():.2e} | cats {Counter(cat)}")

    pre = r.clone().double().to(dev).eval()      # aligned student, exact prefix
    td = t.clone().double().to(dev).eval()       # canonical teacher (same frame)
    rf = (0, 0, 5)                                # L3: 5x5 input, k=5 -> loc 0
    Wg3 = pre.layers[FRONTIER].weight.reshape(120, -1)
    bg3 = pre.layers[FRONTIER].bias
    Wt3 = td.layers[FRONTIER].weight.reshape(120, -1)
    bt3 = td.layers[FRONTIER].bias
    idim = 784

    # ---------------- Phase A ----------------
    P = args.probes_a
    genA = torch.Generator(device=dev).manual_seed(0)
    base = torch.randn(P, idim, generator=genA, device=dev,
                       dtype=torch.float64) * 0.5   # same dist as production
    stats = []
    tA = time.time()
    for c in range(120):
        x0, uh, nu, s_fin = newton_project(pre, Wg3[c], bg3[c], base, rf)
        ok = nu > 1e-12
        with torch.no_grad():
            safe = batched_safe_r(pre, x0, uh, FRONTIER)
            window = torch.minimum(0.9 * safe, torch.ones_like(safe))
        # distance to the TRUE channel-c kink along the ray (teacher-side)
        xg = x0.detach().clone().requires_grad_(True)
        s_t = _cnn_prefix_input(td, xg, FRONTIER, rf) @ Wt3[c] + bt3[c]
        g_t = torch.autograd.grad(s_t.sum(), xg)[0]
        with torch.no_grad():
            gdot = (g_t * uh).sum(1)
            dist = (s_t / gdot.where(gdot.abs() > 1e-30,
                                     torch.full_like(gdot, 1e30))).abs()
            win_ok = window >= 1e-4
            tgt_in = ok & win_ok & (dist <= window)
            # teacher kinks nearer the plane than the target (81-node scan,
            # same resolution as production)
            n_near = torch.zeros(P, device=dev)
            ts = torch.linspace(-1, 1, 81, device=dev, dtype=torch.float64)
            X = (x0[:, None, :] + (ts[None, :, None] * window[:, None, None])
                 * uh[:, None, :]).reshape(P * 81, idim)
            with torch.no_grad():
                g = td(X).view(P, 81, -1)
            kink = ((g[:, 2:] - g[:, 1:-1]) - (g[:, 1:-1] - g[:, :-2])).norm(dim=2)
            tk = ts[1:-1][None, :] * window[:, None]
            is_k = kink > 1e-9
            nearer = is_k & (tk.abs() < dist[:, None].clamp(max=1e29))
            n_near = nearer.sum(1).float()
        stats.append(dict(
            newton_ok=ok.float().mean().item(),
            win_small=(~win_ok).float().mean().item(),
            med_window=window.median().item(),
            med_dist=dist.clamp(max=1e29).median().item(),
            frac_in=tgt_in.float().mean().item(),
            med_nearer=n_near.median().item(),
            plane_res=s_fin.abs().median().item(),
        ))
    print(f"\n[Phase A] {P} probes x 120 ch in {time.time()-tA:.0f}s")
    print(f"{'ch':>3} {'cat':>4} {'g_eps':>8} {'newtOK':>6} {'win<1e-4':>8} "
          f"{'medWin':>8} {'medDist':>8} {'frac_in':>7} {'medNearer':>9}")
    order = sorted(range(120), key=lambda c: -e_as[c])
    for c in order[:15] + ["..."] + order[-10:]:
        if c == "...":
            print("  ...")
            continue
        s = stats[c]
        print(f"{c:>3} {cat[c]:>4} {e_as[c]:8.2e} {s['newton_ok']:6.2f} "
              f"{s['win_small']:8.2f} {s['med_window']:8.2e} "
              f"{s['med_dist']:8.2e} {s['frac_in']:7.2f} {s['med_nearer']:9.1f}")
    fi = torch.tensor([s["frac_in"] for s in stats])
    mw = torch.tensor([s["med_window"] for s in stats])
    md = torch.tensor([s["med_dist"] for s in stats])
    print(f"\n[Phase A summary] med_window: med {mw.median():.2e} | "
          f"med_dist: med {md.median():.2e} | frac_in: med {fi.median():.2f} "
          f"mean {fi.mean():.2f} | ch with frac_in<0.05: "
          f"{int((fi < 0.05).sum())}/120")
    for lo, hi, nm in [(0, 5e-3, "<5e-3"), (5e-3, 2e-2, "<2e-2"),
                       (2e-2, 5e-2, "<5e-2"), (5e-2, 10, ">5e-2")]:
        m = (e_as.cpu() >= lo) & (e_as.cpu() < hi)
        if m.any():
            print(f"  guess eps {nm:>6} (n={int(m.sum()):3d}): "
                  f"frac_in med {fi[m].median():.2f} | "
                  f"med_dist med {md[m].median():.2e} | "
                  f"med_win med {mw[m].median():.2e}")

    if args.skip_b:
        return

    # ---------------- Phase B: instrumented production refiner ----------------
    if args.chans == "all":
        chans = list(range(120))
    elif args.chans == "bad":
        chans = [c for c in range(120) if e_as[c] > 5e-2]
    else:
        chans = list(range(int(args.chans)))
    Din = 400
    print(f"\n[Phase B] instrumented refiner, {len(chans)} ch x "
          f"{args.probes_b} probes")
    orc = _Oracle(td)                            # same canonical frame as pre
    outcomes, tB = [], time.time()
    for c in chans:
        gen = torch.Generator(device=dev).manual_seed(0)   # as production
        wg, bgc = Wg3[c], bg3[c].reshape(())
        cnt = Counter()
        angles_fail, Cblocks, hs = [], [], []
        for _ in range(args.probes_b):
            b0 = torch.randn(idim, generator=gen, device=dev,
                             dtype=torch.float64) * 0.5
            xcur, uh, nu, b_img = b0, None, None, None
            for _ in range(6):
                xg = xcur.detach().clone().requires_grad_(True)
                s0 = (_cnn_prefix_input(pre, xg.unsqueeze(0), FRONTIER, rf)
                      .squeeze(0) * wg).sum() + bgc
                u = torch.autograd.grad(s0, xg)[0]; nu = u.norm()
                if nu < 1e-12:
                    break
                uh = (u / nu).detach()
                b_img = (s0 - u @ xg).reshape(()).detach()
                if abs(float(s0)) < 1e-9:
                    break
                xcur = (xg - (s0 / nu ** 2) * u).detach()
            if uh is None or nu < 1e-12:
                cnt["newton_fail"] += 1
                continue
            with torch.no_grad():
                x0 = xcur
                sr = batched_safe_r(pre, x0.unsqueeze(0), uh.unsqueeze(0),
                                    FRONTIER)[0].item()
                window = min(0.9 * sr, 1.0)
                if window < 1e-4:
                    cnt["window_small"] += 1
                    continue
                out = refine_neuron(orc, (uh * nu).detach(), b_img, x0,
                                    window=window, n_scan=81, n_cand=15,
                                    gen=gen)
                if out is None:
                    cnt["no_kink"] += 1
                    continue
                if out.get("angle_deg") is None or out["angle_deg"] > 2.0:
                    cnt["angle_fail"] += 1
                    if out.get("angle_deg") is not None:
                        angles_fail.append(out["angle_deg"])
                    continue
                cnt["accept"] += 1
                xs = x0 + out["offset"] * uh
                n_k = out["w_refined"].to(dev).double()
                h_k = _cnn_prefix_input(pre, xs.unsqueeze(0), FRONTIER,
                                        rf).squeeze(0)
            A_k = torch.autograd.functional.jacobian(
                lambda z: _cnn_prefix_input(pre, z.unsqueeze(0), FRONTIER,
                                            rf).squeeze(0), xs.detach())
            with torch.no_grad():
                Cblocks.append(A_k.t() - torch.outer(n_k, A_k @ n_k))
                hs.append(h_k)
            if len(Cblocks) >= 3:
                S = torch.linalg.svdvals(torch.cat(Cblocks, 0))
                if S[-1] < 1e-5 * S[-2]:
                    break
        res = dict(c=c, cat=cat[c], g_eps=float(e_as[c]), cnt=cnt,
                   ang=angles_fail)
        if len(Cblocks) < 3:
            res["outcome"] = "few_kinks"
        else:
            C = torch.cat(Cblocks, 0)
            _, S, Vh = torch.linalg.svd(C, full_matrices=False)
            tol = max(1e-9, 1e-4 * S[0].item())
            null_dim = int((S < tol).sum().item())
            res["null_dim"] = null_dim
            if null_dim == 0 or null_dim > Din // 2:
                res["outcome"] = "null_bad"
            else:
                N = Vh[-null_dim:]
                W = (N @ wg) @ N
                W = W / W.norm().clamp_min(1e-12)
                b = -torch.stack([W @ h for h in hs]).mean()
                wb = torch.cat([W, b.reshape(1)])
                wb = wb / wb.norm().clamp_min(1e-12)
                w, b = wb[:-1], wb[-1]
                if (w * wg).sum() + b * bgc < 0:
                    w, b = -w, -b
                sg = torch.cat([wg, bgc.reshape(1)]).norm().clamp_min(1e-12)
                move = max((w - wg / sg).abs().max().item(),
                           (b - bgc / sg).abs().item())
                res["move"] = move
                if move > 0.3:
                    res["outcome"] = "runaway"
                else:
                    res["outcome"] = "solved"
                    wb_ref = torch.cat([w, b.reshape(1)])
                    res["post_eps"] = float((Dt[c] - wb_ref).abs().max())
                    res["post_eps_fl"] = float((Dt[c] + wb_ref).abs().max())
        outcomes.append(res)
        oc = res["outcome"]
        extra = (f" post_eps {res['post_eps']:.1e}" if oc == "solved" else
                 (f" move {res.get('move', 0):.2f}" if oc == "runaway" else ""))
        print(f"  ch {c:3d} [{cat[c]:4s} g_eps {e_as[c]:.2e}] -> {oc:9s} "
              f"{dict(cnt)}{extra}", flush=True)

    print(f"\n[Phase B summary] {time.time()-tB:.0f}s")
    oc_cnt = Counter(r["outcome"] for r in outcomes)
    print(f"  outcomes: {dict(oc_cnt)}")
    gate = Counter()
    for r in outcomes:
        gate.update(r["cnt"])
    tot = sum(gate.values())
    print("  probe-level gates: " +
          "  ".join(f"{k} {v} ({100*v/max(tot,1):.0f}%)"
                    for k, v in gate.most_common()))
    af = [a for r in outcomes for a in r["ang"]]
    if af:
        af = torch.tensor(af)
        print(f"  failed angles: med {af.median():.1f} deg | "
              f"frac in (2,10] deg {(100*((af > 2) & (af <= 10)).float().mean()):.0f}% "
              f"(near-misses = likely target kink, noisy normal)")
    for oc in ("solved", "few_kinks", "runaway", "null_bad"):
        sel = [r for r in outcomes if r["outcome"] == oc]
        if sel:
            ge = torch.tensor([r["g_eps"] for r in sel])
            print(f"  {oc:9s}: n {len(sel):3d} | guess eps med {ge.median():.2e} "
                  f"max {ge.max():.2e}")
    sol = [r for r in outcomes if r["outcome"] == "solved"]
    if sol:
        pe = torch.tensor([r["post_eps"] for r in sol])
        nf = sum(1 for r in sol if r["post_eps_fl"] < 1e-4)
        print(f"  solved post-eps: med {pe.median():.2e} max {pe.max():.2e} | "
              f"locked-flipped {nf}")


if __name__ == "__main__":
    main()
