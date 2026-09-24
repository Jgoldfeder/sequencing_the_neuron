"""FIX VALIDATION for the L3 peel stall. Standalone copy of _cnn_refine_layer's
per-channel loop with two changes, everything else production-identical:

  FIX 1 (jump_ratio gate): reject accepted kinks whose fd normal is dirty
        (out["jump_ratio"] < 1e6). Diagnosis: 1-2 dirty kinks per channel pass
        the 2-deg angle gate and destroy the constraint null space -> the 37
        'null_bad' abstains.
  FIX 2 (negated-guess retry): if a channel abstains from the as-is guess,
        retry from the NEGATED guess. Diagnosis: 28 channels sit in flipped
        minima; the kink PLANE is sign-invariant, so the negated guess is
        plane-close. (Sign itself stays an open, separate question -- reported
        as 'plane-exact, sign ambiguous'.)

Pre-registered predictions (from diag_l3_stall/null on the run-1 checkpoint):
  - null_bad channels (37 there)  -> solved exact
  - flip channels' planes         -> recovered via FIX 2
  - 'bad' channels (both orientations far) -> mostly still abstain
Scoring vs the teacher is sign-aware and DIAGNOSTIC ONLY; the extraction loop
itself uses only the black box + the frozen prefix.

Usage: python diag_l3_fix.py [--ck ...] [--probes 100] [--jr-gate 1e6]
"""
import argparse
import time
from collections import Counter

import torch

from align import cnn_canonicalize_, cnn_align_to_
from data import make_teacher_cnn
from method import _cnn_prefix_input
from nets import ConvNet
from verify_layer1 import _Oracle, refine_neuron
from diag_l3_stall import batched_safe_r, rows, CFGS, FC, FRONTIER, CK


def attempt(pre, orc, wg, bgc, rf, dev, probes, jr_gate, Din=400):
    """One production-style refine attempt for a single channel from guess
    (wg, bgc). Returns (solved, w, b, stats)."""
    gen = torch.Generator(device=dev).manual_seed(0)
    idim = 784
    cnt = Counter()
    Cblocks, hs = [], []
    for _ in range(probes):
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
                                window=window, n_scan=81, n_cand=15, gen=gen)
            if out is None:
                cnt["no_kink"] += 1
                continue
            if out.get("angle_deg") is None or out["angle_deg"] > 2.0:
                cnt["angle_fail"] += 1
                continue
            if out.get("jump_ratio", 0.0) < jr_gate:          # FIX 1
                cnt["dirty_jr"] += 1
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
    if len(Cblocks) < 3:
        return False, None, None, cnt
    C = torch.cat(Cblocks, 0)
    _, S, Vh = torch.linalg.svd(C, full_matrices=False)
    tol = max(1e-9, 1e-4 * S[0].item())
    null_dim = int((S < tol).sum().item())
    if null_dim == 0 or null_dim > Din // 2:
        cnt["null_bad_final"] += 1
        return False, None, None, cnt
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
    if max((w - wg / sg).abs().max().item(),
           (b - bgc / sg).abs().item()) > 0.3:
        cnt["runaway"] += 1
        return False, None, None, cnt
    return True, w, b, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default=CK)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--probes", type=int, default=100)
    ap.add_argument("--jr-gate", type=float, default=1e6)
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
    pre = r.clone().double().to(dev).eval()
    td = t.clone().double().to(dev).eval()
    orc = _Oracle(td)
    rf = (0, 0, 5)
    Wg3 = pre.layers[FRONTIER].weight.reshape(120, -1)
    bg3 = pre.layers[FRONTIER].bias
    Dt = rows(t, FRONTIER).double()
    Dg = rows(r, FRONTIER).double()
    gn = Dg / Dg.norm(dim=1, keepdim=True).clamp_min(1e-12)
    e_as = (Dt - gn).abs().max(1).values
    e_fl = (Dt + gn).abs().max(1).values
    cat = ["good" if a <= 0.05 else ("flip" if f < a else "bad")
           for a, f in zip(e_as, e_fl)]
    print(f"[ck] {args.ck}\n[guess cats] {Counter(cat)} | probes/orient "
          f"{args.probes} | jr gate {args.jr_gate:g}", flush=True)

    res = []
    t0 = time.time()
    for c in range(120):
        solved, w, b, cnt = attempt(pre, orc, Wg3[c], bg3[c].reshape(()),
                                    rf, dev, args.probes, args.jr_gate)
        orient = "as-is"
        if not solved:                                        # FIX 2
            solved, w, b, cnt2 = attempt(pre, orc, -Wg3[c],
                                         (-bg3[c]).reshape(()), rf, dev,
                                         args.probes, args.jr_gate)
            cnt.update(cnt2)
            orient = "negated"
        if solved:
            wb = torch.cat([w, b.reshape(1)])
            pe_as = float((Dt[c] - wb).abs().max())
            pe_fl = float((Dt[c] + wb).abs().max())
            plane = min(pe_as, pe_fl)
            res.append(dict(c=c, cat=cat[c], solved=True, orient=orient,
                            pe_as=pe_as, pe_fl=pe_fl, plane=plane))
            print(f"  ch {c:3d} [{cat[c]:4s}] SOLVED ({orient:7s}) "
                  f"plane_eps {plane:.1e} sign {'OK' if pe_as < pe_fl else 'FLIPPED'}",
                  flush=True)
        else:
            res.append(dict(c=c, cat=cat[c], solved=False, cnt=dict(cnt)))
            print(f"  ch {c:3d} [{cat[c]:4s}] abstain {dict(cnt)}", flush=True)

    ns = sum(r_["solved"] for r_ in res)
    print(f"\n[SUMMARY] {args.ck}")
    print(f"  solved {ns}/120 in {time.time()-t0:.0f}s "
          f"({orc.n/1e6:.1f}M oracle queries)")
    for k in ("good", "flip", "bad"):
        sel = [r_ for r_ in res if r_["cat"] == k]
        s = [r_ for r_ in sel if r_["solved"]]
        pl = torch.tensor([r_["plane"] for r_ in s]) if s else None
        sgn_ok = sum(1 for r_ in s if r_["pe_as"] < r_["pe_fl"])
        print(f"  {k:4s}: {len(s)}/{len(sel)} solved"
              + (f" | plane eps med {pl.median():.1e} max {pl.max():.1e} "
                 f"| sign correct {sgn_ok}/{len(s)}" if s else ""))
    sol = [r_ for r_ in res if r_["solved"]]
    pl = torch.tensor([r_["plane"] for r_ in sol])
    print(f"  ALL solved: plane eps med {pl.median():.1e} max {pl.max():.1e} | "
          f"sign-correct {sum(1 for r_ in sol if r_['pe_as'] < r_['pe_fl'])}"
          f"/{ns} (sign of negated-orient solves is inherited, not inferred)")
    print(f"  unsolved: {[(r_['c'], r_['cat']) for r_ in res if not r_['solved']]}")


if __name__ == "__main__":
    main()
