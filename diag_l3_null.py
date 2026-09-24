"""Post-mortem of the null_bad abstains at L3: collect accepted kinks exactly
like _cnn_refine_layer, then inspect the constraint spectrum -- is the smallest
singular vector the TRUE row (gate too strict) or corrupted (noisy normals)?

For each channel: print S tail, forced null_dim=1 recovery eps (as-is/flip),
and the same with only high-jump-ratio kinks kept.

Usage: python diag_l3_null.py --chans 0,5,6,8 [--probes 40]
"""
import argparse
from collections import Counter

import torch

from align import cnn_canonicalize_, cnn_align_to_
from data import make_teacher_cnn
from method import _cnn_prefix_input
from nets import ConvNet
from verify_layer1 import _Oracle, refine_neuron
from diag_l3_stall import batched_safe_r, rows, CFGS, FC, FRONTIER, CK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default=CK)
    ap.add_argument("--chans", default="0,5,6,8")
    ap.add_argument("--probes", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    idim = 784

    for c in [int(x) for x in args.chans.split(",")]:
        gen = torch.Generator(device=dev).manual_seed(0)
        wg, bgc = Wg3[c], bg3[c].reshape(())
        Cblocks, hs, jrs, angs = [], [], [], []
        for _ in range(args.probes):
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
                continue
            with torch.no_grad():
                x0 = xcur
                sr = batched_safe_r(pre, x0.unsqueeze(0), uh.unsqueeze(0),
                                    FRONTIER)[0].item()
                window = min(0.9 * sr, 1.0)
                if window < 1e-4:
                    continue
                out = refine_neuron(orc, (uh * nu).detach(), b_img, x0,
                                    window=window, n_scan=81, n_cand=15,
                                    gen=gen)
                if (out is None or out.get("angle_deg") is None
                        or out["angle_deg"] > 2.0):
                    continue
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
                jrs.append(out.get("jump_ratio", float("nan")))
                angs.append(out["angle_deg"])
        print(f"\n== ch {c}: {len(Cblocks)} accepted kinks | "
              f"jump_ratio {['%.1e' % j for j in jrs]} | "
              f"angle {['%.2e' % a for a in angs]}")
        if len(Cblocks) < 3:
            print("   too few kinks")
            continue

        def solve(blocks, pts, tag):
            C = torch.cat(blocks, 0)
            _, S, Vh = torch.linalg.svd(C, full_matrices=False)
            tol = max(1e-9, 1e-4 * S[0].item())
            nd = int((S < tol).sum().item())
            W = Vh[-1]
            W = (W / W.norm()) * torch.sign((W * wg).sum() + 0.0)
            b = -torch.stack([W @ h for h in pts]).mean()
            wb = torch.cat([W, b.reshape(1)])
            wb = wb / wb.norm()
            if (wb[:-1] * wg).sum() + wb[-1] * bgc < 0:
                wb = -wb
            e_as = float((Dt[c] - wb).abs().max())
            e_fl = float((Dt[c] + wb).abs().max())
            print(f"   [{tag}] S[0] {S[0]:.2e} S[-3:] "
                  f"{[float('%.2e' % v) for v in S[-3:]]} "
                  f"S[-1]/S[-2] {S[-1]/S[-2]:.2e} | prod null_dim {nd} | "
                  f"forced-1 eps as {e_as:.2e} fl {e_fl:.2e}")

        solve(Cblocks, hs, "all kinks")
        good = [i for i, j in enumerate(jrs) if j > 1e6]
        if len(good) >= 3 and len(good) < len(Cblocks):
            solve([Cblocks[i] for i in good], [hs[i] for i in good],
                  f"jump_ratio>1e6 ({len(good)})")


if __name__ == "__main__":
    main()
