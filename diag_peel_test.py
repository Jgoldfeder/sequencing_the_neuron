"""Dry-run the --resume peel's refine stage on the cheat LeNet checkpoint:
kink-refine L1, L2, C5 from the saved student (exact prefix cascaded like the
real resume path) and score every channel against the teacher SIGN-AWARE.

Answers, per C5 category (good / sign-flipped / other-bad):
  - does the channel solve exactly, abstain, or -- the dangerous case -- get
    "solved" in the FLIPPED orientation (guess-inherited sign + runaway gate
    0.3 > flip plane error ~0.19)?
  - would C5 reach freeze-thresh 0.9, and with how many poisoned channels?
"""
import time

import torch

from align import cnn_canonicalize_, cnn_align_to_
from data import make_teacher_cnn
from method import _cnn_refine_layer
from nets import ConvNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFGS = [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)]
CK = ("recon/mergedbest_cnn_cheat__1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_"
      "16-120-5-1-0-0_fc84__s0_final.pt")


def rows(net, li):
    W, b = net.layers[li].weight, net.layers[li].bias
    return torch.cat([W.view(W.shape[0], -1), b[:, None]], 1)


def main():
    teacher = make_teacher_cnn((1, 28, 28), CFGS, (84,), 10, epochs=25, seed=0,
                               device=DEV, act="leaky_relu")
    ck = torch.load(CK, map_location=DEV, weights_only=False)
    student = ConvNet((1, 28, 28), CFGS, (84,), 10, "leaky_relu").to(DEV)
    student.load_state_dict(ck["state_dict"])

    # student in teacher-aligned canonical frame (function-preserving), so
    # channel c scores directly against teacher channel c, sign-aware
    t, r = teacher.clone(), student.clone()
    cnn_canonicalize_(t); cnn_canonicalize_(r)
    cnn_align_to_(r, t)

    exact = r.clone()
    for li, nm in [(0, "L1"), (1, "L2"), (2, "C5")]:
        Dt = rows(t, li)
        Dg = rows(r, li)
        gn = Dg / Dg.norm(dim=1, keepdim=True).clamp_min(1e-12)
        e_as = (Dt - gn).abs().max(1).values          # guess eps as-is
        e_fl = (Dt + gn).abs().max(1).values          # guess eps if negated
        cat = ["good" if a <= 0.05 else ("flip" if f < a else "bad")
               for a, f in zip(e_as, e_fl)]

        t0 = time.time()
        Wr, br, mask = _cnn_refine_layer(teacher, exact, li, (1, 28, 28),
                                         DEV, "leaky_relu")
        dt = time.time() - t0
        # write solved rows into the cascading prefix (like resume)
        idx = mask.to(DEV).nonzero(as_tuple=True)[0]
        wdt = exact.layers[li].weight.dtype
        if len(idx):
            exact.layers[li].weight.data[idx] = Wr[idx].to(wdt)
            exact.layers[li].bias.data[idx] = br[idx].to(wdt)

        Dref = rows(exact, li)
        rn = Dref / Dref.norm(dim=1, keepdim=True).clamp_min(1e-12)
        p_as = (Dt - rn).abs().max(1).values          # post-refine eps
        p_fl = (Dt + rn).abs().max(1).values
        n = len(e_as)
        print(f"\n== {nm}: {int(mask.sum())}/{n} solved "
              f"({dt:.0f}s) -> freeze@0.9 {'YES' if mask.float().mean() >= 0.9 else 'no'} ==")
        stats = {}
        for c in range(n):
            k = cat[c]
            if not mask[c]:
                stats.setdefault(k, []).append("abstain")
            elif p_as[c] < 1e-4:
                stats.setdefault(k, []).append("EXACT")
            elif p_fl[c] < 1e-4:
                stats.setdefault(k, []).append("LOCKED-FLIPPED")
            elif p_as[c] < min(0.5 * e_as[c], 0.02):
                stats.setdefault(k, []).append("improved")
            else:
                stats.setdefault(k, []).append("solved-inexact")
        for k in ("good", "flip", "bad"):
            if k not in stats:
                continue
            from collections import Counter
            cn = Counter(stats[k])
            print(f"  {k:5s} ({len(stats[k]):3d}): " +
                  "  ".join(f"{v} {c}" for v, c in sorted(cn.items())))
        solved = mask.to(DEV)
        if solved.any():
            print(f"  solved rows: eps med {p_as[solved].median():.2e} "
                  f"max {p_as[solved].max():.2e} | locked-flipped: "
                  f"{int(((p_fl < 1e-4) & solved).sum())}")


if __name__ == "__main__":
    main()
