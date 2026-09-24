"""Why does the COMMITTEE/GD phase drive L1/L2 into peel range (~1e-3) but
leave L3 stuck at ~2e-2 mean / 0.4 max with sign flips?

E1  Kink-crossing leverage per layer (teacher, base query dist N(0,0.5^2)):
    a conv unit is identified by queries that cross its kink; L1 has 784 tied
    positions per query, L2 100, L3 just 1. Reports per-channel minority-sign
    mass and expected crossing events per query.
E2  Flip compensation: for sign-flipped L3 channels, the student's L4 column
    should be ~ -v_teacher/alpha (mostly-positive units) or ~ -alpha*v_teacher
    (mostly-negative): cos ~ -1 with a ~100x or ~0.01x norm ratio -> the flip
    is (near) loss-invisible, i.e. a stable spurious minimum at p=1.
E3  Co-adaptation/blame: MSE of (a) student, (b) TRUE L3 under student L4/L5,
    (c) student L3 under TRUE L4/L5, (d) same as (c) with flipped rows negated.
    If (b) >= (a), GD has no local incentive to move L3 toward the truth.
E4  Oracle isolation: freeze TRUE L4/L5, train ONLY L3 from the checkpoint
    guess. If good channels crash to ~0 and flips persist, the blockers are
    (i) wrong downstream filtering the signal and (ii) the flip barrier.

Usage: python diag_l3_gd.py [--ck ...] [--steps 3000]
"""
import argparse
from collections import Counter

import torch
import torch.nn.functional as F

from align import cnn_canonicalize_, cnn_align_to_
from data import make_teacher_cnn
from nets import ConvNet

CFGS = [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)]
FC = (84,)
CK = ("recon/mergedbest_cnn_cheat__1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_"
      "16-120-5-1-0-0_fc84__s0_final.pt")
ALPHA = 0.01


def rows(net, li):
    W, b = net.layers[li].weight, net.layers[li].bias
    return torch.cat([W.view(W.shape[0], -1), b[:, None]], 1)


def preacts_conv(net, X, li):
    """Pre-activations of conv layer li: (N, C, H, W)."""
    x = X.view(X.shape[0], *net.input_shape)
    with torch.no_grad():
        for i in range(li):
            x = net.act(net.layers[i](x))
            if net.pools[i] > 0:
                x = F.avg_pool2d(x, net.pools[i])
        return net.layers[li](x)


def l3_eps(net, Dt):
    D = rows(net, 2).double()
    gn = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
    e_as = (Dt - gn).abs().max(1).values
    e_fl = (Dt + gn).abs().max(1).values
    return e_as, e_fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default=CK)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=3000)
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

    Dt = rows(t, 2).double()
    e_as, e_fl = l3_eps(r, Dt)
    cat = ["good" if a <= 0.05 else ("flip" if f < a else "bad")
           for a, f in zip(e_as, e_fl)]
    print(f"[cats] {Counter(cat)}")

    gen = torch.Generator(device=dev).manual_seed(0)
    X = torch.randn(args.n, 784, generator=gen, device=dev) * 0.5

    # ---------------- E1: crossing leverage per layer ----------------
    print("\n== E1: kink-crossing leverage (teacher, N(0,0.5^2) queries) ==")
    for li, nm in [(0, "L1"), (1, "L2"), (2, "L3")]:
        z = preacts_conv(t, X, li)                    # (N, C, H, W)
        N, C, H, W = z.shape
        zf = z.permute(1, 0, 2, 3).reshape(C, -1)     # (C, N*H*W)
        pos = (zf > 0).float().mean(1)                # P(z>0) per channel
        bal = torch.minimum(pos, 1 - pos)             # minority mass
        maj = (pos > 0.5).float() * 2 - 1
        minority = ((z > 0).float().permute(1, 0, 2, 3).reshape(C, N, -1)
                    != (maj[:, None, None] > 0)).float()
        ev = minority.sum(2).mean(1)                  # minority positions/query
        print(f"  {nm}: positions {H*W:4d} | minority mass per position: "
              f"med {bal.median():.3f} min {bal.min():.3f} | "
              f"crossing events/query: med {ev.median():.1f} min {ev.min():.2f} "
              f"| ch with <0.1 events/query: {int((ev < 0.1).sum())}/{C}")
        if li == 2:
            b3, e3 = bal.cpu(), ev.cpu()
            for k in ("good", "flip", "bad"):
                m = torch.tensor([c == k for c in cat])
                if m.any():
                    print(f"      {k:4s} (n={int(m.sum()):3d}): minority mass "
                          f"med {b3[m].median():.3f} | events/query med "
                          f"{e3[m].median():.2f}")

    # ---------------- E2: flip compensation in L4 ----------------
    print("\n== E2: L4 column compensation (aligned frames, alpha=0.01) ==")
    Vt = t.layers[3].weight.detach()                  # (84, 120)
    Vs = r.layers[3].weight.detach()
    ratio = Vs.norm(dim=0) / Vt.norm(dim=0).clamp_min(1e-12)
    cosc = F.cosine_similarity(Vs, Vt, dim=0)
    z3 = preacts_conv(t, X, 2).squeeze(-1).squeeze(-1)  # (N, 120)
    p_pos = (z3 > 0).float().mean(0)
    for k in ("good", "flip", "bad"):
        m = torch.tensor([c == k for c in cat], device=dev)
        if m.any():
            print(f"  {k:4s}: |v_s|/|v_t| med {ratio[m].median():.2f} "
                  f"[p10 {ratio[m].quantile(0.1):.2f} p90 {ratio[m].quantile(0.9):.2f}] | "
                  f"cos(v_s,v_t) med {cosc[m].median():+.2f} | "
                  f"teacher P(z>0) med {p_pos[m].median():.2f}")
    fl = torch.tensor([c == "flip" for c in cat], device=dev)
    if fl.any():
        pred = torch.where(p_pos[fl] > 0.5, 1 / ALPHA,
                           torch.full_like(p_pos[fl], ALPHA))
        print(f"  flip norm-ratio vs predicted (-v/alpha or -alpha*v): "
              f"log10 ratio med {ratio[fl].log10().median():.2f} vs predicted "
              f"med {pred.log10().median():.2f}")

    # ---------------- E3: blame / co-adaptation ----------------
    print("\n== E3: MSE blame decomposition (20k probes) ==")
    with torch.no_grad():
        Y = t(X)

        def mse(net):
            return F.mse_loss(net(X), Y).item()

        h = r.clone()                                  # student as-is
        print(f"  student (aligned)                      : {mse(h):.4e}")
        h = r.clone()
        h.layers[2].weight.data.copy_(t.layers[2].weight)
        h.layers[2].bias.data.copy_(t.layers[2].bias)
        print(f"  TRUE L3 under student L4/L5            : {mse(h):.4e}")
        h = t.clone()
        h.layers[2].weight.data.copy_(r.layers[2].weight)
        h.layers[2].bias.data.copy_(r.layers[2].bias)
        print(f"  student L3 under TRUE L4/L5            : {mse(h):.4e}")
        h2 = h.clone()
        idx = torch.tensor([i for i, c in enumerate(cat) if c == "flip"],
                           device=dev, dtype=torch.long)
        if len(idx):
            h2.layers[2].weight.data[idx] *= -1
            h2.layers[2].bias.data[idx] *= -1
            print(f"  ... same, flipped rows negated         : {mse(h2):.4e}")

    # ---------------- E4: oracle isolation training ----------------
    print(f"\n== E4: train ONLY L3, TRUE L4/L5 frozen, {args.steps} steps ==")
    h = r.clone()
    for li in (3, 4):
        h.layers[li].weight.data.copy_(t.layers[li].weight)
        h.layers[li].bias.data.copy_(t.layers[li].bias)
    for i, l in enumerate(h.layers):
        l.weight.requires_grad_(i == 2)
        l.bias.requires_grad_(i == 2)
    opt = torch.optim.Adam([h.layers[2].weight, h.layers[2].bias], lr=1e-3)
    marks = {0, args.steps // 6, args.steps // 3, args.steps // 2,
             2 * args.steps // 3, args.steps}
    for step in range(args.steps + 1):
        if step in marks:
            ea, ef = l3_eps(h, Dt)
            ncat = Counter("good" if a <= 0.05 else ("flip" if f < a else "bad")
                           for a, f in zip(ea, ef))
            print(f"  step {step:5d}: eps med {ea.median():.2e} "
                  f"max {ea.max():.2e} | <1e-3: {int((ea < 1e-3).sum())}/120 | "
                  f"cats {dict(ncat)}", flush=True)
        if step == args.steps:
            break
        Xb = torch.randn(8192, 784, device=dev) * 0.5
        with torch.no_grad():
            Yb = t(Xb)
        opt.zero_grad()
        F.mse_loss(h(Xb), Yb).backward()
        opt.step()

    # same isolation but under the student's own (wrong) L4/L5, for contrast
    print(f"\n== E4b: train ONLY L3 under student's WRONG L4/L5 (frozen) ==")
    h = r.clone()
    for i, l in enumerate(h.layers):
        l.weight.requires_grad_(i == 2)
        l.bias.requires_grad_(i == 2)
    opt = torch.optim.Adam([h.layers[2].weight, h.layers[2].bias], lr=1e-3)
    for step in range(args.steps + 1):
        if step in marks:
            ea, ef = l3_eps(h, Dt)
            print(f"  step {step:5d}: eps med {ea.median():.2e} "
                  f"max {ea.max():.2e} | <1e-3: {int((ea < 1e-3).sum())}/120",
                  flush=True)
        if step == args.steps:
            break
        Xb = torch.randn(8192, 784, device=dev) * 0.5
        with torch.no_grad():
            Yb = t(Xb)
        opt.zero_grad()
        F.mse_loss(h(Xb), Yb).backward()
        opt.step()


if __name__ == "__main__":
    main()
