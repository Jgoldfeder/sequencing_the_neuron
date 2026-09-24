"""Why does the cheat LeNet run recover L1/L2 but stall on C5/F6/head?

Loads the cached LeNet teacher + the saved cheat student, aligns them, and
tests the candidate explanations:
  A) localization: per-unit eps per layer (is the stall a few units or all?)
  B) excitation: per-channel kink-crossing balance min(P(z>0),P(z<0)) under
     (i) the actual mined cheat-query distribution, (ii) uniform init dist,
     (iii) MNIST -- a channel that never crosses its kink is linear ->
     unidentifiable (leaky_relu)
  C) influence: downstream column norm in the canonical frame (a channel the
     head barely reads is invisible at the current loss floor)
  D) duplicates: bad student units whose best-cosine teacher match is a
     channel already claimed by another student unit
  E) substitution: hybrid teacher-prefix/student-suffix nets localize the
     FUNCTION error by layer; row-patch test shows whether fixing the worst
     k units alone snaps the fit

Usage: python diag_deep_stall.py [--ck recon/..._final.pt] [--device cuda]
"""
import argparse

import torch

from align import cnn_canonicalize_, cnn_align_to_
from data import make_teacher_cnn, load_data
from method import Cfg, gen_queries
from nets import ConvNet

INPUT_SHAPE = (1, 28, 28)
CONV_CFGS = [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)]
FC_DIMS = (84,)
OUT_DIM = 10


def preacts(net, X, li):
    """Pre-activation of hidden layer li, flattened to (N, units)."""
    x = X.view(X.shape[0], *net.input_shape)
    with torch.no_grad():
        for i in range(len(net.layers) - 1):
            z = net.layers[i](x)
            if i == li:
                return z.flatten(1) if z.dim() > 2 else z
            x = net.act(z)
            if i < net.n_conv and net.pools[i] > 0:
                x = torch.nn.functional.avg_pool2d(x, net.pools[i])
            if i == net.n_conv - 1:
                x = torch.flatten(x, 1)
    raise ValueError(li)


def unit_eps(t, r, li):
    """Per-unit inf-norm eps of [W|b] rows at layer li (canonical+aligned)."""
    Wt, bt = t.layers[li].weight, t.layers[li].bias
    Wr, br = r.layers[li].weight, r.layers[li].bias
    Dt = torch.cat([Wt.view(Wt.shape[0], -1), bt[:, None]], 1)
    Dr = torch.cat([Wr.view(Wr.shape[0], -1), br[:, None]], 1)
    return (Dt - Dr).abs().max(1).values


def l1(net, ref, X, bs=8192):
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs]
            tot += (net(xb) - ref(xb)).abs().mean(1).sum().item()
            n += len(xb)
    return tot / n


def hybrid(t, r, k_teacher):
    """Teacher layers 0..k_teacher-1 + student layers k_teacher.. (canonical
    aligned frame, so channels line up)."""
    h = r.clone()
    for i in range(k_teacher):
        h.layers[i].weight.data.copy_(t.layers[i].weight)
        h.layers[i].bias.data.copy_(t.layers[i].bias)
    return h


def pearson(a, b):
    a, b = a.float(), b.float()
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return (a @ b / d).item() if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default="recon/mergedbest_cnn_cheat__1x28x28__"
                    "1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__s0_final.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--q", type=int, default=8192)
    args = ap.parse_args()
    dev = args.device

    teacher = make_teacher_cnn(INPUT_SHAPE, CONV_CFGS, FC_DIMS, OUT_DIM,
                               epochs=25, seed=0, device=dev, act="leaky_relu")
    ck = torch.load(args.ck, map_location=dev, weights_only=False)
    student = ConvNet(INPUT_SHAPE, CONV_CFGS, FC_DIMS, OUT_DIM, "leaky_relu").to(dev)
    student.load_state_dict(ck["state_dict"])
    teacher.eval(); student.eval()

    # --- input distributions ---------------------------------------------
    gen = torch.Generator(device=dev).manual_seed(0)
    X_uni = (torch.rand(args.q, 784, generator=gen, device=dev) * 2 - 1)
    (_, _), (xte, _) = load_data([784, OUT_DIM], dev)
    X_mnist = xte[:args.q]
    cfg = Cfg(p=1, q=args.q, qg_steps=30, qg_lr=0.1, qg_dist="l1",
              qg_init="uniform", qg_range=1.0, disagree="median_pair")
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    X_mined = gen_queries([student, teacher], cfg, 784, dev, gen)
    dists = [("mined", X_mined), ("uniform", X_uni), ("mnist", X_mnist)]

    print("== function L1(student, teacher) per distribution ==")
    for name, X in dists:
        print(f"  {name:8s} {l1(student, teacher, X):.4e}")

    # --- canonical aligned frames ----------------------------------------
    t, r = teacher.clone(), student.clone()
    cnn_canonicalize_(t); cnn_canonicalize_(r)
    cnn_align_to_(r, t)

    print("\n== per-unit eps (inf-norm of [W|b] row diff, canonical) ==")
    names = ["L1 C1(6)", "L2 C3(16)", "L3 C5(120)", "L4 F6(84)"]
    for li, nm in enumerate(names):
        e = unit_eps(t, r, li)
        print(f"  {nm:11s} max {e.max():.2e} med {e.median():.2e} | "
              f">0.1: {(e > 0.1).sum().item():3d}  >0.01: {(e > 0.01).sum().item():3d}"
              f"  /{len(e)}")
    eh = (torch.cat([t.layers[-1].weight, t.layers[-1].bias[:, None]], 1)
          - torch.cat([r.layers[-1].weight, r.layers[-1].bias[:, None]], 1)).abs()
    print(f"  head        max {eh.max():.2e} med {eh.median():.2e}")
    print(f"  head |W| norms: teacher {t.layers[-1].weight.norm():.2f}  "
          f"student {r.layers[-1].weight.norm():.2f}")

    # --- per-channel excitation + influence at L3/L4 ----------------------
    for li, nm, dn in [(2, "L3 C5", 3), (3, "L4 F6", 4)]:
        e = unit_eps(t, r, li)
        Wn = t.layers[dn].weight
        col = (Wn.view(Wn.shape[0], e.shape[0], -1).norm(dim=(0, 2))
               if Wn.shape[1] != e.shape[0] else Wn.norm(dim=0))
        bal = {}
        for name, X in dists:
            z = preacts(t, X, li)
            pos = (z > 0).float().mean(0)
            bal[name] = torch.minimum(pos, 1 - pos)
        print(f"\n== {nm}: worst units (eps | kink balance mined/uni/mnist | "
              f"downstream colnorm) ==")
        order = e.argsort(descending=True)
        for i in order[:12]:
            print(f"  u{i.item():3d}  eps {e[i]:.2e} | "
                  f"{bal['mined'][i]:.3f} {bal['uniform'][i]:.3f} "
                  f"{bal['mnist'][i]:.3f} | col {col[i]:.3f}")
        good = order[-3:]
        for i in good:
            print(f"  u{i.item():3d}  eps {e[i]:.2e} | "
                  f"{bal['mined'][i]:.3f} {bal['uniform'][i]:.3f} "
                  f"{bal['mnist'][i]:.3f} | col {col[i]:.3f}   (best)")
        print(f"  corr(eps, mined balance) {pearson(e, bal['mined']):+.3f}   "
              f"corr(eps, colnorm) {pearson(e, col):+.3f}")

        # duplicate check: bad units' best-cosine teacher row
        Dt = torch.cat([t.layers[li].weight.view(e.shape[0], -1),
                        t.layers[li].bias[:, None]], 1)
        Dr = torch.cat([r.layers[li].weight.view(e.shape[0], -1),
                        r.layers[li].bias[:, None]], 1)
        C = torch.nn.functional.normalize(Dr, dim=1) @ \
            torch.nn.functional.normalize(Dt, dim=1).T
        bad = order[:8]
        print("  dup check (bad unit -> best-cos teacher unit):")
        for i in bad:
            j = C[i].argmax()
            tag = "self" if j == i else f"-> u{j.item()} (TAKEN)"
            print(f"    u{i.item():3d} cos_self {C[i, i]:+.3f}  best {C[i, j]:+.3f} {tag}")

    # --- substitution: localize the FUNCTION error by layer ---------------
    print("\n== hybrid teacher-prefix + student-suffix: L1 vs teacher (mined) ==")
    print(f"  full student            {l1(r, t, X_mined):.4e}")
    for k in range(1, 5):
        print(f"  teacher L1..L{k} + rest   {l1(hybrid(t, r, k), t, X_mined):.4e}")
    hs = t.clone()   # inverse: student HEAD only on teacher body
    hs.layers[-1].weight.data.copy_(r.layers[-1].weight)
    hs.layers[-1].bias.data.copy_(r.layers[-1].bias)
    print(f"  teacher body + stu head {l1(hs, t, X_mined):.4e}")

    print("\n== row-patch: teacher rows into student's worst-k units ==")
    for li, nm in [(2, "L3"), (3, "L4")]:
        e = unit_eps(t, r, li)
        for k in (5, 10, 20):
            rp = r.clone()
            idx = e.argsort(descending=True)[:k]
            rp.layers[li].weight.data[idx] = t.layers[li].weight.data[idx]
            rp.layers[li].bias.data[idx] = t.layers[li].bias.data[idx]
            print(f"  {nm} worst {k:2d} patched -> {l1(rp, t, X_mined):.4e}")


if __name__ == "__main__":
    main()
