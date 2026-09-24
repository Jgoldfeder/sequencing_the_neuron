"""SGD-side fix test: STRADDLE MINING. The diagnosis says GD starves because
~30 L3 units almost never cross their kink under realizable queries (and the
existing mining targets |z|<0.1 sigma -- ON the kink, where value-signal is
smallest). Fix candidate: synthesize images (attacker-side: optimize x through
the frozen prefix against the student's OWN guessed planes) that put each
unit's preact at +/- (0.1..1) sigma -- both sides, finite margin -- and mix
them into training.

Arms (both warm-start from the checkpoint, train L3+L4+L5, frozen L1/L2):
  control : batches of random images only
  straddle: half random, half straddle-pool images (pool refreshed from the
            current guess every 4k steps)
After training, run the FIXED refiner (jr gate + negated retry) on both arms'
final L3 guesses and count solved channels vs the 82/120 baseline.

Scoring vs teacher is diagnostic only. Usage:
  python diag_l3_sgdfix.py --arm straddle|control [--steps 12000] [--ck ...]
"""
import argparse
import time
from collections import Counter

import torch
import torch.nn.functional as F

from align import cnn_canonicalize_, cnn_align_to_
from data import make_teacher_cnn
from method import _cnn_prefix_input
from nets import ConvNet
from verify_layer1 import _Oracle
from diag_l3_stall import rows, CFGS, FC, FRONTIER, CK
from diag_l3_fix import attempt

RF = (0, 0, 5)


def synth_straddle(pre_ref, W3g, b3g, n_per, dev, gen, steps=50, lr=0.05):
    """Attacker-side pool: images whose unit-c preact (student guess, frozen
    exact prefix) sits at side * m * sd_c, m ~ U(0.1, 1). Round-robin over
    (unit, side)."""
    with torch.no_grad():
        Xp = torch.randn(4096, 784, generator=gen, device=dev) * 0.5
        z = _cnn_prefix_input(pre_ref, Xp, FRONTIER, RF) @ W3g.T + b3g
        sd = z.std(0).clamp_min(1e-6)
    P = 120 * 2 * n_per
    c = torch.arange(P, device=dev) % 120
    side = torch.where((torch.arange(P, device=dev) // 120) % 2 == 0, 1.0, -1.0)
    m = 0.1 + 0.9 * torch.rand(P, generator=gen, device=dev)
    tgt = side * m * sd[c]
    x = (torch.randn(P, 784, generator=gen, device=dev) * 0.5).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        zc = (_cnn_prefix_input(pre_ref, x, FRONTIER, RF)
              @ W3g.T + b3g).gather(1, c[:, None]).squeeze(1)
        ((zc - tgt) ** 2).mean().backward()
        opt.step()
    with torch.no_grad():
        zc = (_cnn_prefix_input(pre_ref, x, FRONTIER, RF)
              @ W3g.T + b3g).gather(1, c[:, None]).squeeze(1)
        hit = ((zc - tgt).abs() < 0.25 * sd[c]).float()
        per_unit = torch.zeros(120, device=dev).scatter_add_(0, c, hit)
    print(f"  [pool] {P} straddle imgs, target hit-rate "
          f"{hit.mean():.2f}, units with <10 hits: "
          f"{int((per_unit < 10).sum())}/120", flush=True)
    return x.detach()


def cats_eps(net, Dt):
    D = rows(net, FRONTIER).double()
    gn = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ea = (Dt - gn).abs().max(1).values
    ef = (Dt + gn).abs().max(1).values
    cat = Counter("good" if a <= 0.05 else ("flip" if f < a else "bad")
                  for a, f in zip(ea, ef))
    return ea, dict(cat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default=CK)
    ap.add_argument("--arm", choices=("straddle", "control"), required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--probes", type=int, default=100)
    args = ap.parse_args()
    dev = args.device

    teacher = make_teacher_cnn((1, 28, 28), CFGS, FC, 10, epochs=25, seed=0,
                               device=dev, act="leaky_relu")
    ck = torch.load(args.ck, map_location=dev, weights_only=False)
    student = ConvNet((1, 28, 28), CFGS, FC, 10, "leaky_relu").to(dev)
    student.load_state_dict(ck["state_dict"])
    t, r = teacher.clone(), student.clone()
    cnn_canonicalize_(t); cnn_canonicalize_(r)
    cnn_align_to_(r, t)                       # frame only; function-preserving
    Dt = rows(t, FRONTIER).double()
    ea0, cat0 = cats_eps(r, Dt)
    print(f"[{args.arm}] start: eps med {ea0.median():.2e} | {cat0}", flush=True)

    # trainable: L3, L4, L5; frozen: exact L1/L2
    for i, l in enumerate(r.layers):
        req = i >= FRONTIER
        l.weight.requires_grad_(req); l.bias.requires_grad_(req)
    opt = torch.optim.Adam([p for p in r.parameters() if p.requires_grad],
                           lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(0)
    pool = None
    t0 = time.time()
    for step in range(args.steps + 1):
        if args.arm == "straddle" and step % 4000 == 0 and step < args.steps:
            with torch.no_grad():
                W3g = r.layers[FRONTIER].weight.reshape(120, -1).detach().clone()
                b3g = r.layers[FRONTIER].bias.detach().clone()
            pool = synth_straddle(r, W3g, b3g, n_per=60, dev=dev, gen=gen)
        if step % 3000 == 0:
            ea, cat = cats_eps(r, Dt)
            print(f"  step {step:6d}: eps med {ea.median():.2e} "
                  f"max {ea.max():.2e} | {cat} | {time.time()-t0:.0f}s",
                  flush=True)
        if step == args.steps:
            break
        xr = torch.randn(2048 if pool is not None else 4096, 784,
                         generator=gen, device=dev) * 0.5
        if pool is not None:
            idx = torch.randint(0, len(pool), (2048,), generator=gen, device=dev)
            x = torch.cat([xr, pool[idx]])
        else:
            x = xr
        with torch.no_grad():
            y = t(x)                          # black-box image queries only
        opt.zero_grad()
        F.mse_loss(r(x), y).backward()
        opt.step()

    # ---- fixed refiner on the trained guesses ----
    print(f"\n[{args.arm}] refiner pass (jr gate + negated retry)", flush=True)
    pre = r.clone().double().to(dev).eval()
    orc = _Oracle(t.clone().double().to(dev).eval())
    Wg3 = pre.layers[FRONTIER].weight.reshape(120, -1)
    bg3 = pre.layers[FRONTIER].bias
    solved, sign_ok, planes = 0, 0, []
    unsolved = []
    for c in range(120):
        ok, w, b, cnt = attempt(pre, orc, Wg3[c], bg3[c].reshape(()),
                                RF, dev, args.probes, 1e6)
        if not ok:
            ok, w, b, _ = attempt(pre, orc, -Wg3[c], (-bg3[c]).reshape(()),
                                  RF, dev, args.probes, 1e6)
        if ok:
            wb = torch.cat([w, b.reshape(1)])
            pa = float((Dt[c] - wb).abs().max())
            pf = float((Dt[c] + wb).abs().max())
            solved += 1; planes.append(min(pa, pf)); sign_ok += pa < pf
        else:
            unsolved.append(c)
    pl = torch.tensor(planes)
    print(f"[{args.arm} FINAL] refiner solved {solved}/120 "
          f"(sign-correct {sign_ok}) | plane eps med {pl.median():.1e} "
          f"max {pl.max():.1e}\n  unsolved: {unsolved}", flush=True)


if __name__ == "__main__":
    main()
