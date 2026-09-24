"""Alternation test: refine -> FREEZE solved rows exactly -> train only the
unsolved rows (+L4/L5) -> refine again. The straddle-arm failure showed why
retraining everything fails: while flip/bad rows are trainable, SGD recruits
the good rows into compensation. Anchoring the solved rows removes that escape.

Diagnostic shortcut, acknowledged: the frozen set = sign-correct solves (the
attacker-legal version needs a crossing-stats classifier for sign-suspects).
Plane-solved-but-sign-suspect channels are initialized at their exact plane
(inherited sign) but stay trainable.

Arms: --queries straddle (half straddle on unsolved channels) | random.
Usage: python diag_l3_alt.py --queries straddle [--steps 8000]
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
from diag_l3_sgdfix import synth_straddle, cats_eps

RF = (0, 0, 5)


def refine_all(pre, orc, chans, Dt, dev, probes):
    Wg3 = pre.layers[FRONTIER].weight.reshape(120, -1)
    bg3 = pre.layers[FRONTIER].bias
    out = {}
    for c in chans:
        ok, w, b, _ = attempt(pre, orc, Wg3[c], bg3[c].reshape(()), RF, dev,
                              probes, 1e6)
        orient = "as-is"
        if not ok:
            ok, w, b, _ = attempt(pre, orc, -Wg3[c], (-bg3[c]).reshape(()),
                                  RF, dev, probes, 1e6)
            orient = "negated"
        if ok:
            wb = torch.cat([w, b.reshape(1)])
            pa = float((Dt[c] - wb).abs().max())
            pf = float((Dt[c] + wb).abs().max())
            out[c] = dict(w=w, b=b, pe_as=pa, pe_fl=pf, orient=orient,
                          sign_ok=pa < pf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default=CK)
    ap.add_argument("--queries", choices=("straddle", "random"), required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--probes", type=int, default=100)
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
    Dt = rows(t, FRONTIER).double()
    td = t.clone().double().to(dev).eval()

    # ---- pass 1: refine everything ----
    t0 = time.time()
    pre = r.clone().double().to(dev).eval()
    sol1 = refine_all(pre, _Oracle(td), range(120), Dt, dev, args.probes)
    frozen = [c for c, s in sol1.items() if s["sign_ok"]]
    suspect = [c for c, s in sol1.items() if not s["sign_ok"]]
    print(f"[pass1] solved {len(sol1)}/120 (freeze {len(frozen)} sign-correct, "
          f"{len(suspect)} plane-only/sign-suspect) {time.time()-t0:.0f}s",
          flush=True)

    # inject solved rows (plane-solves in inherited orientation), freeze the
    # sign-correct set, train the rest + L4/L5
    W3 = r.layers[FRONTIER].weight
    b3 = r.layers[FRONTIER].bias
    with torch.no_grad():
        for c, s in sol1.items():
            wb = torch.cat([s["w"], s["b"].reshape(1)])
            if s["orient"] == "negated":
                wb = -wb                       # back to the guess's orientation
            W3.data[c] = wb[:-1].view(W3.shape[1:]).to(W3.dtype)
            b3.data[c] = wb[-1].to(b3.dtype)
    for i, l in enumerate(r.layers):
        req = i >= FRONTIER
        l.weight.requires_grad_(req); l.bias.requires_grad_(req)
    fmask = torch.zeros(120, dtype=torch.bool, device=dev)
    fmask[torch.tensor(frozen, device=dev, dtype=torch.long)] = True
    free = [c for c in range(120) if c not in frozen]
    opt = torch.optim.Adam([p for p in r.parameters() if p.requires_grad],
                           lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(0)
    pool = None
    ea, cat = cats_eps(r, Dt)
    print(f"[train] start eps med {ea.median():.2e} | {cat} | free rows "
          f"{len(free)}", flush=True)
    for step in range(args.steps + 1):
        if (args.queries == "straddle" and step % 4000 == 0
                and step < args.steps):
            with torch.no_grad():
                W3g = W3.reshape(120, -1).detach().clone()
                b3g = b3.detach().clone()
            pool_all = synth_straddle(r, W3g, b3g, n_per=60, dev=dev, gen=gen)
            # keep only rows targeting FREE channels (c = index % 120)
            keep = torch.tensor([i for i in range(len(pool_all))
                                 if (i % 120) in set(free)], device=dev)
            pool = pool_all[keep]
        if step % 2000 == 0:
            ea, cat = cats_eps(r, Dt)
            print(f"  step {step:5d}: eps med {ea.median():.2e} | {cat}",
                  flush=True)
        if step == args.steps:
            break
        xr = torch.randn(2048 if pool is not None else 4096, 784,
                         generator=gen, device=dev) * 0.5
        if pool is not None:
            idx = torch.randint(0, len(pool), (2048,), generator=gen,
                                device=dev)
            x = torch.cat([xr, pool[idx]])
        else:
            x = xr
        with torch.no_grad():
            y = t(x)
        opt.zero_grad()
        F.mse_loss(r(x), y).backward()
        with torch.no_grad():                  # anchor the frozen rows
            W3.grad[fmask] = 0
            b3.grad[fmask] = 0
        opt.step()

    # ---- pass 2: refine the still-unsolved ----
    pre = r.clone().double().to(dev).eval()
    todo = [c for c in range(120) if c not in sol1]
    sol2 = refine_all(pre, _Oracle(td), todo, Dt, dev, args.probes)
    # also re-check the sign-suspect planes (their rows moved during training)
    sol2b = refine_all(pre, _Oracle(td), suspect, Dt, dev, args.probes)
    total = len(frozen) + len(suspect) + len(sol2)
    new = sorted(sol2.keys())
    print(f"\n[{args.queries} FINAL] pass2 newly solved {len(sol2)}/{len(todo)}"
          f" -> TOTAL planes {total}/120 | newly: "
          f"{[(c, 'signOK' if sol2[c]['sign_ok'] else 'FLIP') for c in new]}",
          flush=True)
    if sol2b:
        print(f"  sign-suspect recheck: {[(c, 'signOK' if sol2b[c]['sign_ok'] else 'still-flip') for c in sorted(sol2b)]}")
    ea, cat = cats_eps(r, Dt)
    print(f"  end guesses: eps med {ea.median():.2e} | {cat} | wall "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
