"""Experiment A: information-gain probe selection vs the baseline random probes, same solver, same budget.
Weak parameter directions at the current guess are found by inverse-power iteration (CG solves of
(J^T J + lam I) v = w from random w on the BASELINE probe set); candidate points are scored by
sum_v |A(x) v|^2 (the model-jet sensitivity at x along the weak directions, no oracle needed); the top
points are queried and tanh_solve.refine runs on them. Usage: python peel/diag/probe_design.py <gpu> <variant>
variant: baseline | design | mixed"""
import sys, os, math, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, Problem, fmt
import tanh_solve as ts
from method import param_errors
dev = f"cuda:{sys.argv[1]}"; variant = sys.argv[2]; ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
dims, t64, net, bb = load(dev)
d = dims[0]
X0 = ts.sample_points(512, d, scales=(0.5, 1.0, 2.0), seed=0, device=dev)
if variant == "baseline":
    X = X0
else:
    t0 = time.time()
    P = Problem(bb, net, X0, dirs=64)
    r, g, JtJ, Jv = P.linearize()
    gen = torch.Generator(device="cpu").manual_seed(1)
    v = torch.randn(P.numel, generator=gen, dtype=torch.float64).to(dev); lam = 1e-6 * float(v @ JtJ(v)) / P.numel
    # inverse-power: v <- (JtJ + lam)^-1 w amplifies the weakly constrained directions
    weak = []
    for i in range(8):
        w = torch.randn(P.numel, generator=gen, dtype=torch.float64).to(dev)
        for u in weak: w -= (w @ u) * u
        sol, n = ts._cg(lambda z: JtJ(z) + lam * z, w / w.norm(), 200, 1e-6)
        for u in weak: sol -= (sol @ u) * u
        sol = sol / sol.norm(); weak.append(sol)
        print(f"  weak dir {i}: cg {n}, Rayleigh v^T JtJ v = {float(sol @ JtJ(sol)):.3e} (mean diag {lam/1e-6:.3e})", flush=True)
    # candidates: fresh Gaussian points at 4 scales; score = sum_v |A(x) v|^2 via one jvp per direction
    C = ts.sample_points(4096, d, scales=(0.5, 1.0, 2.0, 3.0), seed=11, device=dev)
    Pc = Problem.__new__(Problem); Pc.X = C.double(); Pc.dims = P.dims; Pc.n_layers = P.n_layers; Pc.act = P.act
    genc = torch.Generator(device="cpu").manual_seed(11)
    Uc = torch.randn(len(C), 64, d, generator=genc, dtype=torch.float64).to(dev); Pc.U = Uc / Uc.norm(dim=2, keepdim=True)
    Pc.freeze = P.freeze; Pc.keys = P.keys; Pc.theta = P.theta
    score = torch.zeros(len(C), dtype=torch.float64, device=dev)
    from torch.func import jvp
    CH = 256                                                   # score candidates in chunks (4096 at once OOMs a 24 GB card)
    for c0 in range(0, len(C), CH):
        Xc, Ucc = Pc.X[c0:c0 + CH], Pc.U[c0:c0 + CH]
        def jets_only(p, Xc=Xc, Ucc=Ucc):
            y, ju = ts.model_jets(p, Xc, Ucc, Pc.act, Pc.n_layers); return y, ju
        for u in weak:
            (_, _), (dy, dju) = jvp(jets_only, (P.theta,), (P.unflat(u),))
            score[c0:c0 + CH] += (dy ** 2).sum(1) + (dju ** 2).sum((1, 2))
        del dy, dju
    order = torch.argsort(score, descending=True)
    print(f"  candidate scores: top {float(score[order[0]]):.3e} | 512th {float(score[order[511]]):.3e} | median {float(score.median()):.3e} | baseline-set-like (random) median {float(score[torch.randperm(len(C))[:512]].median()):.3e}", flush=True)
    sel = order[:512] if variant == "design" else order[:256]
    X = C[sel] if variant == "design" else torch.cat([X0[:256], C[sel]])
    print(f"  design done in {time.time()-t0:.0f}s; selected scales: {[round(float(C[i].norm()/math.sqrt(d)),1) for i in sel[:8].tolist()]}...", flush=True)
net64, info = ts.refine(bb, net, X, "tanh", dirs=64, iters=ITERS, cg_iters=300, seed=0,
                        score=lambda n: param_errors(n, t64), verbose=True, tag=f"[{variant}]", freeze={"layers.0.weight"})
pe = param_errors(net64, t64)
print(f"FINAL [{variant}] max_eps {pe['max_eps']:.3e} | max per matrix [{fmt(pe['max_eps_per_matrix'])}] | mean per matrix [{fmt(pe['mean_eps_per_matrix'])}] | {info['queries']} queries | {info['wall_s']}s", flush=True)
