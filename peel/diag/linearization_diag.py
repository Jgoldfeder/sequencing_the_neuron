"""Diagnostic (ChatGPT's section 3) on the stalled 10-layer snapshot: at the current guess, solve the damped
normal equations (J^T J + lam I) d = -J^T r by CG to HIGH accuracy and report E_before = |r|, E_linear =
|r + J d|, E_actual = |r(theta + d)|, the gain ratio, and per-layer eps after the step -- for several lam
and with the CG residual history (the production solver caps CG at 300 iterations). Also a fresh held-out
probe set to check the step generalises. Usage: python peel/diag/linearization_diag.py <gpu>"""
import sys, os, math, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, Problem, fmt
import tanh_solve as ts
dev = f"cuda:{sys.argv[1] if len(sys.argv) > 1 else 0}"
dims, t64, net, bb = load(dev)
X = ts.sample_points(512, dims[0], scales=(0.5, 1.0, 2.0), seed=0, device=dev)
P = Problem(bb, net, X, dirs=64)
Xh = ts.sample_points(256, dims[0], scales=(0.5, 1.0, 2.0), seed=7, device=dev)
Ph = Problem(bb, net, Xh, dirs=64, seed=7)
print(f"probe jets: {P.queries} queries | free params {P.numel} | residual rows {P.Y.numel() * (1 + 64)}", flush=True)
r, g, JtJ, Jv = P.linearize()
E0 = float(r.norm()); print(f"E_before |r| = {E0:.4e}  (loss {E0**2:.3e}) | |J^T r| = {float(g.norm()):.3e} | held-out |r| = {math.sqrt(Ph.loss(P.theta)):.4e}", flush=True)
e0, pm0, mm0 = P.errors(P.theta, t64); print(f"eps before: max {e0:.3e} | max per matrix [{fmt(pm0)}]", flush=True)
gen = torch.Generator(device="cpu").manual_seed(0)
v = torch.randn(P.numel, generator=gen, dtype=torch.float64).to(dev); diag_mean = float(v @ JtJ(v)) / P.numel
print(f"mean diag(J^T J) ~ {diag_mean:.3e}", flush=True)
for c in (1e-4, 1e-7, 1e-10):
    lam = c * diag_mean
    hist = []
    t0 = time.time()
    # CG with a residual history: record |JtJ d + g| at checkpoints
    x = torch.zeros_like(g); rr = -g.clone(); p = rr.clone(); rz = rr @ rr; b_norm = math.sqrt(float(rz))
    checkpoints = {300, 1000, 2000, 4000}
    for k in range(1, 4001):
        Ap = JtJ(p) + lam * p
        alpha = rz / (p @ Ap); x += alpha * p; rr -= alpha * Ap
        rn = math.sqrt(float(rr @ rr)) / b_norm
        if k in checkpoints or rn < 1e-9:
            cand = P.step(P.theta, x)
            with torch.no_grad():
                Jd = Jv(x)
            E_lin = float((r + Jd).norm()); E_act = math.sqrt(P.loss(cand)); E_held = math.sqrt(Ph.loss(cand))
            pred = float(-2.0 * (g @ x) - (x @ (JtJ(x)))); rho = (E0**2 - E_act**2) / pred if pred > 0 else float('nan')
            em, pm, mm = P.errors(cand, t64)
            hist.append((k, rn, E_lin, E_act, E_held, rho, em))
            print(f"lam={c:.0e}*diag  cg {k:4d} | CG rel res {rn:.1e} | |d| {float(x.norm()):.2e} | E_linear {E_lin:.4e} | E_actual {E_act:.4e} | held-out {E_held:.4e} | rho {rho:.3f} | eps max {em:.3e} [{fmt(pm)}] | {time.time()-t0:.0f}s", flush=True)
            if rn < 1e-9:
                break
        rz_new = rr @ rr; p = rr + (rz_new / rz) * p; rz = rz_new
print("DONE", flush=True)
