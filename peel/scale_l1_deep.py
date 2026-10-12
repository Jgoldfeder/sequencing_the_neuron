"""Stage 2: layer-1 row SCALE by tail decay, given the exact direction (from isolate_l1_deep.py).
Along x = t v + s u (v in row(W1), v.u=0, t large) only neuron i is unsaturated; the derivative
vector g(s) = d f/ds = G_i'(h) |w| sech^2(|w| s + b) decays as exp(-2|w| |s - s_c|) once |z| > 3.
Fit, per side, the projection of g onto the output direction of the SHALLOW tail end:
    p(tau) = A exp(-2 a tau) (1 + c1 exp(-2 a tau) + c2 exp(-4 a tau)) + B      (B = leakage floor)
with a = |w|. Usage: python peel/scale_l1_deep.py <teacher.pt> <iso_results.pt> <out.pt> [gpu]"""
import sys, os, time, math, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nets import MLP
from tanh_solve import Oracle, measure_jets
from scipy.optimize import least_squares
teacher_path, iso_path, out_path = sys.argv[1:4]
dev = f"cuda:{sys.argv[4]}" if len(sys.argv) > 4 else "cuda:0"
torch.manual_seed(1)
ck = torch.load(teacher_path, map_location=dev, weights_only=False); dims = ck["dims"]
teacher = MLP(dims, act="tanh").to(dev); teacher.load_state_dict(ck["teacher_state"]); teacher = teacher.double()
bb = Oracle(lambda x: teacher(x.double())); d = dims[0]
iso = torch.load(iso_path, map_location="cpu", weights_only=False); res = iso["results"]
# truth for scoring: pair isolated directions to teacher rows by max |cos|
Wt = teacher.layers[0].weight.detach(); Wtn = Wt / Wt.norm(dim=1, keepdim=True)
# row space of W1 (for v)
X = torch.randn(40, d, dtype=torch.float64, device=dev)
_, J = measure_jets(bb, X, None, h=1e-3); S, Vh = torch.linalg.svd(J.reshape(-1, d), full_matrices=False)[1:]
Vsub = Vh[:int((S > 1e-8 * S[0]).sum())]
def rand_perp(u):
    gg = torch.randn(d, dtype=torch.float64, device=dev); gg = Vsub.T @ (Vsub @ gg); gg = gg - (gg @ u) * u
    return gg / gg.norm()
def deriv(P, u, h=1e-3):          # 4th-order central difference of f along u at the points P
    return (8 * (bb(P + h * u) - bb(P - h * u)) - (bb(P + 2 * h * u) - bb(P - 2 * h * u))) / (12 * h)
out = {}
t0 = time.time()
for i, r in sorted(res.items()):
    q0 = bb.n
    u = r["direction"].to(dev).double(); u = u / u.norm()
    j = int((Wtn @ u).abs().argmax()); wn_true = float(Wt[j].norm())
    # candidate probe lines: 8 random v per t, sorted by purity at the centre; a side is accepted only if
    # its tail is a clean single exponential (local slope constant) and the LM fit agrees with the slope.
    ests, tried, purity = [], 0, None
    for t in (3000.0, 10000.0, 30000.0):
        cands = []
        for m in range(8):
            v = rand_perp(u)
            s = torch.linspace(-6, 6, 1201, dtype=torch.float64, device=dev)
            P = t * v[None] + s[:, None] * u[None]
            gq = deriv(P, u).norm(dim=1); sc = float(s[gq.argmax()])
            _, Jx = measure_jets(bb, (t * v + sc * u)[None], None, h=1e-3); Sg = torch.linalg.svdvals(Jx[0])
            cands.append((float(Sg[1] / Sg[0]), v, sc))
        cands.sort(key=lambda c: c[0])
        for pur, v, sc in cands[:4]:
            if pur > 1e-9:                                   # dirty at the centre: never trust this line
                continue
            tried += 1
            s = torch.linspace(sc - 6, sc + 6, 4801, dtype=torch.float64, device=dev)
            P = t * v[None] + s[:, None] * u[None]; G = deriv(P, u)                     # (n, out)
            gn = G.norm(dim=1); pk = float(gn.max())
            for side in (+1, -1):
                m = ((s - sc) * side > 0)
                tau = ((s - sc) * side)[m]; Gs = G[m]; gs = gn[m]
                sh = (gs < 1e-2 * pk) & (gs > 1e-3 * pk)
                if int(sh.sum()) < 5:
                    continue
                c = Gs[sh].mean(0); c = c / c.norm()
                p = (Gs @ c)
                win = (gs < 1e-2 * pk) & (gs > 1e-9 * pk)
                tau_w = tau[win].cpu().numpy(); p_w = p[win].cpu().numpy()
                if len(tau_w) < 20 or np.any(np.sign(p_w) != np.sign(p_w[0])):
                    continue
                # slope-constancy check on the clean part (1e-3 .. 1e-7 of the peak)
                chk = (gs < 1e-3 * pk) & (gs > 1e-7 * pk)
                tc = tau[chk].cpu().numpy(); lc = np.log(np.abs(p[chk].cpu().numpy()))
                if len(tc) < 10:
                    continue
                loc = np.diff(lc[::4]) / np.diff(tc[::4])
                if loc.max() - loc.min() > 0.05:
                    continue
                lin = (gs < 1e-2 * pk) & (gs > 1e-5 * pk)
                A0 = np.polyfit(tau[lin].cpu().numpy(), np.log(np.abs(p[lin].cpu().numpy()) + 1e-300), 1)
                a0 = -A0[0] / 2; sgn = np.sign(p_w[0])
                def model(th, tau):
                    lA, a, c1, c2, B = th; e = np.exp(-2 * a * tau)
                    return sgn * np.exp(lA) * e * (1 + c1 * e + c2 * e * e) + B
                def resid(th):
                    return (model(th, tau_w) - p_w) / (np.abs(p_w) + 1e-6 * np.abs(p_w).max())
                th0 = np.array([A0[1], a0, 0.0, 0.0, 0.0])
                fit = least_squares(resid, th0, method="lm", max_nfev=4000)
                if not fit.success or abs(fit.x[1] / a0 - 1) > 5e-3:
                    continue
                ests.append((side, float(fit.x[1]), a0, int(win.sum())))
            if len(ests) >= 2:                               # two estimates agreeing to 2e-5 -> done
                a = np.array(sorted(e[1] for e in ests)); gap = np.diff(a) / a[:-1]
                if gap.min() < 2e-5:
                    k = int(gap.argmin()); ests = [e for e in ests if a[k] <= e[1] <= a[k + 1]]; purity = pur
                    break
        if len(ests) >= 2 and purity is not None:
            break
    if ests and purity is None:
        purity = -1.0                                        # no agreeing pair: fall back to the median
        ests = [sorted(ests, key=lambda e: e[1])[len(ests) // 2]]
    if not ests:
        out[i] = dict(scale=None); print(f"neuron {i:3d}: no usable tail", flush=True); continue
    a_fit = float(np.mean([e[1] for e in ests])); a_lin = float(np.mean([e[2] for e in ests]))
    out[i] = dict(scale=a_fit, scale_linear=a_lin, purity=purity, teacher_unit=j, sides=len(ests), lines_tried=tried, t=t, queries=bb.n - q0)
    print(f"neuron {i:3d} (teacher unit {j}): |w| true {wn_true:.6f} | tail fit {a_fit:.6f} (rel err {abs(a_fit/wn_true-1):.1e}) | plain log-linear {a_lin:.6f} (rel err {abs(a_lin/wn_true-1):.1e}) | line purity sv2/sv1 {purity:.0e} | sides {len(ests)} lines {tried} t {t:.0f} | {bb.n-q0} q", flush=True)
torch.save(dict(results=out, dims=dims), out_path)
errs = [abs(out[i]["scale"] / float(Wt[out[i]["teacher_unit"]].norm()) - 1) for i in out if out[i].get("scale")]
print(f"DONE {len(errs)} scales | rel err median {sorted(errs)[len(errs)//2]:.1e} max {max(errs):.1e} | {time.time()-t0:.0f}s", flush=True)
