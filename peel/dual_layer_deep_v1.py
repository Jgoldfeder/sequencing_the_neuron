"""Layer-L row DIRECTIONS by self-saturation duals (Judah's recursion for layers >= 2).
Prefix (layers < L) exact, layer L rows guessed (sign pattern only is used), deeper layers unknown.
At the corner a = sign(w_hat_j) neuron j saturates and drops out of the Jacobian w.r.t. a (the layer's
input); stacked Jacobians at NPTS points near the corner span S_{-j}; their null direction inside
rowspace(W_L) is the dual u_j (w_k . u_j = 0 for k != j). All duals -> rows of pinv(U) = row directions.
Also reports the LP-maximal |z_j| reachable along each dual inside the 0.995-box (tail-decay feasibility).
Usage: python peel/dual_layer_deep.py <pop.pt> <guess.pt> <L> <out.pt> [gpu] [prefix: teacher|<net.pt>] [npts] [neurons a-b]
L is 1-indexed: L=2 is the first 200x200 layer (its input a = tanh(W1 x + b1))."""
import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nets import MLP
from tanh_solve import Oracle
from scipy.optimize import linprog
pop_path, guess_path, L, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
dev = f"cuda:{sys.argv[5]}" if len(sys.argv) > 5 else "cuda:0"
prefix_src = sys.argv[6] if len(sys.argv) > 6 else "teacher"
NPTS = int(sys.argv[7]) if len(sys.argv) > 7 else 40
BUDGET = float(os.environ.get("BUDGET", "1.5"))
torch.manual_seed(0)
ck = torch.load(pop_path, map_location=dev, weights_only=False); dims = ck["dims"]
T = MLP(dims, act="tanh").to(dev); T.load_state_dict(ck["teacher_state"]); T = T.double()
bb = Oracle(lambda x: T(x.double()))
g = torch.load(guess_path, map_location=dev, weights_only=False)
G = MLP(dims, act="tanh").to(dev); G.load_state_dict(g["state_dict"]); G = G.double()
# ---- prefix: layers 0..L-2 (exact), in the coordinate system of `src` ----------------------------
src = T if prefix_src == "teacher" else G if prefix_src == "guess" else None
if src is None:
    pn = torch.load(prefix_src, map_location=dev, weights_only=False); src = MLP(dims, act="tanh").to(dev); src.load_state_dict(pn["state_dict"]); src = src.double()
pre = [(l.weight.detach().clone(), l.bias.detach().clone()) for l in src.layers[:L - 1]]
print(f"layer {L}: prefix = {prefix_src} ({L-1} layers), guess rows from {os.path.basename(guess_path)} | NPTS {NPTS} BUDGET {BUDGET}", flush=True)
width = dims[L - 1]                                    # input width of layer L
# map a (input of layer L) -> x: invert the prefix layer by layer (square layers by solve, L1 by pinv)
def x_of_a(A):
    Z = torch.atanh(A.clamp(-0.999999, 0.999999))
    for (W, b) in reversed(pre[1:]):
        Z = torch.linalg.solve(W, (Z - b).T).T; Z = torch.atanh(Z.clamp(-0.999999, 0.999999))
    W1, b1 = pre[0]
    return (Z - b1) @ torch.linalg.pinv(W1).T
def reach_ok(A):
    """fraction of rows whose inversion stays inside |h|<0.995 at every intermediate level."""
    Z = torch.atanh(A.clamp(-0.999999, 0.999999)); ok = torch.ones(len(A), dtype=torch.bool, device=dev)
    for (W, b) in reversed(pre[1:]):
        Hprev = torch.linalg.solve(W, (Z - b).T).T; ok &= (Hprev.abs() < 0.995).all(1); Z = torch.atanh(Hprev.clamp(-0.999999, 0.999999))
    return float(ok.float().mean())
def jac_a(A, h=1e-4):
    """Jacobian of the black box w.r.t. a at rows of A (N, width): FD in a directly along e_k."""
    N = len(A); E = torch.eye(width, dtype=torch.float64, device=dev)
    f = [bb(x_of_a((A[:, None, :] + s * h * E[None]).reshape(-1, width))).reshape(N, width, -1) for s in (1.0, -1.0, 2.0, -2.0)]
    return ((8 * (f[0] - f[1]) - (f[2] - f[3])) / (12 * h)).transpose(1, 2)      # (N, out, width)
# ---- the guess rows of layer L, expressed in the prefix's input coordinates --------------------------
Wg = G.layers[L - 1].weight.detach().clone()
if prefix_src != "guess":
    # pair the guess's layer-(L-1) units to the prefix's by |cos| of their incoming rows (+sign gauge)
    Rg = G.layers[L - 2].weight.detach(); Rs = src.layers[L - 2].weight.detach()
    C = (Rg / Rg.norm(dim=1, keepdim=True)) @ (Rs / Rs.norm(dim=1, keepdim=True)).T
    k = C.abs().argmax(1); sgn = torch.sign(C[torch.arange(width), k])
    assert len(set(k.tolist())) == width, "guess->prefix unit pairing is not a permutation"
    Wg_t = torch.zeros_like(Wg); Wg_t[:, k] = Wg * sgn[None]; Wg = Wg_t
    print(f"  guess layer-{L-1} units paired to prefix units: min |cos| {float(C.abs().max(1).values.min()):.4f}", flush=True)
# ---- truth (scoring only) ----------------------------------------------------------------------
Wt = T.layers[L - 1].weight.detach(); bt = T.layers[L - 1].bias.detach()
if prefix_src != "teacher":   # express truth in the prefix's coordinates
    Rs = src.layers[L - 2].weight.detach(); Rt = T.layers[L - 2].weight.detach()
    C = (Rs / Rs.norm(dim=1, keepdim=True)) @ (Rt / Rt.norm(dim=1, keepdim=True)).T
    kt = C.abs().argmax(1); sg = torch.sign(C[torch.arange(width), kt])
    Wt = Wt[:, kt] * sg[None]
Wtn = Wt / Wt.norm(dim=1, keepdim=True)
Ut = torch.linalg.pinv(Wt)                                                        # columns = true duals
def derr(u, v):
    u = u / u.norm(); v = v / v.norm(); return float(torch.minimum((u - v).norm(), (u + v).norm()))
# ---- row space of W_L from interior Jacobians ---------------------------------------------------------
A0 = (torch.rand(40, width, dtype=torch.float64, device=dev) * 2 - 1) * 0.9
J0 = jac_a(A0).reshape(-1, width); S0, Vh0 = torch.linalg.svd(J0, full_matrices=False)[1:]
r = int((S0 > 1e-7 * S0[0]).sum()); VS = Vh0[:r]
print(f"  rowspace(W{L}): rank {r}/{width} (sv[r-1]/sv0 {float(S0[r-1]/S0[0]):.1e}, sv[r]/sv0 {float(S0[min(r, width-1)]/S0[0]):.1e}) | queries {bb.n}", flush=True)
# ---- duals --------------------------------------------------------------------------------------------
sel = range(dims[L]) if len(sys.argv) <= 8 else range(int(sys.argv[8].split("-")[0]), int(sys.argv[8].split("-")[1]) + 1)
U = torch.zeros(width, dims[L], dtype=torch.float64, device=dev); info = {}
t0 = time.time()
for j in sel:
    q0 = bb.n; wg = Wg[j]; s = torch.sign(wg)
    rr = torch.rand(NPTS, width, dtype=torch.float64, device=dev) * 0.8 + 0.2
    delta = (BUDGET * rr / (width * 0.6 * wg.abs()[None])).clamp(0.02, 1.2)
    A = (s[None] * (1 - delta)).clamp(-0.98, 0.98)
    rf = reach_ok(A) if L > 2 else 1.0
    J = jac_a(A).reshape(-1, width) @ VS.T
    Sg, Vh = torch.linalg.svd(J, full_matrices=False)[1:]
    u = VS.T @ Vh[-1]; U[:, j] = u
    jt = int((Wtn @ (wg / wg.norm())).abs().argmax())                              # teacher unit (scoring)
    zj = float((A @ Wt[jt] + bt[jt]).abs().min()); e = derr(u, Ut[:, jt])
    others = (A @ Wt.T + bt[None]); others[:, jt] = 0; zo = float(others.abs().max())
    info[j] = dict(dual=u.cpu(), gap=float(Sg[-1] / Sg[-2]), teacher_unit=jt, dual_err=e, zmin=zj, reach=rf, queries=bb.n - q0)
    if j < 3 or j % 25 == 0 or e > 1e-6:
        print(f"  neuron {j:3d} (teacher {jt:3d}): min|z_j| at probes {zj:5.1f} | max|z_k| others {zo:4.1f} | sv_last/sv_prev {float(Sg[-1]/Sg[-2]):.1e} | reach {rf:.2f} | DUAL dir err {e:.2e} | {bb.n-q0} q", flush=True)
de = torch.tensor([info[j]["dual_err"] for j in sel])
print(f"DUALS {len(sel)}: dir err median {float(de.median()):.2e} max {float(de.max()):.2e} | {time.time()-t0:.0f}s | queries {bb.n}", flush=True)
# ---- primal row directions from all duals -------------------------------------------------------------
if len(sel) == dims[L]:
    West = torch.linalg.pinv(U).T.T                                                  # (n_L, width), rows ∝ w_j
    perr = torch.tensor([derr(West[j], Wt[info[j]["teacher_unit"]]) for j in sel])
    gerr = torch.tensor([derr(Wg[j], Wt[info[j]["teacher_unit"]]) for j in sel])
    print(f"PRIMAL rows from all duals: dir err median {float(perr.median()):.2e} max {float(perr.max()):.2e} (guess was median {float(gerr.median()):.2e} max {float(gerr.max()):.2e})", flush=True)
    # ---- tail-decay feasibility along the dual inside the 0.995-box (LP: maximise reachable |z_j|) ----
    print("tail-decay feasibility along the dual (LP, 0.995-box): neuron | kappa=|w.u| | max z_top reachable (+side, -side)", flush=True)
    feas = []
    for j in list(sel)[:8]:
        u = (U[:, j] / U[:, j].norm()).cpu().numpy(); w = West[j]; w = (w / w.norm() * Wg[j].norm()).cpu().numpy(); b = float(G.layers[L - 1].bias[j])
        kappa = abs(float(np.dot(w, u)))
        best = []
        for sign in (+1, -1):
            # variables (a0 in R^width, tau): maximise tau s.t. |a0| <= .995, |a0 + tau*sign*u| <= .995, w.a0 + b = 0
            n = width; c = np.zeros(n + 1); c[-1] = -1.0
            A_ub = np.zeros((2 * n, n + 1)); b_ub = np.full(2 * n, 0.995)
            A_ub[:n, :n] = np.eye(n); A_ub[:n, -1] = sign * u
            A_ub[n:, :n] = -np.eye(n); A_ub[n:, -1] = -sign * u
            A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = w; b_eq = np.array([-b])
            bounds = [(-0.995, 0.995)] * n + [(0, None)]
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            best.append(kappa * float(res.x[-1]) if res.success else float("nan"))
        feas.append(best); print(f"  {j:3d} | kappa {kappa:.4f} | z_top {best[0]:.2f} / {best[1]:.2f}", flush=True)
    torch.save(dict(U=U.cpu(), West=West.cpu(), info=info, dims=dims, L=L, prefix=prefix_src, feas=feas), out_path)
else:
    torch.save(dict(U=U.cpu(), info=info, dims=dims, L=L, prefix=prefix_src), out_path)
print(f"total queries {bb.n}", flush=True)
