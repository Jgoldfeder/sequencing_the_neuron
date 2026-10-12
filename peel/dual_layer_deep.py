"""Layer-L row DIRECTIONS by self-saturation duals (Judah's recursion for layers >= 2).
Prefix (layers < L) exact, layer L rows guessed (the guess net supplies sign patterns / ascent targets only),
deeper layers unknown. Where neuron j of layer L saturates it drops out of the Jacobian w.r.t. the layer's
input a; stacked Jacobians at NPTS such probes span S_{-j}; the null direction inside rowspace(W_L) is the
dual u_j (w_k . u_j = 0, k != j). All duals -> rows of pinv(U) = row directions.
Probe modes (env PROBE): corner (L=2 default) = a near sign(w_hat_j) via the pinv of layer 1;
ascent (L>=3 default) = Adam on x with the GUESS net, maximising +-z_L,j with a penalty keeping every
layer-(L-1) preactivation inside |z| <= ZMAX (so every input coordinate stays movable); Jacobian w.r.t. a
from jets along pinv(J_prefix(x)) e_k. Usage:
python peel/dual_layer_deep.py <pop.pt> <guess.pt> <L> <out.pt> [gpu] [prefix: teacher|<net.pt>] [npts] [neurons a-b]
env: PROBE, BUDGET (corner, 1.5), ZMAX (ascent, 2.3), ZMIN (ascent min predicted |z_j|, 6)."""
import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nets import MLP
from tanh_solve import Oracle, measure_jets
from scipy.optimize import linprog
pop_path, guess_path, L, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
dev = f"cuda:{sys.argv[5]}" if len(sys.argv) > 5 else "cuda:0"
prefix_src = sys.argv[6] if len(sys.argv) > 6 else "teacher"
NPTS = int(sys.argv[7]) if len(sys.argv) > 7 else 40
BUDGET = float(os.environ.get("BUDGET", "1.5")); ZMAX = float(os.environ.get("ZMAX", "2.3")); ZMIN = float(os.environ.get("ZMIN", "6")); ZOTH = float(os.environ.get("ZOTH", "3.0"))
PROBE = os.environ.get("PROBE", "corner" if L == 2 else "ascent")
torch.manual_seed(0); last_amp = 0.0
ck = torch.load(pop_path, map_location=dev, weights_only=False); dims = ck["dims"]
T = MLP(dims, act="tanh").to(dev); T.load_state_dict(ck["teacher_state"]); T = T.double()
bb = Oracle(lambda x: T(x.double()))
g = torch.load(guess_path, map_location=dev, weights_only=False)
G = MLP(dims, act="tanh").to(dev); G.load_state_dict(g["state_dict"]); G = G.double()
src = T if prefix_src == "teacher" else G if prefix_src == "guess" else None
if src is None:
    pn = torch.load(prefix_src, map_location=dev, weights_only=False); src = MLP(dims, act="tanh").to(dev); src.load_state_dict(pn["state_dict"]); src = src.double()
pre = [(l.weight.detach().clone(), l.bias.detach().clone()) for l in src.layers[:L - 1]]       # exact prefix
width, nL = dims[L - 1], dims[L]
print(f"layer {L}: prefix = {prefix_src} ({L-1} layers), guess = {os.path.basename(guess_path)} | PROBE {PROBE} NPTS {NPTS} BUDGET {BUDGET} ZMAX {ZMAX} ZOTH {ZOTH} ZMIN {ZMIN}", flush=True)
def preacts(net, x):
    zs = []; h = x
    for l in net.layers[:-1]:
        z = l(h); zs.append(z); h = torch.tanh(z)
    return zs
def pair_units(netA, netB, layer):
    """perm, sign: unit i of netA's layer `layer` (1-indexed) <-> unit perm[i] of netB, by standardised preact corr."""
    X = torch.randn(4096, dims[0], dtype=torch.float64, device=dev) * 0.3
    za = preacts(netA, X)[layer - 1]; zb = preacts(netB, X)[layer - 1]
    za = (za - za.mean(0)) / za.std(0); zb = (zb - zb.mean(0)) / zb.std(0); C = (za.T @ zb) / len(X)
    perm = C.abs().argmax(1); sgn = torch.sign(C[torch.arange(C.shape[0]), perm])
    assert len(set(perm.tolist())) == C.shape[0], f"layer {layer} pairing not a permutation"
    return perm, sgn, float(C.abs().max(1).values.min())
# ---- prefix maps -----------------------------------------------------------------------------------
def a_of_x(x):
    h = x
    for (W, b) in pre: h = torch.tanh(h @ W.T + b)
    return h
def zprev_of_x(x):                       # layer-(L-1) preactivations (exact prefix)
    h = x
    for (W, b) in pre[:-1]: h = torch.tanh(h @ W.T + b)
    W, b = pre[-1]; return h @ W.T + b
def prefix_zmax(x):                      # max |z| over ALL prefix layers (exact prefix)
    h = x; m = torch.zeros(len(x), dtype=torch.float64, device=dev)
    for (W, b) in pre: z = h @ W.T + b; m = torch.maximum(m, z.abs().max(1).values); h = torch.tanh(z)
    return m
def Jpre(x):                             # (N, width, d): d a / d x from the exact prefix
    N = len(x); h = x; J = None
    for (W, b) in pre:
        z = h @ W.T + b; D = 1 - torch.tanh(z) ** 2
        J = D[:, :, None] * (W[None] if J is None else W[None] @ J); h = torch.tanh(z)
    return J
def x_of_a(A):                           # corner mode (L=2): invert layer 1 by pinv
    W1, b1 = pre[0]; return (torch.atanh(A.clamp(-0.999999, 0.999999)) - b1) @ torch.linalg.pinv(W1).T
def jac_a_corner(A, h=1e-4):
    N = len(A); E = torch.eye(width, dtype=torch.float64, device=dev)
    f = [bb(x_of_a((A[:, None, :] + s * h * E[None]).reshape(-1, width))).reshape(N, width, -1) for s in (1.0, -1.0, 2.0, -2.0)]
    return ((8 * (f[0] - f[1]) - (f[2] - f[3])) / (12 * h)).transpose(1, 2)
def jac_a_at_x(X, h=1e-3):
    """Jacobian of the black box w.r.t. a = layer-L input at inputs X: jets along pinv(Jpre) e_k."""
    V = torch.linalg.pinv(Jpre(X))                                   # (N, d, width): column k moves a_k alone
    nrm = V.norm(dim=1)                                              # (N, width)
    global last_amp; last_amp = float(nrm.max())
    U = (V / nrm[:, None, :]).transpose(1, 2)                        # (N, width, d) unit directions
    _, JU = measure_jets(bb, X, U, h=h)                              # (N, out, width)
    return JU * nrm[:, None, :]
# ---- ascent probes (guess net): saturate neuron j, keep layer L-1 movable --------------------------------
NSTART = int(os.environ.get("NSTART", "100"))
def ascent_probes(j, n, steps=2500):
    """NSTART starts per sign; keep probes whose exact prefix stays inside ZMAX and whose predicted |z_j| >= ZMIN;
    return the n most saturated (a handful of near-duplicate probes gives a rank-deficient stack)."""
    xs, zs_ = [], []
    for sign in (+1, -1):
        x = (torch.randn(NSTART, dims[0], dtype=torch.float64, device=dev) * (0.1 + 0.4 * torch.rand(NSTART, 1, dtype=torch.float64, device=dev))).requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.03); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0.002)
        for it in range(steps):
            zs = preacts(G, x); zo = zs[L - 1].clone(); zo[:, j] = 0
            pen = sum(torch.relu(z.abs() - ZMAX).pow(2).sum(1) for z in zs[:L - 1]) + torch.relu(zo.abs() - ZOTH).pow(2).sum(1)
            loss = (-(sign * zs[L - 1][:, j]) + 300.0 * pen).sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            zg = sign * preacts(G, x)[L - 1][:, j]; ok = (zg >= ZMIN) & (prefix_zmax(x) <= ZMAX + 0.1)   # exact-prefix check
            xs.append(x.detach()[ok]); zs_.append(zg[ok])
    X = torch.cat(xs, 0); Z = torch.cat(zs_, 0)
    if len(X) == 0: return X
    idx = Z.argsort(descending=True)[:n]
    return X[idx]
# ---- truth (scoring only), in the prefix's coordinates ---------------------------------------------
permG, sgnG, cG = pair_units(G, T, L)                       # guess layer-L unit j <-> teacher unit permG[j]
permS, sgnS, cS = pair_units(src, T, L - 1) if src is not T else (torch.arange(width, device=dev), torch.ones(width, dtype=torch.float64, device=dev), 1.0)
Wt = T.layers[L - 1].weight.detach()[:, permS] * sgnS[None]; bt = T.layers[L - 1].bias.detach()
Ut = torch.linalg.pinv(Wt)
print(f"  pairing: guess layer-{L} <-> teacher min|corr| {cG:.4f}; prefix layer-{L-1} <-> teacher min|corr| {cS:.4f}", flush=True)
def derr(u, v):
    u = u / u.norm(); v = v / v.norm(); return float(torch.minimum((u - v).norm(), (u + v).norm()))
# ---- row space of W_L from interior Jacobians -----------------------------------------------------------
X0 = torch.randn(40, dims[0], dtype=torch.float64, device=dev) * 0.3
J0 = jac_a_at_x(X0).reshape(-1, width); S0, Vh0 = torch.linalg.svd(J0, full_matrices=False)[1:]
r = int((S0 > 1e-7 * S0[0]).sum()); VS = Vh0[:r]
print(f"  rowspace(W{L}): rank {r}/{width} (sv[r-1]/sv0 {float(S0[r-1]/S0[0]):.1e}) | queries {bb.n}", flush=True)
# ---- duals ----------------------------------------------------------------------------------------------
sel = range(nL) if len(sys.argv) <= 8 else range(int(sys.argv[8].split("-")[0]), int(sys.argv[8].split("-")[1]) + 1)
U = torch.zeros(width, nL, dtype=torch.float64, device=dev); info = {}; t0 = time.time()
Wg_src = None
if PROBE == "corner":
    pg, sg_, _ = pair_units(G, src, L - 1); Wg_src = torch.zeros_like(G.layers[L - 1].weight).detach(); Wg_src[:, pg] = G.layers[L - 1].weight.detach() * sg_[None]
for j in sel:
    q0 = bb.n; jt = int(permG[j])
    if PROBE == "corner":
        wg = Wg_src[j]; s = torch.sign(wg)
        rr = torch.rand(NPTS, width, dtype=torch.float64, device=dev) * 0.8 + 0.2
        A = (s[None] * (1 - (BUDGET * rr / (width * 0.6 * wg.abs()[None])).clamp(0.02, 1.2))).clamp(-0.98, 0.98)
        J = jac_a_corner(A); npr = len(A)
    else:
        X = ascent_probes(j, NPTS); npr = len(X)
        if npr < 3:
            info[j] = dict(dual=None, teacher_unit=jt, probes=npr); print(f"  neuron {j:3d}: only {npr} usable probes", flush=True); continue
        A = a_of_x(X); J = jac_a_at_x(X); zg_min = float((preacts(G, X)[L - 1][:, j]).abs().min())
    Jp = J.reshape(-1, width) @ VS.T; Sg, Vh = torch.linalg.svd(Jp, full_matrices=False)[1:]
    u = VS.T @ Vh[-1]; U[:, j] = u
    zt = A @ Wt.T + bt[None]; zj = float(zt[:, jt].abs().min()); others = zt.abs().clone(); others[:, jt] = 0
    e = derr(u, Ut[:, jt])
    info[j] = dict(dual=u.cpu(), gap=float(Sg[-1] / Sg[-2]), teacher_unit=jt, dual_err=e, zmin=zj, probes=npr, queries=bb.n - q0, amp=(last_amp if PROBE != "corner" else None))
    if j < 3 or j % 25 == 0 or e > 1e-5:
        print(f"  neuron {j:3d} (teacher {jt:3d}): probes {npr:3d} | min|z_j| {zj:5.1f} | others max|z| {float(others.max()):4.1f} | sv_last/sv_prev {float(Sg[-1]/Sg[-2]):.1e}{'' if PROBE == 'corner' else f' | pinv amp {last_amp:.0f}'} | DUAL dir err {e:.2e} | {bb.n-q0} q | {time.time()-t0:.0f}s", flush=True)
good = [j for j in sel if info[j].get("dual") is not None]
de = torch.tensor([info[j]["dual_err"] for j in good])
print(f"DUALS {len(good)}/{len(sel)}: dir err median {float(de.median()):.2e} max {float(de.max()):.2e} | {time.time()-t0:.0f}s | queries {bb.n}", flush=True)
out = dict(U=U.cpu(), info=info, dims=dims, L=L, prefix=prefix_src, probe=PROBE, permG=permG.cpu(), permS=permS.cpu(), sgnS=sgnS.cpu())
if len(good) == nL:
    West = torch.linalg.pinv(U).T.T
    perr = torch.tensor([derr(West[j], Wt[info[j]["teacher_unit"]]) for j in sel])
    print(f"PRIMAL rows from all duals: dir err median {float(perr.median()):.2e} max {float(perr.max()):.2e}", flush=True)
    out["West"] = West.cpu(); torch.save(out, out_path)
    if PROBE == "corner":
        print("tail-decay feasibility along the dual (LP, 0.995-box): neuron | kappa=|w.u| | max z_top reachable", flush=True)
        for j in list(sel)[:8]:
            u = (U[:, j] / U[:, j].norm()).cpu().numpy(); w = (West[j] / West[j].norm() * Wg_src[j].norm()).cpu().numpy(); b = float(G.layers[L - 1].bias[j])
            kappa = abs(float(np.dot(w, u))); n = width; c = np.zeros(n + 1); c[-1] = -1.0
            A_ub = np.zeros((2 * n, n + 1)); b_ub = np.full(2 * n, 0.995); A_ub[:n, :n] = np.eye(n); A_ub[:n, -1] = u; A_ub[n:, :n] = -np.eye(n); A_ub[n:, -1] = -u
            A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = w; res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.array([-b]), bounds=[(-0.995, 0.995)] * n + [(0, None)], method="highs")
            print(f"  {j:3d} | kappa {kappa:.4f} | z_top {kappa * float(res.x[-1]) if res.success else float('nan'):.2f}", flush=True)
torch.save(out, out_path); print(f"total queries {bb.n}", flush=True)
