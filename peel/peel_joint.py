"""Joint peel of M consecutive layers L..L+M-1 (layers 1..L-1 known exactly,
coarse guesses of layers L..L+M-1, nothing about deeper layers) with the rank
criterion applied at layer L+M: after undoing the M layers,
    d(out)/d(h_{L+M-1})  stacked over points has rank = width(L+M).
This removes the requirement that layer L+1 be narrower than layer L: only
SOME layer below must narrow (the linear output layer always does). The price
is a joint fit of M layers (guesses for all of them).

Undo of one layer l (n_l <= n_{l-1} required, so W_l has full row rank):
    G_{l-1} = G_l D_l W_l   =>   G_l = G_{l-1} pinv(W_l) / act'(z_l).

env: TEACHER (.pt state dict) + ARCH (comma list) or the default deep tanh
     teacher; LAYER (first peeled layer), JOINT (=M), NOISE, NB, ITERS, ACT.
"""
import os, sys, math, time, torch
from torch.func import jvp, vmap
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nets import MLP
from tanh_solve import Oracle
from scipy.optimize import linear_sum_assignment

E = os.environ.get
torch.manual_seed(int(E("SEED", 0)))
dev = "cuda"
ACT = E("ACT", "tanh")
L = int(E("LAYER", 2)); M = int(E("JOINT", 2))
NOISE = float(E("NOISE", 1e-2)); NB = int(E("NB", 120)); ITERS = int(E("ITERS", 40))
arch = [int(v) for v in E("ARCH", "784,128,80,40,32,10").split(",")]
path = E("TEACHER") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                   f"teachers/teacher_784x128x80x40x32x10_e25_s0_{ACT}.pt")
if ACT == "tanh":
    act, inv_act = torch.tanh, torch.atanh
    dact_a = lambda a: 1 - a * a
    inv_dact_z = lambda z: torch.cosh(z) ** 2
else:
    act, inv_act = torch.sigmoid, torch.logit
    dact_a = lambda a: a * (1 - a)
    inv_dact_z = lambda z: 1.0 / (torch.sigmoid(z) * (1 - torch.sigmoid(z)))
t00 = time.time()

T = MLP(arch, act=ACT).to(dev, torch.float64)
T.load_state_dict({k: v.double() for k, v in torch.load(path, map_location="cpu").items()})
bb = Oracle(lambda x: T(x))
d_in = arch[0]

# ---- known prefix: layers 1..L-1 -------------------------------------------------
Wk = [T.layers[i].weight.detach().clone() for i in range(L - 1)]
bk = [T.layers[i].bias.detach().clone() for i in range(L - 1)]
Wp = [torch.linalg.pinv(W) for W in Wk]
n_in = Wk[-1].shape[0]

def prefix(x):
    zs, hs = [], []
    h = x
    for i in range(L - 1):
        z = h @ Wk[i].T + bk[i]
        zs.append(z); hs.append(h)
        h = act(z)
    return zs, hs

def x_for_dz(x0, zs, hs, dz):
    z_t = zs[-1][:, None, :] + dz
    for i in range(L - 2, -1, -1):
        h_t = hs[i][:, None, :] + (z_t - zs[i][:, None, :]) @ Wp[i].T
        if i == 0:
            return h_t
        z_t = inv_act(h_t)

def jac_h(X, h=1e-3):
    zs, hs = prefix(X)
    N = len(X)
    Eye = torch.eye(n_in, dtype=torch.float64, device=dev)[None].expand(N, -1, -1)
    f = [bb(x_for_dz(X, zs, hs, s * h * Eye).reshape(-1, d_in)).reshape(N, n_in, -1)
         for s in (1.0, -1.0, 2.0, -2.0)]
    Jz = (8 * (f[0] - f[1]) - (f[2] - f[3])) / (12 * h)
    a = act(zs[-1])
    return (Jz / dact_a(a)[:, :, None]).transpose(1, 2), a

def base_points(n, scale=0.9, margin=0.97):
    keep = []
    while sum(len(k) for k in keep) < n:
        if ACT == "tanh":
            h1 = (torch.rand(4 * n, arch[1], dtype=torch.float64, device=dev) * 2 - 1) * scale
        else:
            h1 = torch.rand(4 * n, arch[1], dtype=torch.float64, device=dev) * scale + (1 - scale) / 2
        X = (inv_act(h1) - bk[0]) @ Wp[0].T
        zs, _ = prefix(X)
        ok = torch.ones(len(X), dtype=torch.bool, device=dev)
        for z in zs[1:]:
            a = act(z)
            ok &= (a.abs().max(1).values < margin) if ACT == "tanh" else ((a > 1 - margin) & (a < margin)).all(1)
        keep.append(X[ok])
        if len(X[ok]) == 0:
            raise RuntimeError("no interior base points; lower SCALE")
    return torch.cat(keep)[:n]

# ---- truth (scoring only) + guesses ------------------------------------------------
Wt = [T.layers[L - 1 + m].weight.detach() for m in range(M)]
bt = [T.layers[L - 1 + m].bias.detach() for m in range(M)]
widths = [W.shape[0] for W in Wt]
for m in range(M):
    assert Wt[m].shape[0] <= Wt[m].shape[1], f"layer {L+m} widens ({Wt[m].shape}); undo is not unique"
K = int(E("K", 0)) or arch[L + M]
assert K < widths[-1], f"rank criterion vacuous: width(L+M)={K} >= width(L+M-1)={widths[-1]}"

def score(We, be, m, tag):
    W0, b0 = Wt[m], bt[m]
    Wn = We / We.norm(dim=1, keepdim=True); Tn = W0 / W0.norm(dim=1, keepdim=True)
    cos = (Wn @ Tn.T).cpu().numpy()
    ri, ci = linear_sum_assignment(-abs(cos))
    perm = torch.as_tensor(ci, device=dev); sg = torch.sign(torch.as_tensor(cos[ri, ci], device=dev))
    We = (sg[:, None] * We)[torch.argsort(perm)]; be = (sg * be)[torch.argsort(perm)]
    rel = (We - W0).norm(dim=1) / W0.norm(dim=1)
    print(f"[L{L+m} {tag}] row rel err med {float(rel.median()):.2e} max {float(rel.max()):.2e} | "
          f"bias abs max {float((be - b0).abs().max()):.2e} | q {bb.n/1e6:.2f}M | {time.time()-t00:.0f}s", flush=True)

Wg, bg = [], []
for m in range(M):
    Nz = torch.randn_like(Wt[m]); nb = torch.randn_like(bt[m])
    Wg.append(Wt[m] + NOISE * Wt[m].norm(dim=1, keepdim=True) * Nz / Nz.norm(dim=1, keepdim=True))
    bg.append(bt[m] + NOISE * max(float(bt[m].norm()), 0.1) * nb / nb.norm())
    score(Wg[m], bg[m], m, "guess")

# ---- measurements ------------------------------------------------------------------
X0 = base_points(NB * int(E("SATCAP", 10)), scale=float(E("SCALE", 0.9)))
# keep the NB least-saturated candidates across the peeled layers (judged by the GUESSES):
# every undo multiplies FD noise by cosh^2(z_l), and that compounds over the M layers
with torch.no_grad():
    hh = act(prefix(X0)[0][-1]); zmax = torch.zeros(len(X0), dtype=torch.float64, device=dev)
    for m in range(M):
        zz = hh @ Wg[m].T + bg[m]; zmax = torch.maximum(zmax, zz.abs().max(1).values); hh = act(zz)
    order = zmax.argsort()
    print(f"saturation filter: max|z_hat| over peeled layers kept <= {float(zmax[order[NB-1]]):.2f} "
          f"(median of candidates {float(zmax.median()):.2f})", flush=True)
    X0 = X0[order[:NB]]
J, H = jac_h(X0)
Sg, Vh = torch.linalg.svd(J.reshape(-1, n_in), full_matrices=False)[1:]
r = int((Sg > 1e-7 * Sg[0]).sum())
VS = Vh[:r]
print(f"rowspace(W{L}): rank {r} of {n_in} | q {bb.n/1e6:.2f}M | {NB} points", flush=True)
assert r == widths[0], (r, widths[0])
JS = J @ VS.T; hS = H @ VS.T

# parameter layout: layer L in S coords (r x r) + b; layers L+1.. full (n_out x n_in) + b
shapes = [(widths[0], r)] + [(widths[m], widths[m - 1]) for m in range(1, M)]
sizes = [s[0] * s[1] + s[0] for s in shapes]
def unpack(th):
    Ws, bs, i = [], [], 0
    for (o, n), sz in zip(shapes, sizes):
        Ws.append(th[i:i + o * n].view(o, n)); bs.append(th[i + o * n:i + sz]); i += sz
    return Ws, bs

ZCAP = float(E("ZCAP", 0)) or 1.5 * float(zmax[order[NB - 1]])   # true |z| at kept points is below this
def Gmat(Ws, bs, with_barrier=False):
    """Undo layers L..L+M-1: returns stacked d(out)/d(h_{L+M-1}), column-normalised.
    with_barrier: also return relu(|z|-ZCAP) over all undone pre-activations -- a
    soft barrier that is exactly zero at the truth and removes the degenerate
    minimiser 'saturate a few points so their rows dominate every column'."""
    G = JS; h = hS; bar = []
    for m in range(M):
        C = G @ torch.linalg.pinv(Ws[m])                     # (N, 10, n_m) = G_m D_m
        z = h @ Ws[m].T + bs[m]
        bar.append(torch.relu(z.abs() - ZCAP).reshape(-1))
        G = C * inv_dact_z(z)[:, None, :]
        h = act(z)
    Mx = G.reshape(-1, widths[-1])
    Mx = Mx / Mx.norm(dim=0, keepdim=True)
    Mx = Mx / Mx.norm()
    return (Mx, torch.cat(bar)) if with_barrier else Mx

RANKW = float(E("RANKW", 1.0)); rank_floor = None
def proj_resid(Mx):
    """Energy outside the top-K right singular subspace, plus a guard that the
    K-th singular value stays above a floor (set from the guess): forbids the
    degenerate minimisers where the stack collapses to rank < K."""
    ev, V = torch.linalg.eigh(Mx.T @ Mx)
    r = (Mx @ V[:, :widths[-1] - K]).reshape(-1)
    if rank_floor is None:
        return r
    sK = torch.sqrt(ev[-K].clamp_min(0) / ev[-1])
    return torch.cat([r, RANKW * torch.relu(rank_floor - sK)[None]])
BARW = float(E("BARW", 1.0))
def resid(th):
    Ws, bs = unpack(th)
    Mx, bar = Gmat(Ws, bs, with_barrier=True)
    return torch.cat([proj_resid(Mx), BARW * bar])

Wt_S = [Wt[0] @ VS.T] + Wt[1:]
svt = torch.linalg.svdvals(Gmat(Wt_S, bt))
print(f"spectrum at TRUTH (diag): sv[{K-1}]/sv0 {float(svt[K-1]/svt[0]):.1e}, sv[{K}]/sv0 {float(svt[K]/svt[0]):.1e}", flush=True)
th0 = torch.cat([(Wg[0] @ VS.T).reshape(-1), bg[0]] + sum([[Wg[m].reshape(-1), bg[m]] for m in range(1, M)], []))
sv = torch.linalg.svdvals(Gmat(*unpack(th0)))
rank_floor = 0.3 * float(sv[K - 1] / sv[0])        # truth is well above this (checked: pair 0.63x, trunk 2.9x of guess)
print(f"rank guard: sv[{K-1}]/sv0 must stay >= {rank_floor:.2e}", flush=True)
print(f"z barrier at |z| > {ZCAP:.2f} | spectrum at guess: sv[{K-1}]/sv0 {float(sv[K-1]/sv[0]):.1e}, sv[{K}]/sv0 {float(sv[K]/sv[0]):.1e} | "
      f"{len(th0)} params, {NB * arch[-1] * (widths[-1] - K)} residuals", flush=True)

from torch.func import vjp
SOLVER = E("SOLVER", "dense")                     # dense (Cholesky on J^T J) | cg (matrix-free, any NB)
CGIT = int(E("CGIT", 400))

def _cg(Aop, b, iters, tol=1e-4):
    x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rr = r @ r; b0 = math.sqrt(float(rr))
    for k in range(iters):
        Ap = Aop(p); a = rr / (p @ Ap); x += a * p; r -= a * Ap
        rn = r @ r
        if math.sqrt(float(rn)) <= tol * b0:
            return x, k + 1
        p = r + (rn / rr) * p; rr = rn
    return x, iters

def lm(theta, iters):
    lam = 1e-3
    for it in range(iters):
        rr = resid(theta); loss = float(rr @ rr)
        ok = False
        if SOLVER == "cg":
            _, vjp_fn = vjp(resid, theta)
            gm = vjp_fn(rr)[0]
            JtJ = lambda v: vjp_fn(jvp(resid, (theta,), (v,))[1])[0]
            vprobe = torch.randn_like(theta); dsc = math.sqrt(float(vprobe @ JtJ(vprobe)) / len(theta))
        else:
            eye = torch.eye(len(theta), dtype=torch.float64, device=dev)
            Jm = torch.cat([vmap(lambda v: jvp(resid, (theta,), (v,))[1])(eye[i:i + 64])
                            for i in range(0, len(theta), 64)]).T
            dsc = float(Jm.norm(dim=0).mean())
            Hm = Jm.T @ Jm; gm = Jm.T @ rr                  # normal equations (memory: J is the big one)
            del Jm
        for _ in range(12):
            if SOLVER == "cg":
                step, ncg = _cg(lambda v: JtJ(v) + lam * dsc ** 2 * v, -gm, CGIT)
            else:
                Lc, info = torch.linalg.cholesky_ex(Hm + lam * dsc ** 2 * eye)
                step = -torch.cholesky_solve(gm[:, None], Lc)[:, 0]
            if float(step.norm()) > 0.15 * float(theta.norm()):
                lam *= 10; continue
            cand = theta + step
            rc = resid(cand)
            if torch.isfinite(rc).all() and float(rc @ rc) < loss:
                theta, ok = cand, True
                lam = max(lam / 5, 1e-12)
                break
            lam *= 10
        print(f"  it {it+1:2d} loss {loss:.3e} -> {float(rc @ rc):.3e} | lam {lam:.0e} | "
              f"|step| {float(step.norm()):.2e} {'ok' if ok else 'REJECT'} | {time.time()-t00:.0f}s", flush=True)
        if not ok or float(step.norm()) < 1e-13:
            break
    return theta

th = lm(th0, ITERS)
Ws, bs = unpack(th)
We = [Ws[0] @ VS] + Ws[1:]
for m in range(M):
    score(We[m], bs[m], m, "recovered")
sv = torch.linalg.svdvals(Gmat(Ws, bs))
print(f"final spectrum: sv[{K-1}]/sv0 {float(sv[K-1]/sv[0]):.1e}, sv[{K}]/sv0 {float(sv[K]/sv[0]):.1e}")
print(f"total queries {bb.n}")
