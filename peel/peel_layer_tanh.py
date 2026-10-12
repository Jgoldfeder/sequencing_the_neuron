"""Layer-l peel for a smooth (tanh/sigmoid) MLP: layers 1..l-1 known exactly,
a coarse guess of layer l, NOTHING about layers l+1.. except that layer l+1 is
narrower than layer l (its width K is read off a spectrum or passed).

Criterion (see l2_full_tanh.py): with the true layer l,
    G(p) = J_h(p) pinv(W_l) diag(1/act'(W_l h + b_l))  =  d(out)/d(h_l) = Psi W_{l+1}
so rows of G over many points span only rowspace(W_{l+1}) (rank K). Minimise
the energy outside the top-K right singular subspace (Golub-Pereyra VarPro, LM).

Queries: the black box depends on x only through z_{l-1} (pre-activation of
layer l-1), so any preimage will do. FD steps in z_{l-1} are realised by
walking the exact known prefix backwards (min-norm linear inverse + atanh per
layer) from a base point.

env: LAYER (1-indexed hidden layer, default 3), ACT tanh|sigmoid, NOISE,
     NB points, K (0 = detect), PREFIX path to a .pt dict with the recovered
     prefix (default: exact teacher), SEED.
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
L = int(E("LAYER", 3))                       # layer being recovered (1-indexed)
NOISE = float(E("NOISE", 1e-2))
NB = int(E("NB", 400))
ITERS = int(E("ITERS", 40))
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

# ---- KNOWN prefix: layers 1..L-1 (exact teacher unless PREFIX given) -----------
Wk = [T.layers[i].weight.detach().clone() for i in range(L - 1)]
bk = [T.layers[i].bias.detach().clone() for i in range(L - 1)]
if E("PREFIX"):
    P = torch.load(E("PREFIX"), map_location=dev)
    for i in range(L - 1):
        Wk[i], bk[i] = P[f"W{i+1}"].double().to(dev), P[f"b{i+1}"].double().to(dev)
    print(f"prefix from {E('PREFIX')}")
Wp = [torch.linalg.pinv(W) for W in Wk]
n_in = Wk[-1].shape[0]                       # width of layer L-1 = input dim of layer L

def prefix(x):
    """z_1..z_{L-1} and h_1..h_{L-2} along the known prefix."""
    zs, hs = [], []
    h = x
    for i in range(L - 1):
        z = h @ Wk[i].T + bk[i]
        zs.append(z); hs.append(h)
        h = act(z)
    return zs, hs                              # hs[i] = input of layer i+1

def x_for_dz(x0, zs, hs, dz):
    """Preimage of z_{L-1}(x0) + dz along the exact prefix: dz (N, m, n_in) ->
    x (N, m, 784). Walk backwards: min-norm inverse per layer, atanh in between."""
    N, m, _ = dz.shape
    z_t = zs[-1][:, None, :] + dz                          # target z_{L-1}
    for i in range(L - 2, -1, -1):                         # layers L-1 .. 1 (0-indexed i)
        h_t = hs[i][:, None, :] + (z_t - zs[i][:, None, :]) @ Wp[i].T   # input of layer i+1
        if i == 0:
            return h_t                                     # = x
        z_t = inv_act(h_t)                                 # pre-activation of layer i

def jac_h(X, h=1e-3):
    """Jacobian of the black box w.r.t. h_{L-1} = act(z_{L-1}) at base inputs X.
    FD in z_{L-1} (4-point stencil), divided by act'(z)."""
    zs, hs = prefix(X)
    N = len(X)
    Eye = torch.eye(n_in, dtype=torch.float64, device=dev)[None].expand(N, -1, -1)
    f = [bb(x_for_dz(X, zs, hs, s * h * Eye).reshape(-1, 784)).reshape(N, n_in, -1)
         for s in (1.0, -1.0, 2.0, -2.0)]
    Jz = (8 * (f[0] - f[1]) - (f[2] - f[3])) / (12 * h)   # (N, n_in, 10)
    a = act(zs[-1])
    return (Jz / dact_a(a)[:, :, None]).transpose(1, 2), a  # (N, 10, n_in), h_{L-1}

def base_points(n, scale=0.9, margin=0.97):
    """Base inputs: h_1 uniform in the (scaled) cube, mapped to x through the
    exact layer 1; keep those whose deeper prefix activations are interior (so
    the FD walk stays inside every cube). Returns X (n, 784)."""
    keep = []
    while sum(len(k) for k in keep) < n:
        if ACT == "tanh":
            h1 = (torch.rand(4 * n, Wk[0].shape[0], dtype=torch.float64, device=dev) * 2 - 1) * scale
        else:
            h1 = torch.rand(4 * n, Wk[0].shape[0], dtype=torch.float64, device=dev) * scale + (1 - scale) / 2
        X = (inv_act(h1) - bk[0]) @ Wp[0].T
        zs, _ = prefix(X)
        ok = torch.ones(len(X), dtype=torch.bool, device=dev)
        for z in zs[1:]:
            a = act(z)
            ok &= (a.abs().max(1).values < margin) if ACT == "tanh" else \
                  ((a > 1 - margin) & (a < margin)).all(1)
        keep.append(X[ok])
        if len(X[ok]) == 0:
            raise RuntimeError("no interior base points; lower SCALE")
    return torch.cat(keep)[:n]

# ---- TRUTH (scoring only) ----------------------------------------------------------
Wt = T.layers[L - 1].weight.detach(); bt = T.layers[L - 1].bias.detach()
n_out = Wt.shape[0]
def score(We, be, tag):
    Wn = We / We.norm(dim=1, keepdim=True); Tn = Wt / Wt.norm(dim=1, keepdim=True)
    cos = (Wn @ Tn.T).cpu().numpy()
    ri, ci = linear_sum_assignment(-abs(cos))
    perm = torch.as_tensor(ci, device=dev); sg = torch.sign(torch.as_tensor(cos[ri, ci], device=dev))
    We = (sg[:, None] * We)[torch.argsort(perm)]; be = (sg * be)[torch.argsort(perm)]
    rel = (We - Wt).norm(dim=1) / Wt.norm(dim=1)
    print(f"[L{L} {tag}] row rel err med {float(rel.median()):.2e} max {float(rel.max()):.2e} | "
          f"bias abs max {float((be - bt).abs().max()):.2e} | q {bb.n/1e6:.2f}M | {time.time()-t00:.0f}s",
          flush=True)

# ---- the guess ----------------------------------------------------------------------
Nz = torch.randn_like(Wt)
Wg = Wt + NOISE * Wt.norm(dim=1, keepdim=True) * Nz / Nz.norm(dim=1, keepdim=True)
nb = torch.randn_like(bt)
bg = bt + NOISE * bt.norm() * nb / nb.norm()
score(Wg, bg, "guess")

# ---- output layer: linear least squares on the known prefix ----------------------------
if L == len(arch) - 1:
    X0 = base_points(NB, scale=float(E("SCALE", 0.9)))
    Hh = act(prefix(X0)[0][-1]); Y = bb(X0)
    A = torch.cat([Hh, torch.ones(len(Hh), 1, dtype=torch.float64, device=dev)], 1)
    sol = torch.linalg.lstsq(A, Y).solution
    We, bs = sol[:-1].T, sol[-1]
    rel = (We - Wt).norm(dim=1) / Wt.norm(dim=1)
    print(f"[L{L} recovered (LSQ)] row rel err med {float(rel.median()):.2e} max {float(rel.max()):.2e} | "
          f"bias abs max {float((bs - bt).abs().max()):.2e} | q {bb.n/1e6:.2f}M", flush=True)
    if E("SAVE"):
        d = dict(torch.load(E("PREFIX"), map_location="cpu")) if E("PREFIX") else {}
        d[f"W{L}"], d[f"b{L}"] = We.cpu(), bs.cpu()
        torch.save(d, E("SAVE"))
        # end-to-end check of the recovered net
        Wf = [d[f"W{i+1}"].to(dev) for i in range(L)]; bf = [d[f"b{i+1}"].to(dev) for i in range(L)]
        Xt = torch.randn(4096, 784, dtype=torch.float64, device=dev)
        h = Xt
        for i in range(L):
            h = h @ Wf[i].T + bf[i]
            if i < L - 1: h = act(h)
        Yt = bb(Xt)
        print(f"END-TO-END recovered net vs black box on fresh N(0,1) inputs: max |dy| {float((h - Yt).abs().max()):.2e}, "
              f"rel fro {float((h - Yt).norm() / Yt.norm()):.2e}", flush=True)
    sys.exit(0)

# ---- measurements ---------------------------------------------------------------------
X0 = base_points(NB, scale=float(E("SCALE", 0.9)))
zs0, _ = prefix(X0)
print(f"base points: |z_{L-1}| max {float(zs0[-1].abs().max()):.2f}, guess |z_{L}| median "
      f"{float((act(zs0[-1]) @ Wg.T + bg).abs().median()):.2f}", flush=True)
J, H = jac_h(X0)                                          # (N,10,n_in), (N,n_in)
Sg, Vh = torch.linalg.svd(J.reshape(-1, n_in), full_matrices=False)[1:]
r = int((Sg > 1e-7 * Sg[0]).sum())
VS = Vh[:r]                                                # rowspace(W_L), r <= n_in
print(f"rowspace(W{L}): rank {r} of {n_in} (sv[{r-1}]/sv0 {float(Sg[r-1]/Sg[0]):.1e}, "
      f"next {float(Sg[r]/Sg[0]) if r < len(Sg) else 0:.1e}) | q {bb.n/1e6:.2f}M", flush=True)
assert r == n_out, (r, n_out)
JS = J @ VS.T                                              # (N, 10, r)
hS = H @ VS.T                                              # (N, r)

def Gmat(Ws, b):
    C = JS @ torch.linalg.inv(Ws)                          # (N, 10, r) = G D
    z = hS @ Ws.T + b
    M = (C * inv_dact_z(z)[:, None, :]).reshape(-1, n_out)
    M = M / M.norm(dim=0, keepdim=True)
    return M / M.norm()

K = int(E("K", 0)) or arch[L + 1]          # width of the next layer (spectrum gap at the guess is unreliable)
svt = torch.linalg.svdvals(Gmat(Wt @ VS.T, bt))          # diagnostic only (uses truth)
print(f"spectrum at TRUTH (diagnostic): sv[{K-1}]/sv0 {float(svt[K-1]/svt[0]):.1e}, sv[{K}]/sv0 {float(svt[K]/svt[0]):.1e}", flush=True)
sv = torch.linalg.svdvals(Gmat(Wg @ VS.T, bg))
print(f"spectrum at guess: sv[{K-1}]/sv0 {float(sv[K-1]/sv[0]):.1e}, sv[{K}]/sv0 "
      f"{float(sv[K]/sv[0]):.1e} -> K = {K} (layer {L+1} width {arch[L+1]})", flush=True)

def proj_resid(M):
    ev, V = torch.linalg.eigh(M.T @ M)
    return (M @ V[:, :n_out - K]).reshape(-1)

def unpack(th):
    return th[:n_out * r].view(n_out, r), th[n_out * r:]
def resid_base(th):
    return proj_resid(Gmat(*unpack(th)))

PRIOR = float(E("PRIOR", 0))
def lm(theta, iters):
    lam = 1e-3
    theta_g = theta.clone()
    mu = 0.0
    def resid(th):
        r = resid_base(th)
        return r if mu == 0 else torch.cat([r, math.sqrt(mu) * (th - theta_g)])
    for it in range(iters):
        if PRIOR and it == 0:
            r0 = resid(theta); mu = PRIOR * float(r0 @ r0) / (NOISE ** 2 * float(theta_g.norm()) ** 2)
        rr = resid(theta); loss = float(rr @ rr)
        eye = torch.eye(len(theta), dtype=torch.float64, device=dev)
        Jm = torch.cat([vmap(lambda v: jvp(resid, (theta,), (v,))[1])(eye[i:i + 128])
                        for i in range(0, len(theta), 128)]).T
        dsc = float(Jm.norm(dim=0).mean())
        ok = False
        for _ in range(12):
            Ja = torch.cat([Jm, math.sqrt(lam) * dsc * eye])
            ra = torch.cat([-rr, torch.zeros_like(theta)])
            step = torch.linalg.lstsq(Ja, ra[:, None]).solution[:, 0]
            if float(step.norm()) > 0.3 * float(theta.norm()):
                lam *= 10; continue
            cand = theta + step
            rc = resid(cand)
            if torch.isfinite(rc).all() and float(rc @ rc) < loss:
                theta, ok = cand, True
                lam = max(lam / 5, 1e-12)
                mu *= 0.3                                   # anneal the prior away
                break
            lam *= 10
        print(f"  it {it+1:2d} loss {loss:.3e} -> {float(rc @ rc):.3e} | lam {lam:.0e} | "
              f"|step| {float(step.norm()):.2e} {'ok' if ok else 'REJECT'}", flush=True)
        if not ok or float(step.norm()) < 1e-13:
            break
    return theta

th = lm(torch.cat([(Wg @ VS.T).reshape(-1), bg]), ITERS)
Ws, bs = unpack(th)
We = Ws @ VS
score(We, bs, "recovered")
sv = torch.linalg.svdvals(Gmat(Ws, bs))
print(f"final spectrum: sv[{K-1}]/sv0 {float(sv[K-1]/sv[0]):.1e}, sv[{K}]/sv0 {float(sv[K]/sv[0]):.1e}")
print(f"total queries {bb.n}")
if E("SAVE"):
    d = dict(torch.load(E("PREFIX"), map_location="cpu")) if E("PREFIX") else {}
    for i in range(L - 1):
        d[f"W{i+1}"], d[f"b{i+1}"] = Wk[i].cpu(), bk[i].cpu()
    d[f"W{L}"], d[f"b{L}"] = We.cpu(), bs.cpu()
    torch.save(d, E("SAVE")); print(f"saved {E('SAVE')}")
