"""Recover ALL of layer 2 (W2, b2) of a tanh MLP, given layer 1 exactly and a
~1e-2 (works from 1e-1) guess of layer 2. Nothing about layers 3+ is used except that layer 3 is
NARROWER than layer 2 (its width is read off a spectrum, not assumed).

Stage A  directions : self-saturation duals (see l2_neuron_tanh.py) -> row
                      directions of W2.
Stage B  scale+bias : peel criterion. With the true layer 2,
                          G(a) = J_a(a) pinv(W2) diag(1/sech^2(W2 a + b2))
                      is d(out)/d(h2) = Psi(a) W3, so the rows of G over many
                      points span only rowspace(W3) (dim = width of layer 3).
                      Wrong scale/bias rescales the columns of G point by
                      point and destroys that rank. Minimise the energy
                      outside the top-k right singular subspace.
Stage C  polish     : same criterion, all of W2 (inside its row space) + b2.

RESULT (2026-09-29): stage C alone, straight from the guess, reaches 2e-11 row
rel err / 7e-12 bias from 1e-2, 5e-2 and 1e-1 guesses with 225k queries; stages
A/B are optional (SKIPA=0) and only reach 6e-8. Per-column normalisation of G
is required (else a 1e-1 guess finds the trivial 'blow up z_j' minimiser).
Golub-Pereyra (projector differentiated) converges quadratically; a frozen
projector (Kaufman) only linearly at ~6%/step.
"""
import os, sys, math, time, torch
from torch.func import jvp, vmap
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nets import MLP
from tanh_solve import Oracle

torch.manual_seed(int(os.environ.get("SEED", 0)))
dev = "cuda"
ACT = os.environ.get("ACT", "tanh")            # tanh | sigmoid
INIT = os.environ.get("INIT", "guess")         # guess | random  (random = NO guess at all)
arch = [784, 128, 80, 40, 32, 10]
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    f"teachers/teacher_784x128x80x40x32x10_e25_s0_{ACT}.pt")
if ACT == "tanh":
    inv_act = torch.atanh
    dact = lambda a: 1 - a * a                    # da/dz as a function of a
    inv_dact_z = lambda z: torch.cosh(z) ** 2     # 1/act'(z)
    interior = lambda n: (torch.rand(n, 128, dtype=torch.float64, device=dev) * 2 - 1) * 0.9
else:
    inv_act = torch.logit
    dact = lambda a: a * (1 - a)
    inv_dact_z = lambda z: 1.0 / (torch.sigmoid(z) * (1 - torch.sigmoid(z)))
    interior = lambda n: torch.rand(n, 128, dtype=torch.float64, device=dev) * 0.9 + 0.05
NOISE = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 400        # stage B/C points
CIT = int(sys.argv[3]) if len(sys.argv) > 3 else 15
SKIPA = int(sys.argv[4]) if len(sys.argv) > 4 else 1       # 1 (default) = peel criterion straight from the raw guess; 0 = also run stages A/B
t00 = time.time()

T = MLP(arch, act=ACT).to(dev, torch.float64)
T.load_state_dict({k: v.double() for k, v in torch.load(path, map_location="cpu").items()})
bb = Oracle(lambda x: T(x))

# ---- KNOWN: layer 1 -------------------------------------------------------
W1 = T.layers[0].weight.detach(); b1 = T.layers[0].bias.detach()
W1p = torch.linalg.pinv(W1)
n1 = W1.shape[0]

def g_of_z(Z):
    return bb((Z - b1) @ W1p.T)

def jac_a(A, h=1e-3, chunk=64):
    out = []
    E = torch.eye(n1, dtype=torch.float64, device=dev)
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]; Z = inv_act(a); N = len(Z)
        f = [g_of_z((Z[:, None, :] + s * h * E[None]).reshape(-1, n1)).reshape(N, n1, -1)
             for s in (1.0, -1.0, 2.0, -2.0)]
        Jz = (8 * (f[0] - f[1]) - (f[2] - f[3])) / (12 * h)
        out.append((Jz / dact(a)[:, :, None]).transpose(1, 2))
    return torch.cat(out)                                         # (N, 10, 128)

# ---- TRUTH: scoring only --------------------------------------------------
W2 = T.layers[1].weight.detach(); b2 = T.layers[1].bias.detach()
n2 = W2.shape[0]
from scipy.optimize import linear_sum_assignment
def score(We, be, tag):
    Wn = We / We.norm(dim=1, keepdim=True); Tn = W2 / W2.norm(dim=1, keepdim=True)
    cos = (Wn @ Tn.T).cpu().numpy()
    ri, ci = linear_sum_assignment(-abs(cos))          # match est row ri -> true row ci
    perm = torch.as_tensor(ci, device=dev); sg = torch.sign(torch.as_tensor(cos[ri, ci], device=dev))
    We = (sg[:, None] * We)[torch.argsort(perm)]; be = (sg * be)[torch.argsort(perm)]
    flips = int((sg < 0).sum()); moved = int((perm != torch.arange(n2, device=dev)).sum())
    if flips or moved:
        print(f"  (gauge: {moved} rows permuted, {flips} sign-flipped before scoring)")
    rel = ((We - W2).norm(dim=1) / W2.norm(dim=1))
    d = (We / We.norm(dim=1, keepdim=True) - W2 / W2.norm(dim=1, keepdim=True)).norm(dim=1)
    sc = (We.norm(dim=1) / W2.norm(dim=1) - 1).abs()
    be_ = (be - b2).abs()
    print(f"[{tag}] W2 row rel err med {float(rel.median()):.2e} max {float(rel.max()):.2e} | "
          f"dir max {float(d.max()):.2e} | scale max {float(sc.max()):.2e} | "
          f"bias abs med {float(be_.median()):.2e} max {float(be_.max()):.2e} | "
          f"q {bb.n/1e6:.2f}M | {time.time()-t00:.0f}s", flush=True)

# ---- the guess ------------------------------------------------------------
Nz = torch.randn_like(W2)
W2g = W2 + NOISE * W2.norm(dim=1, keepdim=True) * Nz / Nz.norm(dim=1, keepdim=True)
nb = torch.randn_like(b2)
b2g = b2 + NOISE * b2.norm() * nb / nb.norm()
if INIT == "random":
    W2g = torch.randn_like(W2) * (W2.norm() / math.sqrt(W2.numel()))
    b2g = torch.zeros_like(b2)
score(W2g, b2g, "guess" if INIT == "guess" else "RANDOM init")

# ---- row space of W2 ----------------------------------------------------------
A = interior(40)
Sg, Vh = torch.linalg.svd(jac_a(A).reshape(-1, n1), full_matrices=False)[1:]
r = int((Sg > 1e-7 * Sg[0]).sum())
VS = Vh[:r]
assert r == n2, r

# ---- Stage A: duals -> directions (optional; not needed for convergence) --------
Wdir = W2g / W2g.norm(dim=1, keepdim=True)
if not SKIPA:
  U = torch.zeros(n1, n2, dtype=torch.float64, device=dev)
  for j in range(n2):
      wg = W2g[j]
      rr = torch.rand(30, n1, dtype=torch.float64, device=dev) * 0.8 + 0.2
      delta = (1.5 * rr / (n1 * 0.6 * wg.abs()[None])).clamp(0.02, 1.2)
      Aj = (torch.sign(wg)[None] * (1 - delta)).clamp(-0.98, 0.98)
      Vh = torch.linalg.svd(jac_a(Aj).reshape(-1, n1) @ VS.T, full_matrices=False)[2]
      U[:, j] = VS.T @ Vh[-1]
  Wdir = torch.linalg.pinv(U)                                       # rows ∝ w_j
  Wdir = Wdir / Wdir.norm(dim=1, keepdim=True)
  Wdir = Wdir * torch.sign((Wdir * W2g).sum(1, keepdim=True))       # sign gauge from the guess
  score(Wdir * W2g.norm(dim=1, keepdim=True), b2g, "A dirs ")

# ---- measurements for stages B/C ---------------------------------------------
AB = interior(NB)
SATCAP = float(os.environ.get("SATCAP", 0))          # keep the NB least-saturated of 10*NB candidates (judged by the GUESS)
if SATCAP:
    cand = interior(10 * NB)
    zc = (cand @ W2g.T + b2g).abs().max(1).values
    AB = cand[zc.argsort()[:NB]]
    print(f"saturation filter: max|z_hat| over kept points {float(zc.sort().values[NB-1]):.2f} "
          f"(unfiltered median {float(zc.median()):.2f})", flush=True)
JS = jac_a(AB) @ VS.T                                             # (N, 10, 80) in S coords
aS = AB @ VS.T                                                    # (N, 80): only S part matters
print(f"stage B/C jets: {NB} points | q {bb.n/1e6:.2f}M", flush=True)

def Gmat(Ws, b):
    """Stacked d(out)/d(h2) implied by layer-2 candidate (Ws: 80x80 in S coords)."""
    C = JS @ torch.linalg.inv(Ws)                                  # (N, 10, 80) = G D
    z = aS @ Ws.T + b
    M = C * inv_dact_z(z)[:, None, :]
    M = M.reshape(-1, n2)
    M = M / M.norm(dim=0, keepdim=True)     # per-column: kills the trivial "blow up z_j" minimiser
    return M / M.norm()

def spectrum(Ws, b):
    return torch.linalg.svdvals(Gmat(Ws, b))

def proj_resid(M):
    """M Q with Q = bottom-(80-K) eigenvectors of M^T M, differentiable in M
    (Golub-Pereyra variable projection: the projector moves with theta)."""
    ev, V = torch.linalg.eigh(M.T @ M)
    return (M @ V[:, :n2 - K]).reshape(-1)

def lm(resid, theta, iters, tag):
    """Dense Levenberg-Marquardt; resid(theta, Q) with the projector Q frozen per
    step (variable projection, Kaufman approximation)."""
    lam = 1e-3
    for it in range(iters):
        r = resid(theta)
        loss = float(r @ r)
        f = resid
        eye = torch.eye(len(theta), dtype=torch.float64, device=dev)
        J = torch.cat([vmap(lambda v: jvp(f, (theta,), (v,))[1])(eye[i:i + 128])
                       for i in range(0, len(theta), 128)]).T
        dsc = J.norm(dim=0)
        ok = False
        for _ in range(10):
            # Levenberg lam*I (scaled by the mean column norm), NOT Marquardt lam*diag:
            # diag damping leaves weakly-excited directions undamped -> wild steps (sigmoid)
            Ja = torch.cat([J, math.sqrt(lam) * float(dsc.mean()) * torch.eye(len(theta), dtype=J.dtype, device=dev)])
            ra = torch.cat([-r, torch.zeros_like(theta)])
            step = torch.linalg.lstsq(Ja, ra[:, None]).solution[:, 0]
            cand = theta + step
            if float(step.norm()) > 0.3 * float(theta.norm()):   # trust region
                lam *= 10; continue
            rc = resid(cand)
            if torch.isfinite(rc).all() and float(rc @ rc) < loss:
                theta, ok = cand, True
                lam = max(lam / 5, 1e-12)
                break
            lam *= 10
        print(f"  {tag} it {it+1:2d} loss {loss:.3e} -> {float(rc @ rc):.3e} | lam {lam:.0e} | "
              f"|step| {float(step.norm()):.2e} {'ok' if ok else 'REJECT'}", flush=True)
        if not ok or float(step.norm()) < 1e-13:
            break
    return theta

if os.environ.get("TRUTHSPEC"):                       # diagnostics only
    svt = spectrum(W2 @ VS.T, b2)
    print("spectrum at TRUTH (diagnostic): " + " ".join(f"{float(v/svt[0]):.1e}" for v in svt[36:44]), flush=True)

# ---- read the width of layer 3 off the spectrum at the stage-A estimate --------
WsA = (Wdir * W2g.norm(dim=1, keepdim=True)) @ VS.T
sv = spectrum(WsA, b2g)
gaps = sv[:-1] / sv[1:]
K = int(os.environ.get("K", 0)) or int(gaps[5:70].argmax()) + 6
print(f"spectrum of stacked G at stage-A estimate: sv[{K-1}]/sv0 {float(sv[K-1]/sv[0]):.1e}, "
      f"sv[{K}]/sv0 {float(sv[K]/sv[0]):.1e} -> layer-3 width K = {K}", flush=True)

# ---- Stage B: scale kappa_j and bias b_j (160 params), directions frozen ---------
WsB, bB = W2g @ VS.T, b2g
if not SKIPA:
  Wd = Wdir @ VS.T                                                   # unit rows, S coords
  def unpackB(th):
      return th[:n2, None] * Wd, th[n2:]
  def resB(th):
      return proj_resid(Gmat(*unpackB(th)))
  thB = lm(resB, torch.cat([W2g.norm(dim=1), b2g]), 30, "B")
  WsB, bB = unpackB(thB)
  score(WsB @ VS, bB, "B scale+bias")

# ---- Stage C: everything (80x80 + 80) ---------------------------------------------
def unpackC(th):
    return th[:n2 * n2].view(n2, n2), th[n2 * n2:]
sub = torch.randperm(NB, device=dev)[:min(NB, 120)]               # dense J memory cap
JS_all, aS_all = JS, aS
JS, aS = JS_all[sub], aS_all[sub]
def resC(th):
    return proj_resid(Gmat(*unpackC(th)))
thC = lm(resC, torch.cat([WsB.reshape(-1), bB]), CIT, "C")
WsC, bC = unpackC(thC)
score(WsC @ VS, bC, "C polish")
sv = spectrum(WsC, bC)
print(f"final spectrum: sv[{K-1}]/sv0 {float(sv[K-1]/sv[0]):.1e}, sv[{K}]/sv0 {float(sv[K]/sv[0]):.1e}")
print(f"total queries {bb.n}")
