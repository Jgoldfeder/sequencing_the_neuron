"""Experiment B: block-preconditioned Gauss-Newton on the stalled 10-layer snapshot. Same probes, same
LM damping / gain-ratio loop and CG budget as tanh_solve.refine, but CG is preconditioned per layer with
a K-FAC-style Kronecker factorisation of the value-residual curvature: for layer l,
H_l ~ E[B_l^T B_l] (x) E[h~_{l-1} h~_{l-1}^T]  (B_l = d y / d z_l from the STUDENT, h~ = activation with a 1
appended for the bias), M^-1 V = (G_l + a I)^-1 V (A_l + b I)^-1 with factored damping. Layer-1 bias
(weights frozen) uses the G factor only. Usage: python peel/diag/precond_gn.py <gpu> <kfac|jacobi|none> [iters]"""
import sys, os, math, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, Problem, fmt
import tanh_solve as ts
from torch.func import vmap, jacrev
dev = f"cuda:{sys.argv[1]}"; PC = sys.argv[2]; ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
dims, t64, net, bb = load(dev)
X = ts.sample_points(512, dims[0], scales=(0.5, 1.0, 2.0), seed=0, device=dev)
P = Problem(bb, net, X, dirs=64)
L = P.n_layers
tag = f"[pc={PC}]"


def hidden_acts(theta):
    """h_0 = x, h_l = tanh(z_l) for l < L; z_l = h_{l-1} W_l^T + b_l. Returns list of h_{l} (N, width) for l=0..L-1 and z list."""
    a = P.X; hs = [a]; zs = []
    for l in range(L):
        z = a @ theta[f"layers.{l}.weight"].t() + theta[f"layers.{l}.bias"]; zs.append(z)
        if l < L - 1:
            a = torch.tanh(z); hs.append(a)
    return hs, zs


def kfac_factors(theta, lam):
    """Per layer l: (Ginv (out x out), Ainv (in+1 x in+1)) with factored damping; layer 0: Ginv only (bias)."""
    hs, zs = hidden_acts(theta)
    facs = {}
    N = P.X.shape[0]
    for l in range(L):
        # B_l(x) = d y / d z_l : jacrev of the suffix from z_l, batched over points
        def suffix(z, l=l):
            a = torch.tanh(z) if l < L - 1 else z
            for m in range(l + 1, L):
                z2 = a @ theta[f"layers.{m}.weight"].t() + theta[f"layers.{m}.bias"]
                a = torch.tanh(z2) if m < L - 1 else z2
            return a
        B = vmap(jacrev(suffix))(zs[l])                       # (N, dout, out_l)
        G = torch.einsum("nij,nik->jk", B, B) / N                # E[B^T B] (out_l x out_l)
        if l == 0:
            A = None
        else:
            h = torch.cat([hs[l], torch.ones(N, 1, dtype=hs[l].dtype, device=dev)], 1)   # (N, in+1)
            A = h.t() @ h / N
        # factored Tikhonov damping (Martens & Grosse): pi = sqrt((tr A / dA) / (tr G / dG))
        if A is not None:
            pi = math.sqrt((float(A.trace()) / A.shape[0]) / max(float(G.trace()) / G.shape[0], 1e-300))
            a_d = math.sqrt(lam) * pi; g_d = math.sqrt(lam) / pi
            Ainv = torch.linalg.inv(A + a_d * torch.eye(A.shape[0], dtype=A.dtype, device=dev))
        else:
            g_d = lam; Ainv = None
        Ginv = torch.linalg.inv(G + g_d * torch.eye(G.shape[0], dtype=G.dtype, device=dev))
        facs[l] = (Ginv, Ainv)
    return facs


def apply_kfac(facs, v):
    """M^-1 v for the flat free-parameter vector v (layer blocks: weight (out,in) and bias (out))."""
    d = P.unflat(v); out = {}
    for l in range(L):
        Ginv, Ainv = facs[l]
        kw, kb = f"layers.{l}.weight", f"layers.{l}.bias"
        if kw in P.freeze:                       # layer 0: bias only
            out[kb] = Ginv @ d[kb]
            continue
        V = torch.cat([d[kw], d[kb][:, None]], 1)           # (out, in+1)
        R = Ginv @ V @ Ainv
        out[kw] = R[:, :-1]; out[kb] = R[:, -1]
    return P.flat(out)


def pcg(Aop, b, Minv, iters, tol):
    x = torch.zeros_like(b); r = b.clone(); z = Minv(r); p = z.clone(); rz = r @ z; bn = math.sqrt(float(r @ r))
    for k in range(iters):
        Ap = Aop(p); alpha = rz / (p @ Ap); x += alpha * p; r -= alpha * Ap
        if math.sqrt(float(r @ r)) <= tol * bn:
            return x, k + 1
        z = Minv(r); rz_new = r @ z; p = z + (rz_new / rz) * p; rz = rz_new
    return x, iters


theta = P.theta; loss = P.loss(theta); t0 = time.time()
em, pm, mm = P.errors(theta, t64)
print(f"{tag} init loss {loss:.3e} | eps max {em:.3e} [{fmt(pm)}]", flush=True)
gen = torch.Generator(device="cpu").manual_seed(0)
lam = None
for it in range(ITERS):
    r, g, JtJ, Jv = P.linearize(theta)
    if lam is None:
        v = torch.randn(P.numel, generator=gen, dtype=torch.float64).to(dev); lam = 1e-3 * float(v @ JtJ(v)) / P.numel
    if PC == "kfac":
        facs = kfac_factors(theta, lam); Minv = lambda z: apply_kfac(facs, z)
    elif PC == "jacobi":
        groups = ts._row_groups(theta, P.keys); D = ts._diag_estimate(JtJ, groups, int(groups.max()) + 1, P.numel, gen, dev)
        Minv = lambda z, D=D: z / (D + lam)
    else:
        Minv = lambda z: z
    accepted = False
    for _try in range(12):
        delta, ncg = pcg(lambda z: JtJ(z) + lam * z, -g, Minv, 300, 1e-3)
        cand = P.step(theta, delta); new_loss = P.loss(cand)
        pred = float(-2.0 * (g @ delta) - (delta @ JtJ(delta))); rho = (loss - new_loss) / pred if pred > 0 else -1.0
        if new_loss < loss and rho > 1e-3:
            theta, loss, accepted = cand, new_loss, True
            lam = max(lam / 3.0, 1e-300) if rho > 0.75 else (lam * 2.0 if rho < 0.25 else lam)
            break
        lam *= 4.0
    em, pm, mm = P.errors(theta, t64)
    print(f"{tag} it {it+1:2d} | loss {loss:.3e} | lam {lam:.1e} | cg {ncg} | rho {rho:.2f} | |d| {float(delta.norm()):.2e} | {'ok' if accepted else 'REJECT'} | eps max {em:.3e} mean {sum(mm)/len(mm):.3e} [{fmt(pm)}] | {time.time()-t0:.0f}s", flush=True)
    if not accepted:
        break
print(f"FINAL {tag} loss {loss:.3e} eps max {em:.3e}", flush=True)
