"""Two bias diagnostics on layer 2 (prefix exact):
 A. Paper-2 style estimator: bias from the location of the peak of |f'| along the neuron's
    dual line (exact only if the downstream is affine in h_j).  Reports the true
    pre-activation at the peak = bias error.
 B. Full-domain first-order conditioning kappa (offline, teacher weights): residual = Jacobians
    of the output w.r.t. h2 at N random cube points; a = d r/d b_j ; B = d r/d(all downstream
    params). kappa = |(I - B B^+) a| / |a|.  Compared with the 1-D slice value (6e-15).
env: TEACHER, ARCH (default: the no-narrowing trunk 784,128,80,80,80,10), NEURONS, NPTS
"""
import os, sys, math, torch, numpy as np
from torch.func import functional_call, jvp, vmap
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scipy.optimize import linprog
from nets import MLP
from tanh_solve import Oracle
E = os.environ.get
torch.manual_seed(0); dev = "cuda"
arch = [int(v) for v in E("ARCH", "784,128,80,80,80,10").split(",")]
path = E("TEACHER")
T = MLP(arch, act="tanh").to(dev, torch.float64)
T.load_state_dict({k: v.double() for k, v in torch.load(path, map_location="cpu").items()})
bb = Oracle(lambda x: T(x))
W1, b1 = T.layers[0].weight.detach(), T.layers[0].bias.detach(); W1p = torch.linalg.pinv(W1)
W2, b2 = T.layers[1].weight.detach(), T.layers[1].bias.detach(); U = torch.linalg.pinv(W2)
n2 = W2.shape[0]
def f_of_h(H): return bb((torch.atanh(H) - b1) @ W1p.T)
BOX = 0.995
def longest_line(j):
    v = (U[:, j] / (W2[j] @ U[:, j])).cpu().numpy()
    c = np.zeros(129); c[-1] = -1
    A_ub = np.vstack([np.hstack([np.eye(128), v[:, None] * 0.5]), np.hstack([-np.eye(128), v[:, None] * 0.5]),
                      np.hstack([np.eye(128), -v[:, None] * 0.5]), np.hstack([-np.eye(128), -v[:, None] * 0.5])])
    r = linprog(c, A_ub=A_ub, b_ub=np.full(512, BOX), bounds=[(-BOX, BOX)] * 128 + [(0, None)], method="highs")
    return r.x[:128], r.x[-1]

# ---- A. peak-of-|f'| bias estimator (Paper 2, sigmoid/tanh section) ----------------------
print("A. bias from the peak of |f'| along the dual line (Paper-2 estimator; exact iff downstream affine in h_j)")
NEURONS = [int(x) for x in E("NEURONS", "0,9,17,42,63").split(",")]
for j in NEURONS:
    hm, L = longest_line(j)
    w = W2[j]; v = U[:, j] / (w @ U[:, j]); hmid = torch.tensor(hm, dtype=torch.float64, device=dev)
    zc = float(w @ hmid + b2[j])
    if not (zc - L / 2 < -1.5 and zc + L / 2 > 1.5):
        print(f"  neuron {j}: transition not inside the reachable line (z in [{zc-L/2:.2f},{zc+L/2:.2f}]) -> skip"); continue
    t = torch.linspace(-L / 2, L / 2, 20001, dtype=torch.float64, device=dev)
    P = hmid[None] + t[:, None] * v[None]; h = 1e-4
    g = ((f_of_h(P + h * v) - f_of_h(P - h * v)) / (2 * h))            # (N, 10) = G'(h_j) * sech^2(z)
    z = zc + t
    per_out = []
    for k in range(10):
        gk = g[:, k].abs(); i = int(gk.argmax()); per_out.append(float(z[i]))
    inorm = int(g.norm(dim=1).argmax())
    # parabolic sub-grid refinement of the norm peak
    gn = g.norm(dim=1); i = inorm
    a0, a1, a2 = float(gn[i - 1]), float(gn[i]), float(gn[i + 1]); off = 0.5 * (a0 - a2) / (a0 - 2 * a1 + a2)
    zpk = float(z[i]) + off * float(t[1] - t[0])
    print(f"  neuron {j}: b_true {float(b2[j]):+.4f} | z at peak of |f'| (=bias error): norm {zpk:+.4f}; "
          f"per-output range [{min(per_out):+.3f}, {max(per_out):+.3f}]")

# ---- B. full-domain kappa (uses teacher weights; offline diagnostic) --------------------------
print("\nB. first-order conditioning of the bias against the FULL downstream (Jacobian data on the 80-dim cube)")
NPTS = int(E("NPTS", 30)); j = NEURONS[1] if len(NEURONS) > 1 else NEURONS[0]
down = MLP(arch[2:], act="tanh").to(dev, torch.float64)
with torch.no_grad():
    for i, l in enumerate(down.layers):
        l.weight.copy_(T.layers[2 + i].weight); l.bias.copy_(T.layers[2 + i].bias)
keys = [k for k, _ in down.named_parameters()]; shapes = [p.shape for p in down.parameters()]
def unflat(th):
    out, i = {}, 0
    for k, sh in zip(keys, shapes):
        n = int(np.prod(sh)); out[k] = th[i:i + n].view(sh); i += n
    return out
eta = torch.cat([p.detach().reshape(-1) for p in down.parameters()])
A = (torch.rand(NPTS, 128, dtype=torch.float64, device=dev) * 2 - 1) * 0.9        # layer-2 inputs
def resid(th, bj):
    """Jacobian of the output w.r.t. h2 (= layer-2 output) at the N points, flattened: (N*10*80,)."""
    b2m = b2.clone(); b2m = torch.cat([b2m[:j], bj[None], b2m[j + 1:]])
    z2 = A @ W2.T + b2m; h2 = torch.tanh(z2)
    # d out / d h2 at h2 (analytic, via jvp over the 80 basis directions)
    def g(hh): return functional_call(down, unflat(th), (hh[None],))[0]
    Eye = torch.eye(80, dtype=torch.float64, device=dev)
    J = torch.stack([vmap(lambda e: jvp(g, (h2[n],), (e,))[1])(Eye) for n in range(NPTS)])   # (N, 80, 10)
    return J.reshape(-1)
bj0 = b2[j].clone()
r0 = resid(eta, bj0)
a_vec = jvp(lambda bj: resid(eta, bj), (bj0,), (torch.ones((), dtype=torch.float64, device=dev),))[1]
eye = torch.eye(len(eta), dtype=torch.float64, device=dev)
B = torch.cat([vmap(lambda vv: jvp(lambda th: resid(th, bj0), (eta,), (vv,))[1])(eye[i:i + 64])
               for i in range(0, len(eta), 64)]).T
print(f"  residual rows {B.shape[0]}, downstream params {B.shape[1]} (rows/params {B.shape[0]/B.shape[1]:.1f})")
Q, R = torch.linalg.qr(B)
proj = a_vec - Q @ (Q.T @ a_vec)
print(f"  neuron {j}: |a| {float(a_vec.norm()):.3e}, kappa {float(proj.norm()):.3e}, ratio {float(proj.norm()/a_vec.norm()):.3e}"
      f"   (1-D slice gave 6e-15)")
sv = torch.linalg.svdvals(B); print(f"  B: rank {(sv > 1e-10*sv[0]).sum().item()}/{len(eta)}, sv min/max {float(sv[-1]):.1e}/{float(sv[0]):.1e}")
print(f"queries {bb.n}")
