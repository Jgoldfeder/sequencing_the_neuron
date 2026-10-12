"""OFFLINE DIAGNOSTIC (uses teacher weights; not a recovery method).
Does neuron j's dual slice carry first-order information about its bias, given a
downstream of the true architecture (1->40->32->10) that is fitted freely?
 1. exact slice downstream eta* from the teacher -> loss at b_true ~ 0
 2. kappa = |(I - B B^+) a|, a = d r/d b, B = d r/d eta at (eta*, b_true):
    linearised best-compensated residual for a bias displacement delta is delta^2 kappa^2
 3. continuation: displace b, refit eta by LM from the previous solution, train + held-out loss
"""
import os, sys, math, time, torch, numpy as np
from torch.func import functional_call, jvp, vmap
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scipy.optimize import linprog
from nets import MLP
from tanh_solve import Oracle
E = os.environ.get
torch.manual_seed(0); dev = "cuda"
arch = [784, 128, 80, 40, 32, 10]
T = MLP(arch, act="tanh").to(dev, torch.float64)
T.load_state_dict({k: v.double() for k, v in torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                  "teachers/teacher_784x128x80x40x32x10_e25_s0_tanh.pt"), map_location="cpu").items()})
bb = Oracle(lambda x: T(x))
W1, b1 = T.layers[0].weight.detach(), T.layers[0].bias.detach(); W1p = torch.linalg.pinv(W1)
W2, b2 = T.layers[1].weight.detach(), T.layers[1].bias.detach(); U = torch.linalg.pinv(W2)
W3, b3 = T.layers[2].weight.detach(), T.layers[2].bias.detach()
W4, b4 = T.layers[3].weight.detach(), T.layers[3].bias.detach()
W5, b5 = T.layers[4].weight.detach(), T.layers[4].bias.detach()
def f_of_h(H): return bb((torch.atanh(H) - b1) @ W1p.T)
BOX = 0.995; NQ = int(E("NQ", 1500))

def longest_line(j):
    """midpoint h and half-length so that h +- (L/2) v stays in the box; z_j spans L (w.v = 1)."""
    v = (U[:, j] / (W2[j] @ U[:, j])).cpu().numpy()
    c = np.zeros(129); c[-1] = -1
    A_ub = np.vstack([np.hstack([np.eye(128), v[:, None] * 0.5]), np.hstack([-np.eye(128), v[:, None] * 0.5]),
                      np.hstack([np.eye(128), -v[:, None] * 0.5]), np.hstack([-np.eye(128), -v[:, None] * 0.5])])
    r = linprog(c, A_ub=A_ub, b_ub=np.full(512, BOX), bounds=[(-BOX, BOX)] * 128 + [(0, None)], method="highs")
    return r.x[:128], r.x[-1]

if E("NEURON"):
    J = int(E("NEURON")); hm, L = longest_line(J)
else:
    best = None
    for j in range(80):
        hm, L = longest_line(j)
        if best is None or L > best[1]: best = (j, L, hm)
    J, L, hm = best
    print(f"longest single dual line: neuron {J}, z-span {L:.2f}")
w = W2[J]; b_true = float(b2[J]); v = U[:, J] / (w @ U[:, J])
hmid = torch.tensor(hm, dtype=torch.float64, device=dev)
t_all = torch.linspace(-L / 2, L / 2, 2 * NQ, dtype=torch.float64, device=dev)
t_tr, t_ho = t_all[0::2], t_all[1::2]                           # interleaved train / held-out
def points(t): return hmid[None] + t[:, None] * v[None]
Y_tr, Y_ho = f_of_h(points(t_tr)), f_of_h(points(t_ho))
zc = float(w @ hmid + b_true)
print(f"neuron {J}: b_true {b_true:+.5f} | slice z in [{zc - L/2:.2f}, {zc + L/2:.2f}] | tanh' at ends {1-math.tanh(zc-L/2)**2:.1e} / {1-math.tanh(zc+L/2)**2:.1e}")

# ---- 1. exact slice downstream --------------------------------------------------------
h_other = torch.tanh(points(torch.zeros(1, dtype=torch.float64, device=dev)) @ W2.T + b2)[0]   # other neurons' activations (constant on the slice)
mask = torch.ones(80, dtype=torch.bool, device=dev); mask[J] = False
net = MLP([1, 40, 32, 10], act="tanh").to(dev, torch.float64)
with torch.no_grad():
    net.layers[0].weight.copy_(W3[:, J:J + 1]); net.layers[0].bias.copy_(b3 + W3[:, mask] @ h_other[mask])
    net.layers[1].weight.copy_(W4); net.layers[1].bias.copy_(b4)
    net.layers[2].weight.copy_(W5); net.layers[2].bias.copy_(b5)
keys = [k for k, _ in net.named_parameters()]; shapes = [p.shape for p in net.parameters()]
def unflat(th):
    out, i = {}, 0
    for k, sh in zip(keys, shapes):
        n = int(np.prod(sh)); out[k] = th[i:i + n].view(sh); i += n
    return out
eta_star = torch.cat([p.detach().reshape(-1) for p in net.parameters()])
def resid(th, b, t, Y):
    a = torch.tanh(zc - b_true + b + t)                          # tanh(z_j) with trial bias b
    return (functional_call(net, unflat(th), (a[:, None],)) - Y).reshape(-1)
L0 = float(resid(eta_star, b_true, t_tr, Y_tr).pow(2).sum())
print(f"1. exact slice downstream at b_true: train loss {L0:.2e} (per-point rms {math.sqrt(L0/(NQ*10)):.1e})")

# ---- 2. first-order conditioning kappa -------------------------------------------------
a_vec = jvp(lambda b: resid(eta_star, b, t_tr, Y_tr), (torch.tensor(b_true, dtype=torch.float64, device=dev),),
            (torch.ones((), dtype=torch.float64, device=dev),))[1]
eye = torch.eye(len(eta_star), dtype=torch.float64, device=dev)
B = torch.cat([vmap(lambda vv: jvp(lambda th: resid(th, b_true, t_tr, Y_tr), (eta_star,), (vv,))[1])(eye[i:i + 256])
               for i in range(0, len(eta_star), 256)]).T                       # (NQ*10, 1722)
Q, R = torch.linalg.qr(B)
proj = a_vec - Q @ (Q.T @ a_vec)
kappa = float(proj.norm()); an = float(a_vec.norm())
sv = torch.linalg.svdvals(B)
print(f"2. |a| = {an:.3e}, kappa = |(I-BB+)a| = {kappa:.3e}  (ratio {kappa/an:.2e}); B rank {(sv > 1e-10*sv[0]).sum().item()}/{len(eta_star)}, sv min/max {float(sv[-1]):.1e}/{float(sv[0]):.1e}")
for d in (1e-3, 1e-2, 3e-2):
    print(f"   predicted best-compensated loss for |delta b| = {d:g}: {d*d*kappa*kappa:.2e}  (per-point rms {d*kappa/math.sqrt(NQ*10):.1e})")

# ---- 3. continuation in the trial bias --------------------------------------------------
def lm_fit(theta, b, iters=60):
    lam = 1e-6
    r = resid(theta, b, t_tr, Y_tr); loss = float(r @ r)
    for it in range(iters):
        Jm = torch.cat([vmap(lambda vv: jvp(lambda th: resid(th, b, t_tr, Y_tr), (theta,), (vv,))[1])(eye[i:i + 256])
                        for i in range(0, len(theta), 256)]).T
        Hm = Jm.T @ Jm; gm = Jm.T @ r; dsc = float(Jm.norm(dim=0).mean())
        ok = False
        for _ in range(12):
            Lc, info = torch.linalg.cholesky_ex(Hm + lam * dsc ** 2 * eye)
            if info != 0: lam *= 10; continue
            step = -torch.cholesky_solve(gm[:, None], Lc)[:, 0]
            cand = theta + step; rc = resid(cand, b, t_tr, Y_tr); lc = float(rc @ rc)
            if lc < loss: theta, r, loss, ok = cand, rc, lc, True; lam = max(lam / 5, 1e-16); break
            lam *= 10
        if not ok or float(step.norm()) < 1e-14: break
    return theta, loss
print("3. continuation from the exact solution: delta b -> train loss / held-out loss  (compare with delta^2 kappa^2)")
for sgn in (+1, -1):
    theta = eta_star.clone()
    for d in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        b = b_true + sgn * d
        theta, ltr = lm_fit(theta, b)
        lho = float(resid(theta, b, t_ho, Y_ho).pow(2).sum())
        print(f"   delta b = {sgn*d:+.3f}: train {ltr:.2e}  held-out {lho:.2e}  predicted {d*d*kappa*kappa:.2e}", flush=True)
print(f"queries {bb.n}")
