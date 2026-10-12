"""Bias profile L_j(b) = min_eta sum_q |G_eta(tanh(s t_q + b)) - f_j(t_q)|^2 along neuron j's
dual slice, with G_eta a 1->40->32->10 tanh net fitted from scratch (dense LM, fp64, several inits).
A sharp isolated minimum at the true bias => bias identifiable from the slice; flat => not.
Uses TRUE duals and TRUE scale (identifiability test, not black-box recovery)."""
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
def f_of_h(H): return bb((torch.atanh(H) - b1) @ W1p.T)
J = int(E("NEURON", 17)); NQ = int(E("NQ", 1500)); NINIT = int(E("NINIT", 6)); ITERS = int(E("ITERS", 60))
OFFS = [float(x) for x in E("OFFS", "-0.1,-0.03,-0.01,-0.003,0,0.003,0.01,0.03,0.1").split(",")]
w = W2[J]; s_true = float(w.norm()); wd = (w / s_true).cpu().numpy(); b_true = float(b2[J])
v = U[:, J] / (w @ U[:, J])                                     # w.v = 1: z_j = z0 + t
vn = v.cpu().numpy(); BOX = 0.995
# longest segment along v inside the box (LP on both endpoints)
c = np.zeros(129); c[-1] = -1
A_ub = np.vstack([np.hstack([np.eye(128), vn[:, None] * 0.5]), np.hstack([-np.eye(128), vn[:, None] * 0.5]),
                  np.hstack([np.eye(128), -vn[:, None] * 0.5]), np.hstack([-np.eye(128), -vn[:, None] * 0.5])])
b_ub = np.concatenate([np.full(128, BOX)] * 4)
r = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(-BOX, BOX)] * 128 + [(0, None)], method="highs")
hmid = torch.tensor(r.x[:128], dtype=torch.float64, device=dev); L = r.x[-1]
t = torch.linspace(-L / 2, L / 2, NQ, dtype=torch.float64, device=dev)
Hs = hmid[None] + t[:, None] * v[None]
z = (Hs @ w + b_true)                                            # true z along the slice (diagnostic)
Y = f_of_h(Hs)                                                   # (NQ, 10) black-box values
zc = float(w @ hmid + b_true)                                    # z at t=0
print(f"neuron {J}: |w| {s_true:.3f} b {b_true:+.4f} | slice z in [{float(z.min()):.2f}, {float(z.max()):.2f}] ({NQ} points) | q {bb.n}")
print(f"slice net 1->40->32->10: {40+40+40*32+32+32*10+10} params, {NQ*10} residuals")

def fit_slice(a, Y, n_init, iters, seed):
    """Fit G(a) = W3' tanh(W2' tanh(w1 a + c1) + c2) + c3 with dense LM; return best loss over inits."""
    best = math.inf
    for k in range(n_init):
        g = torch.Generator(device="cpu").manual_seed(seed * 100 + k)
        net = MLP([1, 40, 32, 10], act="tanh").to(dev, torch.float64)
        with torch.no_grad():                                    # spread the 40 first-layer kinks over the interval
            net.layers[0].weight.copy_(torch.randn(40, 1, generator=g).double().to(dev) * 3)
            net.layers[0].bias.copy_((torch.rand(40, generator=g).double().to(dev) * 2 - 1) * 2)
        keys = [k_ for k_, _ in net.named_parameters()]
        theta = torch.cat([p.detach().reshape(-1) for p in net.parameters()])
        shapes = [p.shape for p in net.parameters()]
        def unflat(th):
            out, i = {}, 0
            for k_, sh in zip(keys, shapes):
                n = int(np.prod(sh)); out[k_] = th[i:i + n].view(sh); i += n
            return out
        def resid(th):
            return (functional_call(net, unflat(th), (a[:, None],)) - Y).reshape(-1)
        lam = 1e-2
        loss = float(resid(theta).pow(2).sum())
        for it in range(iters):
            rr = resid(theta)
            eye = torch.eye(len(theta), dtype=torch.float64, device=dev)
            Jm = torch.cat([vmap(lambda vv: jvp(resid, (theta,), (vv,))[1])(eye[i:i + 256]) for i in range(0, len(theta), 256)]).T
            Hm = Jm.T @ Jm; gm = Jm.T @ rr; dsc = float(Jm.norm(dim=0).mean())
            ok = False
            for _ in range(10):
                Lc, info = torch.linalg.cholesky_ex(Hm + lam * dsc ** 2 * eye)
                if info != 0: lam *= 10; continue
                step = -torch.cholesky_solve(gm[:, None], Lc)[:, 0]
                cand = theta + step; lc = float(resid(cand).pow(2).sum())
                if lc < loss: theta, loss, ok = cand, lc, True; lam = max(lam / 5, 1e-15); break
                lam *= 10
            if not ok: break
        best = min(best, loss)
    return best

print("bias offset (b_trial - b_true) -> best fit loss over inits   (flat = bias NOT identified by the slice)")
t0 = time.time()
for off in OFFS:
    b_trial = b_true + off
    a = torch.tanh(zc - b_true + b_trial + t)                     # = tanh(s t + b_trial) with the slice's own coordinate
    Lb = fit_slice(a, Y, NINIT, ITERS, seed=int(1000 * abs(off)) + 7)
    print(f"  {off:+.3f}: loss {Lb:.3e}   (per-point rms {math.sqrt(Lb / (NQ * 10)):.2e})  [{time.time() - t0:.0f}s]", flush=True)
