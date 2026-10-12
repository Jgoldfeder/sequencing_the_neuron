"""Rank-aware first-order conditioning of a layer-2 bias against the downstream parameters.
For each cutoff rtol: keep singular directions of B with s > rtol*s0, project a onto them,
report residual/|a| (uncompensable fraction at that cutoff) and the compensation norm
|delta eta| needed per unit bias offset (huge => the linearisation is not meaningful).
MODE=slice : 1-D dual slice, downstream 1->40->32->10 exact from teacher (as bias_slice_diag)
MODE=full  : Jacobian data at NPTS cube points, all downstream params free
Offline diagnostic (teacher weights)."""
import os, sys, math, torch, numpy as np
from torch.func import functional_call, jvp, vmap
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scipy.optimize import linprog
from nets import MLP
from tanh_solve import Oracle
E = os.environ.get
torch.manual_seed(0); dev = "cuda"
arch = [int(v) for v in E("ARCH", "784,128,80,40,32,10").split(",")]
path = E("TEACHER") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "teachers/teacher_784x128x80x40x32x10_e25_s0_tanh.pt")
MODE = E("MODE", "slice"); J = int(E("NEURON", 9)); NPTS = int(E("NPTS", 30))
T = MLP(arch, act="tanh").to(dev, torch.float64)
T.load_state_dict({k: v.double() for k, v in torch.load(path, map_location="cpu").items()})
bb = Oracle(lambda x: T(x))
W1, b1 = T.layers[0].weight.detach(), T.layers[0].bias.detach(); W1p = torch.linalg.pinv(W1)
W2, b2 = T.layers[1].weight.detach(), T.layers[1].bias.detach(); U = torch.linalg.pinv(W2)
def f_of_h(H): return bb((torch.atanh(H) - b1) @ W1p.T)

if MODE == "slice":
    BOX = 0.995; NQ = 1500
    v = (U[:, J] / (W2[J] @ U[:, J])).cpu().numpy(); c = np.zeros(129); c[-1] = -1
    A_ub = np.vstack([np.hstack([np.eye(128), v[:, None] * 0.5]), np.hstack([-np.eye(128), v[:, None] * 0.5]),
                      np.hstack([np.eye(128), -v[:, None] * 0.5]), np.hstack([-np.eye(128), -v[:, None] * 0.5])])
    r = linprog(c, A_ub=A_ub, b_ub=np.full(512, BOX), bounds=[(-BOX, BOX)] * 128 + [(0, None)], method="highs")
    hmid = torch.tensor(r.x[:128], dtype=torch.float64, device=dev); L = r.x[-1]
    w = W2[J]; b_true = float(b2[J]); vt = U[:, J] / (w @ U[:, J])
    t = torch.linspace(-L / 2, L / 2, NQ, dtype=torch.float64, device=dev)
    P = hmid[None] + t[:, None] * vt[None]; Y = f_of_h(P); zc = float(w @ hmid + b_true)
    h_other = torch.tanh(P[:1] @ W2.T + b2)[0]; mask = torch.ones(80, dtype=torch.bool, device=dev); mask[J] = False
    net = MLP([1] + arch[3:], act="tanh").to(dev, torch.float64)
    with torch.no_grad():
        net.layers[0].weight.copy_(T.layers[2].weight[:, J:J + 1]); net.layers[0].bias.copy_(T.layers[2].bias + T.layers[2].weight[:, mask] @ h_other[mask])
        for i in range(1, len(net.layers)):
            net.layers[i].weight.copy_(T.layers[2 + i].weight); net.layers[i].bias.copy_(T.layers[2 + i].bias)
    keys = [k for k, _ in net.named_parameters()]; shapes = [p.shape for p in net.parameters()]
    def unflat(th):
        out, i = {}, 0
        for k, sh in zip(keys, shapes):
            n = int(np.prod(sh)); out[k] = th[i:i + n].view(sh); i += n
        return out
    eta = torch.cat([p.detach().reshape(-1) for p in net.parameters()])
    def resid(th, b):
        a = torch.tanh(zc - b_true + b + t)
        return (functional_call(net, unflat(th), (a[:, None],)) - Y).reshape(-1)
    b0 = torch.tensor(b_true, dtype=torch.float64, device=dev)
    print(f"slice: neuron {J}, {NQ} pts, z in [{zc-L/2:.2f},{zc+L/2:.2f}], downstream 1->{'->'.join(map(str,arch[3:]))}, {len(eta)} params, loss at truth {float(resid(eta,b0).pow(2).sum()):.1e}")
else:
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
    A = (torch.rand(NPTS, 128, dtype=torch.float64, device=dev) * 2 - 1) * 0.9
    Eye = torch.eye(80, dtype=torch.float64, device=dev)
    def resid(th, bj):
        b2m = torch.cat([b2[:J], bj[None], b2[J + 1:]]); h2 = torch.tanh(A @ W2.T + b2m)
        def g(hh): return functional_call(down, unflat(th), (hh[None],))[0]
        return torch.stack([vmap(lambda e: jvp(g, (h2[n],), (e,))[1])(Eye) for n in range(NPTS)]).reshape(-1)
    b0 = b2[J].clone()
    print(f"full: neuron {J}, Jacobian data at {NPTS} cube points, downstream {'->'.join(map(str,arch[2:]))}, {len(eta)} params")

a_vec = jvp(lambda b: resid(eta, b), (b0,), (torch.ones((), dtype=torch.float64, device=dev),))[1]
eye = torch.eye(len(eta), dtype=torch.float64, device=dev)
B = torch.cat([vmap(lambda vv: jvp(lambda th: resid(th, b0), (eta,), (vv,))[1])(eye[i:i + 64]) for i in range(0, len(eta), 64)]).T
print(f"B {tuple(B.shape)}, |a| {float(a_vec.norm()):.3e}")
Ub, sb, Vhb = torch.linalg.svd(B, full_matrices=False)
print(f"sv: s0 {float(sb[0]):.2e}, s[-1] {float(sb[-1]):.2e}; count above 1e-6/1e-8/1e-10/1e-12/1e-14 of s0: "
      + " ".join(str(int((sb > r * sb[0]).sum())) for r in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)))
print(f"{'rtol':>7} {'kept':>6} {'residual/|a|':>14} {'|d eta| per unit db':>20}")
for rtol in (1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
    keep = sb > rtol * sb[0]
    coeff = Ub[:, keep].T @ a_vec
    residual = a_vec - Ub[:, keep] @ coeff
    comp = Vhb[keep].T @ (coeff / sb[keep])
    print(f"{rtol:7.0e} {int(keep.sum()):6d} {float(residual.norm()/a_vec.norm()):14.3e} {float(comp.norm()):20.3e}")
