"""Dual-direction exact isolation + MULTI-HARMONIC tail magnitude fit (the ~5e-6 method)
on the REAL trained sigmoid teacher 784x128x80x40x32x10. Guess 5deg/+-8%/+-0.08. before->after."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch, numpy as np
from nets import MLP
dev = "cuda"
CKPT = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt"
dims = [784, 128, 80, 40, 32, 10]; d = dims[0]; k = dims[1]; O = dims[-1]
teacher = MLP(dims, act="sigmoid").to(dev).double()
teacher.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)); teacher.eval()
W1t = teacher.layers[0].weight.detach(); b1t = teacher.layers[0].bias.detach()
tn = W1t / W1t.norm(dim=1, keepdim=True); g = torch.Generator(device=dev).manual_seed(1)
@torch.no_grad()
def J_at(x, fd=5e-5):
    E = torch.eye(d, device=dev, dtype=torch.float64)
    return ((teacher(x.unsqueeze(0) + fd * E) - teacher(x.unsqueeze(0) - fd * E)) / (2 * fd)).t()
Wg = torch.empty(k, d, device=dev, dtype=torch.float64); bg = torch.empty(k, device=dev, dtype=torch.float64)
for j in range(k):
    u = tn[j]; v = torch.randn(d, generator=g, device=dev, dtype=torch.float64); v = v - (v @ u) * u; v = v / v.norm()
    Wg[j] = (math.cos(math.radians(5)) * u + math.sin(math.radians(5)) * v) * W1t[j].norm() * (1 + 0.08 * (2 * torch.rand(1, generator=g, device=dev, dtype=torch.float64) - 1))
    bg[j] = b1t[j] + 0.08 * (2 * torch.rand(1, generator=g, device=dev, dtype=torch.float64).item() - 1)
Wgpinv = Wg.t() @ torch.linalg.inv(Wg @ Wg.t())
mag_true = W1t.norm(dim=1)
before_abs = (Wg.norm(dim=1) - mag_true).abs(); before_rel = before_abs / mag_true
# directions via SVD
N = torch.empty_like(Wg)
for j in range(k):
    t = torch.full((k,), 20.0, device=dev, dtype=torch.float64); t[j] = 0.0
    x0 = Wgpinv @ (t - bg); U, Sv, Vh = torch.linalg.svd(J_at(x0), full_matrices=False)
    nv = Vh[0]; N[j] = nv if float(nv @ Wg[j]) > 0 else -nv
NNt = N @ N.t(); print(f"cond(N N^T) = {float(torch.linalg.cond(NNt)):.1f}")
V = (N.t() @ torch.linalg.inv(NNt))
W1rec = N * Wg.norm(dim=1, keepdim=True); W1rp = W1rec.t() @ torch.linalg.inv(W1rec @ W1rec.t())
P = 6; nctx = 8; ncand = 40
def fit_mag(tails, a_g):
    ell = np.arange(P + 1)[:, None]
    def res(a):
        tot = 0.0
        for ts, gs, side in tails:
            A = np.exp(-side * a * ell * ts[None, :]).T
            coef, _, _, _ = np.linalg.lstsq(A, gs, rcond=None); tot += float(((A @ coef - gs) ** 2).sum())
        return tot
    lo, hi = 0.85 * a_g, 1.15 * a_g
    for _ in range(70):
        m1 = hi - (hi - lo) * 0.618; m2 = lo + (hi - lo) * 0.618
        if res(m1) < res(m2): hi = m2
        else: lo = m1
    return 0.5 * (lo + hi)
merr = []
for jj in range(k):
    j = jj; a_g = float(Wg[j].norm()); vj = V[:, j]
    TT = (2 * torch.rand(ncand, k, generator=g, device=dev, dtype=torch.float64) - 1) * 2.0; TT[:, j] = 0.0
    X0 = (TT - bg) @ W1rp.t()
    with torch.no_grad():
        swing = (teacher(X0 + (6.0 / a_g) * vj) - teacher(X0 - (6.0 / a_g) * vj)).norm(dim=1)
    top = torch.topk(swing, nctx).indices
    tp = torch.linspace(2.5 / a_g, 7.0 / a_g, 40, device=dev, dtype=torch.float64)
    tm = torch.linspace(-7.0 / a_g, -2.5 / a_g, 40, device=dev, dtype=torch.float64)
    tails = []
    for mi in top.tolist():
        x0 = X0[mi]
        with torch.no_grad():
            Fp = teacher(x0.unsqueeze(0) + tp.unsqueeze(1) * vj.unsqueeze(0)).cpu().numpy()
            Fm = teacher(x0.unsqueeze(0) + tm.unsqueeze(1) * vj.unsqueeze(0)).cpu().numpy()
        for r in range(O):
            tails.append((tp.cpu().numpy(), Fp[:, r], +1)); tails.append((tm.cpu().numpy(), Fm[:, r], -1))
    a_hat = fit_mag(tails, a_g); merr.append(abs(a_hat - float(mag_true[j])))
merr = torch.tensor(merr); mrel = merr / mag_true.cpu()
print("MAGNITUDE via dual-direction + MULTI-HARMONIC tail on REAL 784x128 sigmoid net:")
print(f"  BEFORE (guess):  abs median {float(before_abs.median()):.3e} max {float(before_abs.max()):.3e} | rel median {float(before_rel.median()):.3e} max {float(before_rel.max()):.3e}")
print(f"  AFTER  (recov):  abs median {float(merr.median()):.3e} max {float(merr.max()):.3e} | rel median {float(mrel.median()):.3e} max {float(mrel.max()):.3e}")
