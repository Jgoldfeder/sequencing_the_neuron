"""DEFINITIVE before->after: guess = the REAL consensus (merged) first layer, NOT a fabricated
perturbation. Refine direction (saturate+SVD) + magnitude (multi-harmonic) on the real trained
sigmoid teacher. Report first-layer WEIGHT-row error under alignment (perm+sign), mean & max."""
import sys, math, torch, numpy as np
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from scipy.optimize import linear_sum_assignment
from nets import MLP
dev = "cuda"
CD = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
pop = torch.load(CD + "_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt", map_location=dev, weights_only=False)
merged = torch.load(CD + "mergedbest512_sigmoid__784x128x80x40x32x10__s0_final.pt", map_location=dev, weights_only=False)
dims = pop["dims"]; d = dims[0]; k = dims[1]; O = dims[-1]
teacher = MLP(dims, act="sigmoid").to(dev).double(); teacher.load_state_dict(pop["teacher_state"]); teacher.eval()
W1t = teacher.layers[0].weight.detach(); nt = W1t.norm(dim=1)
# GUESS = real consensus (merged) first layer
Wg = merged["state_dict"]["layers.0.weight"].to(dev).double().clone()
bg = merged["state_dict"]["layers.0.bias"].to(dev).double().clone()
@torch.no_grad()
def J_at(x, fd=5e-5):
    E = torch.eye(d, device=dev, dtype=torch.float64)
    return ((teacher(x.unsqueeze(0) + fd * E) - teacher(x.unsqueeze(0) - fd * E)) / (2 * fd)).t()
Wgpinv = Wg.t() @ torch.linalg.inv(Wg @ Wg.t())
g = torch.Generator(device=dev).manual_seed(1)
# direction via SVD at isolated saturation point
N = torch.empty_like(Wg)
for j in range(k):
    t = torch.full((k,), 20.0, device=dev, dtype=torch.float64); t[j] = 0.0
    x0 = Wgpinv @ (t - bg); U, Sv, Vh = torch.linalg.svd(J_at(x0), full_matrices=False)
    nv = Vh[0]; N[j] = nv if float(nv @ Wg[j]) > 0 else -nv
NNt = N @ N.t(); print(f"cond(N N^T) = {float(torch.linalg.cond(NNt)):.1f}")
V = N.t() @ torch.linalg.inv(NNt)
W1rec_dir = N * Wg.norm(dim=1, keepdim=True); W1rp = W1rec_dir.t() @ torch.linalg.inv(W1rec_dir @ W1rec_dir.t())
P = 6; nctx = 8; ncand = 40
def fit_mag(tails, a_g):
    ell = np.arange(P + 1)[:, None]
    def res(a):
        tot = 0.0
        for ts, gs, side in tails:
            A = np.exp(-side * a * ell * ts[None, :]).T
            coef, _, _, _ = np.linalg.lstsq(A, gs, rcond=None); tot += float(((A @ coef - gs) ** 2).sum())
        return tot
    lo, hi = 0.80 * a_g, 1.20 * a_g
    for _ in range(70):
        m1 = hi - (hi - lo) * 0.618; m2 = lo + (hi - lo) * 0.618
        if res(m1) < res(m2): hi = m2
        else: lo = m1
    return 0.5 * (lo + hi)
a_hat = torch.zeros(k, device=dev, dtype=torch.float64)
for jj in range(k):
    a_g = float(Wg[jj].norm()); vj = V[:, jj]
    TT = (2 * torch.rand(ncand, k, generator=g, device=dev, dtype=torch.float64) - 1) * 2.0; TT[:, jj] = 0.0
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
    a_hat[jj] = fit_mag(tails, a_g)
W1_after = a_hat[:, None] * N
def aligned(W, tag):
    Cp = torch.cdist(W, W1t); Cm = torch.cdist(-W, W1t); C = torch.minimum(Cp, Cm).cpu().numpy()
    ri, ci = linear_sum_assignment(C); e = torch.tensor([C[ri[t], ci[t]] for t in range(len(ri))]); rel = e / nt[ci].cpu()
    print(f"  {tag:26s} abs: mean {float(e.mean()):.3e} max {float(e.max()):.3e} | rel: mean {float(rel.mean()):.3e} max {float(rel.max()):.3e}")
print("first-layer WEIGHT-row error under alignment (real consensus guess -> refined):")
aligned(Wg, "BEFORE (real consensus)")
aligned(W1_after, "AFTER  (refined dir+mag)")
