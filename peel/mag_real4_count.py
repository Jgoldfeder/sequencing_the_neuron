"""Same recovery as mag_real4 but with a query counter wrapping the teacher.
Reports total black-box queries (input rows), split by direction vs magnitude phase."""
import sys, math, torch, numpy as np
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
dev = "cuda"
CD = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
pop = torch.load(CD + "_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt", map_location=dev, weights_only=False)
merged = torch.load(CD + "mergedbest512_sigmoid__784x128x80x40x32x10__s0_final.pt", map_location=dev, weights_only=False)
dims = pop["dims"]; d = dims[0]; k = dims[1]; O = dims[-1]
_teacher = MLP(dims, act="sigmoid").to(dev).double(); _teacher.load_state_dict(pop["teacher_state"]); _teacher.eval()
class Counter:
    def __init__(self, t): self.t = t; self.n = 0
    @torch.no_grad()
    def __call__(self, X): self.n += int(X.shape[0]); return self.t(X)
teacher = Counter(_teacher)
Wg = merged["state_dict"]["layers.0.weight"].to(dev).double().clone()
bg = merged["state_dict"]["layers.0.bias"].to(dev).double().clone()
@torch.no_grad()
def J_at(x, fd=5e-5):
    E = torch.eye(d, device=dev, dtype=torch.float64)
    return ((teacher(x.unsqueeze(0) + fd * E) - teacher(x.unsqueeze(0) - fd * E)) / (2 * fd)).t()
Wgpinv = Wg.t() @ torch.linalg.inv(Wg @ Wg.t())
g = torch.Generator(device=dev).manual_seed(1)
# ---- DIRECTION phase ----
q0 = teacher.n
N = torch.empty_like(Wg)
for j in range(k):
    t = torch.full((k,), 20.0, device=dev, dtype=torch.float64); t[j] = 0.0
    x0 = Wgpinv @ (t - bg); U, Sv, Vh = torch.linalg.svd(J_at(x0), full_matrices=False)
    nv = Vh[0]; N[j] = nv if float(nv @ Wg[j]) > 0 else -nv
q_dir = teacher.n - q0
V = N.t() @ torch.linalg.inv(N @ N.t())
W1rec_dir = N * Wg.norm(dim=1, keepdim=True); W1rp = W1rec_dir.t() @ torch.linalg.inv(W1rec_dir @ W1rec_dir.t())
# ---- MAGNITUDE phase ----
q1 = teacher.n
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
for jj in range(k):
    a_g = float(Wg[jj].norm()); vj = V[:, jj]
    TT = (2 * torch.rand(ncand, k, generator=g, device=dev, dtype=torch.float64) - 1) * 2.0; TT[:, jj] = 0.0
    X0 = (TT - bg) @ W1rp.t()
    with torch.no_grad():
        swing = (teacher(X0 + (6.0 / a_g) * vj) - teacher(X0 - (6.0 / a_g) * vj)).norm(dim=1)
    top = torch.topk(swing, nctx).indices
    tp = torch.linspace(2.5 / a_g, 7.0 / a_g, 40, device=dev, dtype=torch.float64)
    tm = torch.linspace(-7.0 / a_g, -2.5 / a_g, 40, device=dev, dtype=torch.float64)
    for mi in top.tolist():
        x0 = X0[mi]
        with torch.no_grad():
            teacher(x0.unsqueeze(0) + tp.unsqueeze(1) * vj.unsqueeze(0))
            teacher(x0.unsqueeze(0) + tm.unsqueeze(1) * vj.unsqueeze(0))
q_mag = teacher.n - q1
print(f"first-layer recovery on real 784x128 sigmoid net ({k} neurons, d={d}):")
print(f"  DIRECTION phase : {q_dir:>10,} queries  ({q_dir//k:,}/neuron ~ 2d={2*d})")
print(f"  MAGNITUDE phase : {q_mag:>10,} queries  ({q_mag//k:,}/neuron)")
print(f"  TOTAL           : {teacher.n:>10,} queries  ({teacher.n//k:,}/neuron)")
