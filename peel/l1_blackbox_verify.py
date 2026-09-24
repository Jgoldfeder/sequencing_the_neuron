"""VERIFY the layer-1 WEIGHT solver uses NO oracle access / no cheating.

The teacher is wrapped in a BlackBox that exposes ONLY forward evaluation x->output.
Reading weights/bias/state_dict/parameters/intermediate activations raises AttributeError.
The whole solve (direction via finite-difference Jacobian SVD + magnitude via multi-harmonic
tail fit) is routed through that oracle and never sees teacher internals. True W1 is unlocked
ONLY at the end, for scoring. We also count the exact number of oracle forward-evaluations.

This is the same algorithm as mag_real4.py, but structurally forbidden from cheating.
"""
import sys, math, torch, numpy as np
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from scipy.optimize import linear_sum_assignment
from nets import MLP

dev = "cuda" if torch.cuda.is_available() else "cpu"
CD = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"

# ----------------------------------------------------------------------------------
# Black-box oracle: query-only. Net captured in a closure, NOT stored as an attribute.
# ----------------------------------------------------------------------------------
class BlackBox:
    def __init__(self, net):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        def _run(x):
            with torch.no_grad():
                return net(x).detach().clone()
        object.__setattr__(self, "_run", _run)
        object.__setattr__(self, "n_calls", 0)     # number of query() invocations
        object.__setattr__(self, "n_rows", 0)      # total input rows forward-evaluated
    def query(self, x):
        object.__setattr__(self, "n_calls", self.n_calls + 1)
        object.__setattr__(self, "n_rows", self.n_rows + (x.shape[0] if x.dim() > 1 else 1))
        return self._run(x)
    __call__ = query
    def __getattr__(self, name):                   # only hit when normal lookup fails
        raise AttributeError(f"BlackBox is query-only; access to '{name}' is forbidden (would be cheating)")

# ---------- load PUBLIC inputs: the architecture + our own consensus guess ----------
pop    = torch.load(CD + "_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt", map_location=dev, weights_only=False)
merged = torch.load(CD + "mergedbest512_sigmoid__784x128x80x40x32x10__s0_final.pt", map_location=dev, weights_only=False)
dims = pop["dims"]; d = dims[0]; k = dims[1]; O = dims[-1]

# build the teacher and immediately seal it inside the oracle; drop the reference
_teacher = MLP(dims, act="sigmoid").to(dev).double()
_teacher.load_state_dict(pop["teacher_state"]); _teacher.eval()
bb = BlackBox(_teacher)
del _teacher                                        # the only handle left is the oracle

# prove the wall works before we start
print("[black-box self-check] attempts to reach teacher internals:")
for probe in ["layers", "weight", "bias", "state_dict", "parameters", "act"]:
    try:
        getattr(bb, probe); print(f"    bb.{probe:11s} -> LEAKED (BUG!)")
    except AttributeError as e:
        print(f"    bb.{probe:11s} -> blocked")
print()

# GUESS = real consensus (merged) first layer -- this is OUR reconstruction, not the teacher
Wg = merged["state_dict"]["layers.0.weight"].to(dev).double().clone()
bg = merged["state_dict"]["layers.0.bias"].to(dev).double().clone()

# ================= SOLVE (oracle queries only) =================
@torch.no_grad()
def J_at(x, fd=5e-5):
    E = torch.eye(d, device=dev, dtype=torch.float64)
    return ((bb.query(x.unsqueeze(0) + fd * E) - bb.query(x.unsqueeze(0) - fd * E)) / (2 * fd)).t()

Wgpinv = Wg.t() @ torch.linalg.inv(Wg @ Wg.t())
g = torch.Generator(device=dev).manual_seed(1)

# --- direction: at a point where only neuron j is in transition, top right singular
#     vector of the (black-box, finite-difference) output Jacobian is neuron j's direction ---
N = torch.empty_like(Wg)
for j in range(k):
    t = torch.full((k,), 20.0, device=dev, dtype=torch.float64); t[j] = 0.0
    x0 = Wgpinv @ (t - bg)
    U, Sv, Vh = torch.linalg.svd(J_at(x0), full_matrices=False)
    nv = Vh[0]; N[j] = nv if float(nv @ Wg[j]) > 0 else -nv     # sign fixed by the GUESS, not teacher
NNt = N @ N.t(); print(f"cond(N N^T) = {float(torch.linalg.cond(NNt)):.1f}")
V = N.t() @ torch.linalg.inv(NNt)
W1rec_dir = N * Wg.norm(dim=1, keepdim=True)
W1rp = W1rec_dir.t() @ torch.linalg.inv(W1rec_dir @ W1rec_dir.t())

# --- magnitude: sigmoid tail is a sum of exp(-a*t*...); fit the decay rate a per neuron ---
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
    swing = (bb.query(X0 + (6.0 / a_g) * vj) - bb.query(X0 - (6.0 / a_g) * vj)).norm(dim=1)
    top = torch.topk(swing, nctx).indices
    tp = torch.linspace(2.5 / a_g, 7.0 / a_g, 40, device=dev, dtype=torch.float64)
    tm = torch.linspace(-7.0 / a_g, -2.5 / a_g, 40, device=dev, dtype=torch.float64)
    tails = []
    for mi in top.tolist():
        x0 = X0[mi]
        Fp = bb.query(x0.unsqueeze(0) + tp.unsqueeze(1) * vj.unsqueeze(0)).cpu().numpy()
        Fm = bb.query(x0.unsqueeze(0) + tm.unsqueeze(1) * vj.unsqueeze(0)).cpu().numpy()
        for r in range(O):
            tails.append((tp.cpu().numpy(), Fp[:, r], +1)); tails.append((tm.cpu().numpy(), Fm[:, r], -1))
    a_hat[jj] = fit_mag(tails, a_g)
W1_after = a_hat[:, None] * N
# ================= END SOLVE =================

# ---------- scoring ONLY: now the true weights may be looked at ----------
W1t = _true_W1 = torch.load(CD + "_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",
                            map_location=dev, weights_only=False)["teacher_state"]["layers.0.weight"].to(dev).double()
nt = W1t.norm(dim=1)
def aligned(W, tag):
    Cp = torch.cdist(W, W1t); Cm = torch.cdist(-W, W1t); C = torch.minimum(Cp, Cm).cpu().numpy()
    ri, ci = linear_sum_assignment(C)
    e = torch.tensor([C[ri[t], ci[t]] for t in range(len(ri))]); rel = e / nt[ci].cpu()
    print(f"  {tag:26s} abs: mean {float(e.mean()):.3e} max {float(e.max()):.3e} | "
          f"rel: mean {float(rel.mean()):.3e} max {float(rel.max()):.3e}")

print()
print(f"[oracle usage] query() calls: {bb.n_calls}   total forward-evaluated input rows: {bb.n_rows}")
print("first-layer WEIGHT-row error under alignment (real consensus guess -> refined), BLACK-BOX ONLY:")
aligned(Wg, "BEFORE (real consensus)")
aligned(W1_after, "AFTER  (refined dir+mag)")
