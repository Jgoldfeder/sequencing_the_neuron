"""Single-neuron refinement for a tanh MLP, knowing ONLY a guess of one
layer-1 row. Nothing else about the net (other L1 rows, deeper layers) is used.

Idea (saturation isolation): walk far out along v with v . w_hat = 0. Every
other L1 neuron saturates (sech^2 -> 0 exponentially), neuron i does not, so
J_f(x) = G D W1 collapses to (downstream column) x w_i^T: rank one, row = w_i.
"""
import sys, math, torch
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
from nets import MLP
from tanh_solve import Oracle, measure_jets

torch.manual_seed(0)
dev = "cuda"
arch = [784, 128, 64, 10]
path = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x64x10_e25_s0_tanh.pt"
NEURON = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NOISE = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-2
USE_SUB = int(sys.argv[4]) if len(sys.argv) > 4 else 1

teacher = MLP(arch, act="tanh").to(dev, torch.float64)
teacher.load_state_dict({k: v.double() for k, v in torch.load(path, map_location="cpu").items()})
bb = Oracle(lambda x: teacher(x))
d = arch[0]

# ---- truth: used ONLY for scoring -------------------------------------
w_true = teacher.layers[0].weight[NEURON].detach().clone()
b_true = float(teacher.layers[0].bias[NEURON])
def dir_err(u):
    u = u / u.norm(); wt = w_true / w_true.norm()
    return float(torch.minimum((u - wt).norm(), (u + wt).norm()))

# ---- the guess --------------------------------------------------------
n = torch.randn(d, dtype=torch.float64, device=dev)
w_hat = w_true + NOISE * w_true.norm() * n / n.norm()
print(f"neuron {NEURON}: |w|={float(w_true.norm()):.3f} b={b_true:.4f} | guess dir err {dir_err(w_hat):.3e}")

# ---- optional: row space of W1 from Jacobians at random points ----------
Vsub = None
if USE_SUB:
    X = torch.randn(40, d, dtype=torch.float64, device=dev)
    _, J = measure_jets(bb, X, None, h=1e-3)             # (40, 10, d)
    S, Vh = torch.linalg.svd(J.reshape(-1, d), full_matrices=False)[1:]
    r = int((S > 1e-8 * S[0]).sum())
    Vsub = Vh[:r]                                          # (r, d)
    print(f"row space of W1: numerical rank {r} (sv[r-1]/sv[0]={float(S[r-1]/S[0]):.1e}, "
          f"sv[r]/sv[0]={float(S[r]/S[0]):.1e}) | queries {bb.n}")
    print(f"  guess projected onto subspace: dir err {dir_err(Vsub.T @ (Vsub @ w_hat)):.3e}")
    w_hat = Vsub.T @ (Vsub @ w_hat)

def rand_perp(u):
    g = torch.randn(d, dtype=torch.float64, device=dev)
    if Vsub is not None:
        g = Vsub.T @ (Vsub @ g)
    g = g - (g @ u) * u / (u @ u)
    return g / g.norm()

def centre(x0, u, S, n=801, h=1e-3):
    """s maximising |d/ds f(x0 + s u)| on [-S, S] (neuron i's active slab)."""
    s = torch.linspace(-S, S, n, dtype=torch.float64, device=dev)
    P = x0[None] + s[:, None] * u[None]
    g = (bb(P + h * u) - bb(P - h * u)) / (2 * h)
    return float(s[g.norm(dim=1).argmax()]), g.norm(dim=1)

for rnd, (t, M) in enumerate([(30, 6), (100, 6), (300, 6), (1000, 6), (3000, 6)]):
    u = w_hat / w_hat.norm()
    Psum = torch.zeros(d, d, dtype=torch.float64, device=dev)
    ranks, used = [], 0
    for m in range(M):
        v = rand_perp(u)
        S_win = 3.0 + 3.0 * t * max(dir_err(w_hat) if False else 0.0, 0.0)  # no truth used
        S_win = 3.0 + 0.05 * t if rnd == 0 else 5.0
        s_star, prof = centre(t * v, u, S_win)
        x = (t * v + s_star * u)[None]
        _, J = measure_jets(bb, x, None, h=1e-3)
        Sg, Vh = torch.linalg.svd(J[0], full_matrices=False)[1:]
        k = int((Sg > 1e-10 * Sg[0]).sum())
        ranks.append(f"{float(Sg[1]/Sg[0]):.0e}")
        if float(Sg[0]) < 1e-9:
            continue
        Psum += Vh[:k].T @ Vh[:k]; used += 1
    ev, evec = torch.linalg.eigh(Psum)
    cand = evec[:, -1]
    if cand @ u < 0:
        cand = -cand
    print(f"round {rnd} t={t:5d}: sv2/sv1 per v [{' '.join(ranks)}] | common-dir eig {float(ev[-1]):.6f}/{used} "
          f"(next {float(ev[-2]):.3f}) | dir err {dir_err(w_hat):.3e} -> {dir_err(cand):.3e} | queries {bb.n}")
    if float(ev[-1]) > used - 1e-3:        # a direction common to all row spaces: accept
        w_hat = cand * w_hat.norm()

# ---- magnitude: exponential decay rate of the tail along u ----------------
u = w_hat / w_hat.norm()
t = 3000; best = None
for m in range(12):                      # pick the cleanest rank-1 probe line
    v = rand_perp(u)
    s0, _ = centre(t * v, u, 5.0)
    _, J = measure_jets(bb, (t * v + s0 * u)[None], None, h=1e-3)
    Sg = torch.linalg.svdvals(J[0])
    if best is None or float(Sg[1] / Sg[0]) < best[0]:
        best = (float(Sg[1] / Sg[0]), v, s0)
print(f"probe line sv2/sv1 = {best[0]:.1e}")
_, v, s0 = best
s = torch.linspace(s0 - 8, s0 + 8, 3201, dtype=torch.float64, device=dev)
P = t * v[None] + s[:, None] * u[None]
h = 1e-3
g = ((8 * (bb(P + h * u) - bb(P - h * u)) - (bb(P + 2 * h * u) - bb(P - 2 * h * u))) / (12 * h)).norm(dim=1)
lg = g.log()
def slope(mask):
    A = torch.stack([s[mask], torch.ones_like(s[mask])], 1)
    return torch.linalg.lstsq(A, lg[mask][:, None]).solution.flatten()
hi = (g < 1e-4 * g.max()) & (g > 1e-8 * g.max())
aR = slope(hi & (s > s0)); aL = slope(hi & (s < s0))
alpha = (abs(float(aR[0])) + abs(float(aL[0]))) / 4
wn = float(w_true.norm())
print(f"magnitude: tail slopes {float(aL[0]):.8f} / {float(aR[0]):.8f} -> |w| = {alpha:.8f} "
      f"(true {wn:.8f}, rel err {abs(alpha/wn-1):.2e}; guess rel err {abs(float(w_hat.norm())/wn-1):.2e})")
s_c = float((aR[1] - aL[1]) / (aL[0] - aR[0]))
z_true_at = float(w_true @ (t * v + s_c * u)) + b_true
print(f"bias: tail-symmetric centre s_c={s_c:.5f}; TRUE preactivation there = {z_true_at:+.4f} "
      f"(0 would mean the bias is readable off this neuron alone)")
print(f"total queries {bb.n}")
