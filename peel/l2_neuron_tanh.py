"""Layer-2 tanh neuron refinement with layer 1 known exactly, deeper layers
unknown. Black box f(x) = phi(tanh(W2 a + b2)), a = tanh(W1 x + b1).

Inputs to layer 2 are confined to the cube (-1,1)^128, so the layer-1 trick
(saturate every OTHER neuron) is unavailable. Instead saturate neuron j ITSELF
at the corner a = sign(w_hat_j): there its row drops out of the Jacobian, so
the stacked Jacobians span S_{-j} = span{w_k : k != j}. The direction of
rowspace(W2) orthogonal to S_{-j} is the DUAL vector u_j (w_k . u_j = 0, k != j).
One guess -> one dual. All 80 duals -> W2 rows = rows of pinv(U) (up to scale).
"""
import os, sys, math, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nets import MLP
from tanh_solve import Oracle

torch.manual_seed(0)
dev = "cuda"
arch = [784, 128, 80, 40, 32, 10]
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "teachers/teacher_784x128x80x40x32x10_e25_s0_tanh.pt")
NOISE = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2
NPTS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5

T = MLP(arch, act="tanh").to(dev, torch.float64)
T.load_state_dict({k: v.double() for k, v in torch.load(path, map_location="cpu").items()})
bb = Oracle(lambda x: T(x))

# ---- KNOWN: layer 1 -------------------------------------------------------
W1 = T.layers[0].weight.detach(); b1 = T.layers[0].bias.detach()
W1p = torch.linalg.pinv(W1)                                  # (784, 128)
n1 = W1.shape[0]

def g_of_z(Z):                     # black box as a function of L1 pre-activation
    return bb((Z - b1) @ W1p.T)

def jac_a(A, h=1e-3):
    """Jacobian of the black box w.r.t. a = L1 output, at rows of A (N,128).
    FD is taken in z1 (unbounded), then divided by da/dz = 1 - a^2."""
    Z = torch.atanh(A)
    N = len(Z)
    E = torch.eye(n1, dtype=torch.float64, device=dev)
    f = [g_of_z((Z[:, None, :] + s * h * E[None]).reshape(-1, n1)).reshape(N, n1, -1)
         for s in (1.0, -1.0, 2.0, -2.0)]
    Jz = (8 * (f[0] - f[1]) - (f[2] - f[3])) / (12 * h)       # (N, 128, 10)
    return (Jz / (1 - A * A)[:, :, None]).transpose(1, 2)     # (N, 10, 128)

# ---- TRUTH: scoring only --------------------------------------------------
W2 = T.layers[1].weight.detach(); b2 = T.layers[1].bias.detach()
n2 = W2.shape[0]
U_true = torch.linalg.pinv(W2)                                # columns = duals
def derr(u, v):
    u = u / u.norm(); v = v / v.norm()
    return float(torch.minimum((u - v).norm(), (u + v).norm()))

# ---- the guesses (one per neuron; each is used ONLY for its own neuron) -----
Nz = torch.randn_like(W2)
W2g = W2 + NOISE * W2.norm(dim=1, keepdim=True) * Nz / Nz.norm(dim=1, keepdim=True)

# ---- row space S of W2 from Jacobians at interior points --------------------
A = (torch.rand(40, n1, dtype=torch.float64, device=dev) * 2 - 1) * 0.9
J = jac_a(A).reshape(-1, n1)
Sg, Vh = torch.linalg.svd(J, full_matrices=False)[1:]
r = int((Sg > 1e-7 * Sg[0]).sum())
VS = Vh[:r]                                                    # (80, 128)
print(f"rowspace(W2): rank {r} (sv[{r-1}]/sv0 {float(Sg[r-1]/Sg[0]):.1e}, sv[{r}]/sv0 "
      f"{float(Sg[r]/Sg[0]):.1e}) | queries {bb.n}")

def dual_from_guess(wg):
    sgn = torch.sign(wg)
    rr = torch.rand(NPTS, n1, dtype=torch.float64, device=dev) * 0.8 + 0.2
    delta = BUDGET * rr / (n1 * 0.6 * wg.abs()[None])
    delta = delta.clamp(0.02, 1.2)
    A = sgn[None] * (1 - delta)
    A = A.clamp(-0.98, 0.98)
    J = jac_a(A).reshape(-1, n1) @ VS.T                        # coords in S
    Sg, Vh = torch.linalg.svd(J, full_matrices=False)[1:]
    return VS.T @ Vh[-1], float(Sg[-1] / Sg[0]), float(Sg[-2] / Sg[0]), A

U = torch.zeros(n1, n2, dtype=torch.float64, device=dev)
errs, zs = [], []
for j in range(n2):
    u, s_last, s_prev, A = dual_from_guess(W2g[j])
    U[:, j] = u
    e = derr(u, U_true[:, j])
    zmin = float((A @ W2[j] + b2[j]).abs().min())               # diagnostic
    errs.append(e); zs.append(zmin)
    if j < 5:
        print(f"neuron {j}: |w|_1 {float(W2[j].abs().sum()):.2f} min|z_j| at probes {zmin:.2f} | "
              f"sv79/sv0 {s_prev:.1e} sv80/sv0 {s_last:.1e} | DUAL dir err {e:.2e}")
errs = torch.tensor(errs)
print(f"DUAL (single guess each): median {float(errs.median()):.2e} max {float(errs.max()):.2e} "
      f"| queries {bb.n}")

# ---- single neuron: what does the dual buy for the PRIMAL row? -------------
# project the guess so that it is consistent with everything measured for j alone
j = 0
print(f"single-neuron primal (neuron 0): guess dir err {derr(W2g[0], W2[0]):.2e}; "
      f"projected on rowspace {derr(VS.T @ (VS @ W2g[0]), W2[0]):.2e}")

# ---- all duals -> primal rows ------------------------------------------------
West = torch.linalg.pinv(U).T.T                                 # (80, 128) rows ∝ w_j
perr = torch.tensor([derr(West[j], W2[j]) for j in range(n2)])
print(f"PRIMAL rows from all 80 duals: median {float(perr.median()):.2e} max {float(perr.max()):.2e}")
gerr = torch.tensor([derr(W2g[j], W2[j]) for j in range(n2)])
print(f"  (guess was median {float(gerr.median()):.2e})")
torch.save(dict(U=U.cpu(), West=West.cpu(), VS=VS.cpu()), "/tmp/claude-1001/-home-judah/6f297c87-053c-4eab-ac4d-45ceffd92724/scratchpad/l2_duals.pt")
print(f"total queries {bb.n}")
