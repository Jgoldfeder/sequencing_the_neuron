"""Before/after MAX_EPS (max abs weight error, unit-row frame) for layer-1
refinement of the sigmoid net. Good guess -> refine -> how much does the layer's
max error drop? Refinement is black-box (teacher queried); other neurons use the
solved guess for subtraction."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP

dev = "cuda"
pop = torch.load("recon/_pop__v18_lbfgs_sigmoid__3072x256x100__s0.pt", map_location=dev, weights_only=False)
dims = pop["dims"]; d = dims[0]
teacher = MLP(dims, act="sigmoid").to(dev).double(); teacher.load_state_dict(pop["teacher_state"])
fin = torch.load("recon/v18_lbfgs_sigmoid__3072x256x100__s0_final.pt", map_location=dev, weights_only=False)
guess = MLP(dims, act="sigmoid").to(dev).double(); guess.load_state_dict(fin["state_dict"])

W1t = teacher.layers[0].weight.detach(); b1t = teacher.layers[0].bias.detach()
W1g = guess.layers[0].weight.detach(); b1g = guess.layers[0].bias.detach(); W2g = guess.layers[1].weight.detach()
tn = W1t / W1t.norm(dim=1, keepdim=True)
gn = W1g / W1g.norm(dim=1, keepdim=True)

def sig_p(z):
    s = torch.sigmoid(z); return s * (1 - s)

@torch.no_grad()
def teacher_J(x, fd=5e-4):
    E = torch.eye(d, device=dev, dtype=torch.float64)
    return ((teacher(x.unsqueeze(0) + fd * E) - teacher(x.unsqueeze(0) - fd * E)) / (2 * fd)).t()

@torch.no_grad()
def refine_dir(tk, w_guess_unit, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x0 = torch.randn(d, generator=g, device=dev, dtype=torch.float64)
    wg = w_guess_unit * W1t[tk].norm()                        # give guess ~true scale for base
    x0 = x0 - (wg @ x0 + b1t[tk]) / (wg @ wg) * wg
    J = teacher_J(x0)
    gk = int((gn @ tn[tk]).abs().argmax())
    sp = sig_p(W1g @ x0 + b1g)
    contrib = (W2g * sp.unsqueeze(0)) @ W1g
    contrib_gk = sp[gk] * torch.outer(W2g[:, gk], W1g[gk])
    U, S, Vh = torch.linalg.svd(J - (contrib - contrib_gk), full_matrices=False)
    w = Vh[0]
    return w if float(w @ w_guess_unit) > 0 else -w

def maxeps(u_est, tk):                                         # max abs comp err, unit-row, sign-aligned
    ut = tn[tk]; s = 1.0 if float(u_est @ ut) > 0 else -1.0
    return float((s * u_est - ut).abs().max())

N = 40
for start_deg in (2.0, 5.0):
    before, after = [], []
    for tk in range(N):
        g = torch.Generator(device=dev).manual_seed(tk)
        wt = tn[tk]
        v = torch.randn(d, generator=g, device=dev, dtype=torch.float64)
        v = v - (v @ wt) * wt; v = v / v.norm()
        th = math.radians(start_deg)
        w_guess = math.cos(th) * wt + math.sin(th) * v        # unit good guess
        before.append(maxeps(w_guess, tk))
        w_ref = refine_dir(tk, w_guess, seed=tk)
        after.append(maxeps(w_ref, tk))
    print(f"good guess perturbed {start_deg}deg  (over {N} neurons):")
    print(f"   BEFORE  layer-1 max_eps = {max(before):.3e}   (median {sorted(before)[N//2]:.3e})")
    print(f"   AFTER   layer-1 max_eps = {max(after):.3e}   (median {sorted(after)[N//2]:.3e})")
    print(f"   -> max_eps shrinks {max(before)/max(after):.1f}x\n")

# reference: the solved guess's own layer-1 max_eps (unit-row), for context
solved_me = max(maxeps(gn[int((gn @ tn[t]).abs().argmax())], t) for t in range(N))
print(f"[context] solved guess layer-1 max_eps (unit-row, same {N} neurons) = {solved_me:.3e}")
