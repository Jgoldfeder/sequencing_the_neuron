"""Does refinement make a GOOD sigmoid guess MORE precise, and how far?

For neuron k: take a good-but-imperfect guess (perturb the true normal by a known
angle theta), keep it black-box, subtract the guessed contributions of the OTHER
neurons from the teacher Jacobian, and read neuron k back off the rank-1 residual.
Sweep theta -> if refined error is far below theta and flat across theta, that flat
value is the FLOOR: how precise we can make a good guess.  (Teacher weights used
only to build the perturbation and to score -- refinement itself is queries-only.)
"""
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
W1g = guess.layers[0].weight.detach();   b1g = guess.layers[0].bias.detach()
W2g = guess.layers[1].weight.detach()
gn = W1g / W1g.norm(dim=1, keepdim=True)
tn = W1t / W1t.norm(dim=1, keepdim=True)

def sig_p(z):
    s = torch.sigmoid(z); return s * (1 - s)

@torch.no_grad()
def teacher_J(x, fd=5e-4):
    E = torch.eye(d, device=dev, dtype=torch.float64)
    return ((teacher(x.unsqueeze(0) + fd * E) - teacher(x.unsqueeze(0) - fd * E)) / (2 * fd)).t()

def ang(a, b):
    return math.degrees(math.acos(min(1.0, abs(float((a / a.norm()) @ (b / b.norm()))))))

@torch.no_grad()
def refine(tk, theta_deg, fd=5e-4, seed=0):
    """perturb teacher neuron tk by theta, refine via other-neuron subtraction."""
    g = torch.Generator(device=dev).manual_seed(seed)
    wt = W1t[tk]; wt_u = wt / wt.norm()
    v = torch.randn(d, generator=g, device=dev, dtype=torch.float64)
    v = v - (v @ wt_u) * wt_u; v = v / v.norm()
    th = math.radians(theta_deg)
    w_guess = (math.cos(th) * wt_u + math.sin(th) * v) * wt.norm()      # perturbed guess (keep |w|)
    b_guess = b1t[tk]
    # base point on the PERTURBED guess plane
    x0 = torch.randn(d, generator=g, device=dev, dtype=torch.float64)
    x0 = x0 - (w_guess @ x0 + b_guess) / (w_guess @ w_guess) * w_guess
    J = teacher_J(x0, fd)
    # subtract guessed OTHER neurons (solved guess), excluding the one matching tk
    gk = int((gn @ tn[tk]).abs().argmax())                             # solved neuron ~ teacher tk
    zg = W1g @ x0 + b1g
    sp = sig_p(zg)
    contrib = (W2g * sp.unsqueeze(0)) @ W1g                            # all guessed neurons
    contrib_gk = sp[gk] * torch.outer(W2g[:, gk], W1g[gk])             # the one we're refining
    J_resid = J - (contrib - contrib_gk)
    U, S, Vh = torch.linalg.svd(J_resid, full_matrices=False)
    w_ref = Vh[0]
    if float(w_ref @ w_guess) < 0: w_ref = -w_ref
    return ang(w_guess, wt), ang(w_ref, wt)

# pick a few teacher neurons whose guess is well-matched (so "others" are good)
test = [int((gn @ tn[t]).abs().argmax() >= 0) and t for t in range(6)]
print("=== refine a good guess: perturb neuron by theta, does refine beat theta? ===")
print("neuron | theta(guess err) -> refined err (vs teacher) | queries/neuron ~", 2*d)
for tk in range(6):
    row = []
    for theta in (0.1, 0.5, 2.0, 5.0):
        g0, r0 = refine(tk, theta)
        row.append(f"{theta:>4}deg->{r0:.4f}")
    print(f"  t{tk:2d} | " + "  ".join(row))

print("\n=== theta sweep on one neuron (does it converge to a FLOOR?) ===")
for theta in (0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
    g0, r0 = refine(0, theta)
    print(f"  guess err {theta:>5}deg  ->  refined {r0:.5f}deg   (shrink {g0/max(r0,1e-9):.0f}x)")
