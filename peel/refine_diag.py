"""Diagnose base-point sensitivity: refine ONE neuron over many base points,
show the spread of max_eps, and whether averaging the recovered directions
(or picking base points far from other hyperplanes) rescues it."""
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
tn = W1t / W1t.norm(dim=1, keepdim=True); gn = W1g / W1g.norm(dim=1, keepdim=True)

def sig_p(z): s = torch.sigmoid(z); return s*(1-s)

@torch.no_grad()
def J_at(x, fd=5e-4):
    E = torch.eye(d, device=dev, dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E) - teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()

def maxeps(u, tk):
    s = 1.0 if float(u@tn[tk])>0 else -1.0
    return float((s*u/u.norm() - tn[tk]).abs().max())

tk = 0
th = math.radians(5.0)
g = torch.Generator(device=dev).manual_seed(123)
v = torch.randn(d, generator=g, device=dev, dtype=torch.float64); v = v-(v@tn[tk])*tn[tk]; v=v/v.norm()
w_guess = math.cos(th)*tn[tk] + math.sin(th)*v
wg = w_guess*W1t[tk].norm(); gk = int((gn@tn[tk]).abs().argmax())
print(f"neuron {tk}: guess max_eps = {maxeps(w_guess, tk):.3e}  (5deg perturb)\n")

M = 40
dirs, mes, closeness = [], [], []
for m in range(M):
    x0 = torch.randn(d, generator=g, device=dev, dtype=torch.float64)
    x0 = x0 - (wg@x0 + b1t[tk])/(wg@wg)*wg
    # how close is this base to OTHER teacher hyperplanes? (min |z_i|/|w_i| over i!=tk)
    z_others = (W1t@x0 + b1t)
    zn = (z_others / W1t.norm(dim=1)).abs(); zn[tk] = 1e9
    closeness.append(float(zn.min()))
    J = J_at(x0)
    sp = sig_p(W1g@x0 + b1g)
    contrib = (W2g*sp.unsqueeze(0))@W1g
    ck = sp[gk]*torch.outer(W2g[:,gk], W1g[gk])
    U,S,Vh = torch.linalg.svd(J-(contrib-ck), full_matrices=False)
    w = Vh[0]; w = w if float(w@w_guess)>0 else -w
    dirs.append(w); mes.append(maxeps(w, tk))

mes_t = torch.tensor(mes)
print(f"per-base refined max_eps:  min {mes_t.min():.3e}  median {mes_t.median():.3e}  max {mes_t.max():.3e}")
# averaging all recovered directions
avg = torch.stack(dirs).mean(0); print(f"average of {M} directions   -> max_eps {maxeps(avg, tk):.3e}")
# best 25% by closeness (base far from other hyperplanes)
import numpy as np
idx = np.argsort(closeness)[::-1][:M//4]
avg_far = torch.stack([dirs[i] for i in idx]).mean(0)
print(f"average of {len(idx)} FARTHEST-base dirs -> max_eps {maxeps(avg_far, tk):.3e}")
print(f"\nbase 'closeness' to nearest other hyperplane: min {min(closeness):.3f} max {max(closeness):.3f}")
print("(if refined max_eps correlates with closeness, isolation is the limiter)")
