"""Guess of the ENTIRE first layer, nothing downstream. Refine it.

Key downstream-free fact: f depends on x only through z = W1 x + b1, so EVERY
Jacobian row lies in row(W1). Collect J(x) at a few points -> their span IS the
true k-dim row space of W1, recovered exactly with no downstream info. Project
each guessed neuron onto it -> removes all out-of-subspace error at once. The
row space is shared, so it costs ~2d queries TOTAL (not per neuron).
"""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,16,8]; d=dims[0]; k=dims[1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True)

@torch.no_grad()
def J_at(x, fd=1e-3):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()  # (O,d)

# --- measure the row space of W1 from a few black-box Jacobians (downstream-free) ---
g=torch.Generator(device=dev).manual_seed(2)
rows=[]; npts=8
for _ in range(npts):
    x=torch.randn(d,generator=g,device=dev,dtype=torch.float64)*2
    rows.append(J_at(x))
M=torch.cat(rows,0)                                   # (npts*O, d), all in row(W1)
U,S,Vh=torch.linalg.svd(M, full_matrices=False)
basis=Vh[:k]                                          # (k,d) orthonormal ~ row(W1)
q_rowspace = npts*2*d
# sanity: how well does the measured subspace capture the TRUE rows?
capt=(tn@basis.t()).norm(dim=1)                       # ~1 if row k in subspace
print(f"measured row space from {npts} Jacobians ({q_rowspace} queries total):")
print(f"  captures true rows: min {capt.min():.5f} median {capt.median():.5f} (1=perfect)\n")

def maxeps(u,kk): s=1.0 if float(u@tn[kk])>0 else -1.0; return float((s*u/u.norm()-tn[kk]).abs().max())

# --- a whole-first-layer guess: every row 5deg off ---
for deg in (5.0, 2.0):
    th=math.radians(deg); before=[]; after=[]
    for kk in range(k):
        u=tn[kk]; v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
        w_guess=math.cos(th)*u+math.sin(th)*v
        w_proj=(w_guess@basis.t())@basis                 # project onto measured row space
        before.append(maxeps(w_guess,kk)); after.append(maxeps(w_proj,kk))
    ba=lambda a:(max(a),sorted(a)[len(a)//2])
    b,bm=ba(before); a,am=ba(after)
    print(f"whole-layer guess {deg}deg off:")
    print(f"  BEFORE max_eps {b:.3e} (med {bm:.3e})  ->  AFTER row-proj {a:.3e} (med {am:.3e})   [{b/a:.1f}x]")
