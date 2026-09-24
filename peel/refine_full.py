"""FULL refinement of a whole-first-layer guess on a DEEP sigmoid net, doing
everything we can:
  (0) start: whole first layer guessed 5deg off, nothing downstream
  (1) measure row space (guess-free) and project  -> kills off-subspace error
  (2) iterate the nonlinear within-subspace refinement: measure downstream
      Jacobian D(x)=J*M^+ from the current guess, subtract guessed OTHER neurons,
      SVD the residual -> refined neuron; re-project; repeat.
Report layer-1 max_eps at each stage -> where does it plateau?
"""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]           # 3 hidden layers
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True)
def sigp(z): s=torch.sigmoid(z); return s*(1-s)
@torch.no_grad()
def J_at(x, fd=1e-3):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()
g=torch.Generator(device=dev).manual_seed(2)

# --- (1) row space, guess-free ---
rows=[J_at(torch.randn(d,generator=g,device=dev,dtype=torch.float64)*2) for _ in range(8)]
_,_,Vh=torch.linalg.svd(torch.cat(rows,0),full_matrices=False); B=Vh[:k]   # (k,d) basis

def report(Wg, tag):
    errs=[]
    for kk in range(k):
        u=Wg[kk]/Wg[kk].norm(); s=1.0 if float(u@tn[kk])>0 else -1.0
        errs.append(float((s*u-tn[kk]).abs().max()))
    print(f"  {tag:32s} layer-1 max_eps = {max(errs):.3e}  (median {sorted(errs)[k//2]:.3e})")
    return max(errs)

# --- (0) initial guess: 5deg off ---
Wg=torch.empty(k,d,device=dev,dtype=torch.float64)
for kk in range(k):
    u=tn[kk]; v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
    Wg[kk]=(math.cos(math.radians(5))*u+math.sin(math.radians(5))*v)*W1t[kk].norm()
report(Wg, "(0) initial guess")

# --- project onto measured subspace ---
Wg=(Wg@B.t())@B
report(Wg, "(1) after row-space projection")

# --- (2) iterate within-subspace nonlinear refinement ---
b1g=b1t.clone()                                        # (assume bias ~known; refine directions)
@torch.no_grad()
def refine_iter(Wg, M_base=16):
    New=Wg.clone()
    for kk in range(k):
        wgk=Wg[kk]; acc=torch.zeros(d,device=dev,dtype=torch.float64); used=0; tries=0
        while used<M_base and tries<6*M_base:
            tries+=1
            x0=torch.randn(d,generator=g,device=dev,dtype=torch.float64)
            x0=x0-(wgk@x0+b1g[kk])/(wgk@wgk)*wgk
            zn=((W1t@x0+b1t)/W1t.norm(dim=1)).abs(); zn[kk]=9
            if float(zn.min())<0.03: continue
            used+=1
            J=J_at(x0)                                 # teacher Jacobian (black box)
            z1g=Wg@x0+b1g; s1=sigp(z1g)
            M=s1.unsqueeze(1)*Wg                       # diag(sig') Wg  (k,d)
            D=J@ (M.t()@torch.linalg.inv(M@M.t()))     # D = J M^+   (O,k)  downstream Jacobian
            contrib=(D*s1.unsqueeze(0))@Wg             # sum_i s1_i D[:,i] w_i  (O,d)
            ck=s1[kk]*torch.outer(D[:,kk],Wg[kk])
            _,_,Vh=torch.linalg.svd(J-(contrib-ck),full_matrices=False)
            w=Vh[0]; acc=acc+(w if float(w@wgk)>0 else -w)
        New[kk]=acc/acc.norm()*W1t[kk].norm()
    New=(New@B.t())@B                                  # keep in the exact subspace
    return New

prev=1e9
for it in range(1,7):
    Wg=refine_iter(Wg)
    e=report(Wg, f"(2) after iteration {it}")
    if e>prev*0.97: print("  (plateaued)"); break
    prev=e
