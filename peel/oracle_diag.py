"""ORACLE diagnostic (uses true teacher params -- diagnosis only): is 1e-4 in the
magnitude/bias even VISIBLE after the downstream network compensates optimally?
A = dF/d(a_j,b_j) (first-layer mag/bias sensitivities, using true directions),
B = dF/d(downstream params). A_eff = (I - P_B) A is what the downstream can't fake.
Report sigma_min per neuron: a 1e-4 param change makes a functional change
>= sigma_min * 1e-4; compare to numerical floor."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from torch.func import functional_call, jacrev
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]; O=dims[-1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
Nrm=W1t/W1t.norm(dim=1,keepdim=True); g=torch.Generator(device=dev).manual_seed(1)
W1pinv=W1t.t()@torch.linalg.inv(W1t@W1t.t())

# query points in first-layer coords -- need N*O >> #downstream params (1464)
Npts=600; r=torch.randn(Npts,k,generator=g,device=dev,dtype=torch.float64)*2.5
X=(r-b1t)@W1pinv.t()      # N*O = 4800 >> 1464

params={n:p.detach().clone() for n,p in teacher.named_parameters()}
def fwd(p): return functional_call(teacher,p,(X,)).reshape(-1)      # (Npts*O,)
J=jacrev(fwd)(params)
NO=Npts*O
W0=J['layers.0.weight'].reshape(NO,k,d); b0=J['layers.0.bias'].reshape(NO,k)
dFda=torch.einsum('okd,kd->ok',W0,Nrm)                             # (NO,k) dF/da_j
dFdb=b0                                                            # (NO,k) dF/db_j
# downstream Jacobian B
Bcols=[]
for n,Jt in J.items():
    if n.startswith('layers.0'): continue
    Bcols.append(Jt.reshape(NO,-1))
B=torch.cat(Bcols,1)                                               # (NO, n_down)
print(f"outputs {NO}, downstream params {B.shape[1]}")
Qb,_=torch.linalg.qr(B)                                            # orthonormal basis of downstream span
def proj_off(M): return M-Qb@(Qb.t()@M)                            # (I-P_B)
Ea=proj_off(dFda); Eb=proj_off(dFdb)                               # nuisance-projected

print("\nper-neuron sigma_min of (I-P_B)[dF/da_j, dF/db_j]  (raw = before projection):")
smin=[]
for j in range(k):
    raw=torch.linalg.svdvals(torch.stack([dFda[:,j],dFdb[:,j]],1))[-1]
    eff=torch.linalg.svdvals(torch.stack([Ea[:,j],Eb[:,j]],1))[-1]
    smin.append(float(eff))
    if j<3 or float(eff)==min(smin):
        print(f"  n{j:2d}: raw {float(raw):.3e}  ->  after downstream projection {float(eff):.3e}")
smin=torch.tensor(smin)
print(f"\nsigma_min(A_eff) across neurons: min {smin.min():.3e}  median {smin.median():.3e}")
worst=int(smin.argmin())
print(f"worst neuron n{worst}: sigma_min = {float(smin[worst]):.3e}")
print(f"  -> a 1e-4 change in (a,b) makes a functional change >= {float(smin[worst])*1e-4:.2e}")
print(f"  -> a 1e-3 change makes >= {float(smin[worst])*1e-3:.2e}")
print(f"  (query values are float64 ~1e-15; realistic fit floor ~1e-8. If the 1e-4")
print(f"   number is >> that floor, 1e-4 is VISIBLE and my 'impossible' claim is wrong.)")
