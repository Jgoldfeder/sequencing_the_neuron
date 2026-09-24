"""Test the exponential-tail magnitude estimator (chat's step 2) on a deep sigmoid
net. Direction from saturate+SVD; then along the recovered normal, in the tail
F(t)=F_inf + A e^{-a t}, so successive-difference norms ratio -> e^{a*dt} -> a=||w||."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True); g=torch.Generator(device=dev).manual_seed(1)
@torch.no_grad()
def J_at(x,fd=1e-4):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()

# whole-layer guess: 5deg dir, +-8% mag, +-0.08 bias
Wg=torch.empty(k,d,device=dev,dtype=torch.float64); bg=torch.empty(k,device=dev,dtype=torch.float64)
for j in range(k):
    u=tn[j]; v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
    Wg[j]=(math.cos(math.radians(5))*u+math.sin(math.radians(5))*v)*W1t[j].norm()*(1+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64)-1))
    bg[j]=b1t[j]+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64).item()-1)
Wg_pinv=Wg.t()@torch.linalg.inv(Wg@Wg.t())

@torch.no_grad()
def tail_mag(x0, n, wg, S, dt, sign):
    """a = (1/dt) mean log(||D(t)||/||D(t+dt)||), t in the tail (sign=+/-1)."""
    # tail region: z_j from ~3.5 to ~6.5 (in its exponential regime, others still sat.)
    base=3.5/wg
    ts=base+torch.arange(0,9,device=dev,dtype=torch.float64)*dt  # t, t+dt, ...
    ts=sign*ts
    P=x0.unsqueeze(0)+ts.unsqueeze(1)*n.unsqueeze(0)
    F=teacher(P)                                     # (9,O)
    D=(F[:-1]-F[1:]).norm(dim=1)                      # ||F(t)-F(t+dt)||, decaying
    r=(D[:-1]/D[1:].clamp_min(1e-300)).log()/abs(dt) # each ~ a
    return float(r.median())

Sval=20.0
merr_tail=[]; merr_true=[]; derr=[]
for j in range(k):
    t=torch.full((k,),Sval,device=dev,dtype=torch.float64); t[j]=0.0
    x0=Wg_pinv@(t-bg)
    U,Sv,Vh=torch.linalg.svd(J_at(x0),full_matrices=False)
    n=Vh[0]; n=n if float(n@Wg[j])>0 else -n
    dt=0.4/float(Wg[j].norm())
    a_pos=tail_mag(x0,n,float(Wg[j].norm()),Sval,dt,+1)
    a_neg=tail_mag(x0,n,float(Wg[j].norm()),Sval,dt,-1)
    a=0.5*(a_pos+a_neg)
    merr_tail.append(abs(a-float(W1t[j].norm())))
    derr.append(math.degrees(math.acos(min(1.0,abs(float(n@tn[j]))))))
print(f"direction  worst {max(derr):.2e} deg")
print(f"MAGNITUDE via exponential-tail ratio:  worst err {max(merr_tail):.3e}  median {sorted(merr_tail)[k//2]:.3e}")
print(f"  (guess magnitude error was ~8%% ~ {0.08*float(W1t.norm(dim=1).mean()):.3f}; my old FWHM/fit estimator: 6.4e-3)")
