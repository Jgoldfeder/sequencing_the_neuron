"""Joint Levenberg-Marquardt over ALL student params (a,b,downstream) to fit the
black box. Model = exact architecture, so true min has loss 0 and (a,b) is
non-degenerate (oracle sigma_min 2e-2) -> LM should reach machine precision.
Init: directions exact, (a,b) 1e-2 off (post-singularity), downstream Adam-warmed."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from torch.func import jacrev
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]; O=dims[-1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
Ndir=W1t/W1t.norm(dim=1,keepdim=True); a_true=W1t.norm(dim=1); g=torch.Generator(device=dev).manual_seed(1)
W1pinv=W1t.t()@torch.linalg.inv(W1t@W1t.t())
Npts=210; r0=torch.randn(Npts,k,generator=g,device=dev,dtype=torch.float64)*2.5
X=(r0-b1t)@W1pinv.t()
with torch.no_grad(): Y=teacher(X).reshape(-1)
shapes=[(32,24),(32,),(16,32),(16,),(8,16),(8,)]; ndown=sum(int(torch.tensor(s).prod()) for s in shapes)
def unflat(v):
    o=[]; i=0
    for sh in shapes:
        n=int(torch.tensor(sh).prod()); o.append(v[i:i+n].reshape(sh)); i+=n
    return o
def resid(theta):
    a=theta[:k]; b=theta[k:2*k]; p=unflat(theta[2*k:])
    h=torch.sigmoid(X@(a[:,None]*Ndir).t()+b)
    h=torch.sigmoid(h@p[0].t()+p[1]); h=torch.sigmoid(h@p[2].t()+p[3])
    return (h@p[4].t()+p[5]).reshape(-1)-Y
def loss(th): return float((resid(th)**2).sum())
def aberr(th): return max(float((th[:k]-a_true).abs().max()),float((th[k:2*k]-b1t).abs().max()))

EPS0=1e-3   # tighter init (does a convergent basin exist?)
a0=a_true+EPS0*(2*torch.rand(k,generator=g,device=dev,dtype=torch.float64)-1)
b0=b1t+EPS0*(2*torch.rand(k,generator=g,device=dev,dtype=torch.float64)-1)
dv=0.3*torch.randn(ndown,generator=g,device=dev,dtype=torch.float64)
theta=torch.cat([a0,b0,dv]).clone()
# warm the downstream with Adam so LM starts in-basin
dvp=theta[2*k:].detach().requires_grad_(True)
opt=torch.optim.Adam([dvp],lr=3e-3)
for _ in range(4000):
    th=torch.cat([theta[:2*k].detach(),dvp]); l=(resid(th)**2).mean(); opt.zero_grad(); l.backward(); opt.step()
theta=torch.cat([theta[:2*k].detach(),dvp.detach()])
print(f"init: loss {loss(theta):.2e}  (a,b) max err {aberr(theta):.3e}")

lam=1e-3; P=2*k+ndown
for lm in range(14):
    r=resid(theta); J=jacrev(resid)(theta)                 # (NO, P)
    JtJ=J.t()@J; Jtr=J.t()@r; dg=JtJ.diag().clamp_min(1e-30)
    ok=False
    for _ in range(8):
        delta=-torch.linalg.solve(JtJ+lam*torch.diag(dg), Jtr)
        thn=theta+delta
        if loss(thn)<loss(theta): theta=thn; lam=max(lam/3,1e-12); ok=True; break
        else: lam*=4
    print(f"  LM {lm}: loss {loss(theta):.2e}  (a,b) max err {aberr(theta):.3e}  lam {lam:.1e}")
    if not ok or loss(theta)<1e-24: break
