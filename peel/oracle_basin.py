"""Oracle basin experiment (chat's decisive test).
A: downstream = TRUE, perturb only (a,b) by eps -> does joint LM reach machine
   precision? (verifies the endgame).
B: (a,b) at 1e-3, downstream = TRUE + rho*noise -> sweep rho, find the basin
   radius where LM still converges."""
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
dv_true=torch.cat([teacher.layers[1].weight.reshape(-1),teacher.layers[1].bias,
                   teacher.layers[2].weight.reshape(-1),teacher.layers[2].bias,
                   teacher.layers[3].weight.reshape(-1),teacher.layers[3].bias]).detach()
print(f"sanity: loss at true params = {loss(torch.cat([a_true,b1t,dv_true])):.2e}\n")
P=2*k+ndown
def run_lm(theta, niter=9):
    lam=1e-4
    for _ in range(niter):
        r=resid(theta); J=jacrev(resid)(theta); JtJ=J.t()@J; Jtr=J.t()@r; dgn=JtJ.diag().clamp_min(1e-30)
        for _ in range(10):
            delta=-torch.linalg.solve(JtJ+lam*torch.diag(dgn),Jtr); thn=theta+delta
            if loss(thn)<loss(theta): theta=thn; lam=max(lam/3,1e-14); break
            else: lam*=4
        if loss(theta)<1e-26: break
    return theta

print("=== A: downstream=TRUE, perturb (a,b) ===")
for eps in (1e-2,1e-3,1e-4):
    a0=a_true+eps*(2*torch.rand(k,generator=g,device=dev,dtype=torch.float64)-1)
    b0=b1t+eps*(2*torch.rand(k,generator=g,device=dev,dtype=torch.float64)-1)
    th=run_lm(torch.cat([a0,b0,dv_true.clone()]))
    print(f"  eps={eps:.0e}: final (a,b) err {aberr(th):.2e}   loss {loss(th):.2e}")

print("\n=== B: (a,b) at 1e-3, downstream = TRUE + rho*noise (basin radius) ===")
for rho in (1e-1,3e-2,1e-2,3e-3,1e-3):
    a0=a_true+1e-3*(2*torch.rand(k,generator=g,device=dev,dtype=torch.float64)-1)
    b0=b1t+1e-3*(2*torch.rand(k,generator=g,device=dev,dtype=torch.float64)-1)
    xi=torch.randn(ndown,generator=g,device=dev,dtype=torch.float64)
    dv0=dv_true+rho*xi/xi.norm()*math.sqrt(ndown)      # per-param scale ~rho
    th=run_lm(torch.cat([a0,b0,dv0]))
    print(f"  rho={rho:.0e}: final (a,b) err {aberr(th):.2e}   loss {loss(th):.2e}")
