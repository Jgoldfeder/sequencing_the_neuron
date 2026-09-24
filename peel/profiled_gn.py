"""Profiled Gauss-Newton (chat's endgame): from a post-singularity guess (a,b ~1e-2
off, directions exact), fit the downstream, then step (a,b) using ONLY the residual
the downstream can't explain: d = -(A^T P_perp A + lI)^-1 A^T P_perp r, where
P_perp = I - P_B removes the downstream-compensable directions. Does (a,b) -> 1e-4?"""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from torch.func import jacfwd, jacrev
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]; O=dims[-1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
Ndir=W1t/W1t.norm(dim=1,keepdim=True); a_true=W1t.norm(dim=1); g=torch.Generator(device=dev).manual_seed(1)
W1pinv=W1t.t()@torch.linalg.inv(W1t@W1t.t())
Npts=220; r0=torch.randn(Npts,k,generator=g,device=dev,dtype=torch.float64)*2.5
X=(r0-b1t)@W1pinv.t()
with torch.no_grad(): Y=teacher(X).reshape(-1)

shapes=[(32,24),(32,),(16,32),(16,),(8,16),(8,)]
def unflat(v):
    o=[]; i=0
    for sh in shapes:
        n=int(torch.tensor(sh).prod()); o.append(v[i:i+n].reshape(sh)); i+=n
    return o
def forward(a,b,dv):
    W1=a[:,None]*Ndir; h=torch.sigmoid(X@W1.t()+b); p=unflat(dv)
    h=torch.sigmoid(h@p[0].t()+p[1]); h=torch.sigmoid(h@p[2].t()+p[3])
    return (h@p[4].t()+p[5]).reshape(-1)

# init: directions exact, a,b ~1e-2 off (post-singularity level); downstream random
a=(a_true+1e-2*(2*torch.rand(k,generator=g,device=dev,dtype=torch.float64)-1)).clone()
b=(b1t+1e-2*(2*torch.rand(k,generator=g,device=dev,dtype=torch.float64)-1)).clone()
ndown=sum(int(torch.tensor(sh).prod()) for sh in shapes)
dv=(0.3*torch.randn(ndown,generator=g,device=dev,dtype=torch.float64))
def err(): return max(float((a-a_true).abs().max()), float((b-b1t).abs().max()))
print(f"init (a,b) max error: {err():.3e}")

def fit_down():                                    # fit downstream to its FLOOR (LBFGS)
    global dv; dv=dv.detach().requires_grad_(True)
    opt=torch.optim.LBFGS([dv],lr=1.0,max_iter=100,history_size=60,line_search_fn='strong_wolfe')
    def cl(): opt.zero_grad(); l=((forward(a,b,dv)-Y)**2).mean(); l.backward(); return l
    for _ in range(30): opt.step(cl)
    dv=dv.detach(); return float(((forward(a,b,dv)-Y)**2).mean())
for gn in range(6):
    loss=fit_down()
    Ja,Jb=jacfwd(forward,argnums=(0,1))(a,b,dv); A=torch.cat([Ja,Jb],1)
    B=jacrev(forward,argnums=2)(a,b,dv); Qb,_=torch.linalg.qr(B)
    r=(forward(a,b,dv)-Y)
    Pr=r-Qb@(Qb.t()@r); PA=A-Qb@(Qb.t()@A)
    H=PA.t()@PA; gvec=PA.t()@Pr; lam=1e-6*float(H.diag().max())
    delta=-torch.linalg.solve(H+lam*torch.eye(2*k,device=dev,dtype=torch.float64), gvec)
    # backtracking on the projected residual norm
    step=1.0
    for _ in range(6):
        an=a+step*delta[:k]; bn=b+step*delta[k:]
        rn=forward(an,bn,dv)-Y; rn=rn-Qb@(Qb.t()@rn)
        if float((rn**2).sum())<float((Pr**2).sum()): break
        step*=0.5
    a=a+step*delta[:k]; b=b+step*delta[k:]
    print(f"  GN step {gn}: (a,b) max error {err():.3e}  down-loss {loss:.2e}  step {step:.3f}")
