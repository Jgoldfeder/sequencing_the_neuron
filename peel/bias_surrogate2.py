"""Bias recovery with OUTPUT + JACOBIAN matching (chat's key claim). W1 frozen=true,
beta trainable (init b_hat, 8e-2 off), downstream random. Jacobian matching should
pin beta far better than output matching alone."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch, torch.func as tf
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
W1pinv=W1t.t()@torch.linalg.inv(W1t@W1t.t())
g=torch.Generator(device=dev).manual_seed(1)
b_hat=b1t+0.08*(2*torch.rand(k,generator=g,device=dev,dtype=torch.float64)-1)
print(f"bias guess error (init): {float((b_hat-b1t).abs().max()):.3e}")

def mkX(n,tau=2.5):
    r=torch.randn(n,k,generator=g,device=dev,dtype=torch.float64)*tau
    return (r-b_hat)@W1pinv.t()
X=mkX(16384)
with torch.no_grad(): Y=teacher(X)
# Jacobian-match set: cache teacher Jacobians (finite diff), one-time
Xj=mkX(96)
@torch.no_grad()
def tJ(x,fd=1e-4):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()
Jt=torch.stack([tJ(Xj[i]) for i in range(len(Xj))])       # (96,O,d)

stu=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    stu.layers[0].weight.copy_(W1t); stu.layers[0].bias.copy_(b_hat)
stu.layers[0].weight.requires_grad_(False)
beta=stu.layers[0].bias
down=[p for l in stu.layers[1:] for p in l.parameters()]
opt=torch.optim.Adam([{"params":down,"lr":3e-3},{"params":[beta],"lr":2e-3}])

def stuJ(xb):                                             # student Jacobian batch (O,d), differentiable
    return tf.vmap(tf.jacrev(lambda x: stu(x.unsqueeze(0))[0]))(xb)

for step in range(6001):
    idx=torch.randint(0,len(X),(1024,),generator=g,device=dev)
    Lf=((stu(X[idx])-Y[idx])**2).mean()
    ji=torch.randint(0,len(Xj),(24,),generator=g,device=dev)
    LJ=((stuJ(Xj[ji])-Jt[ji])**2).mean()
    loss=Lf+50.0*LJ
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): beta.clamp_(b_hat-0.08,b_hat+0.08)
    if step in (0,1000,2000,4000,6000):
        be=float((stu.layers[0].bias.detach()-b1t).abs().max())
        print(f"  step {step:4d}: Lf {float(Lf):.2e}  LJ {float(LJ):.2e}  bias max_err {be:.3e}")
