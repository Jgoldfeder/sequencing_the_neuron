"""Bias recovery via ALTERNATING (chat's recommendation): freeze beta, fit
downstream; then freeze downstream, fit beta (clamped); repeat. W1 frozen=true.
Does alternating break the degeneracy that joint fitting couldn't?"""
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
W1pinv=W1t.t()@torch.linalg.inv(W1t@W1t.t()); g=torch.Generator(device=dev).manual_seed(1)
b_hat=b1t+0.08*(2*torch.rand(k,generator=g,device=dev,dtype=torch.float64)-1)
def mkX(n,tau=2.5):
    r=torch.randn(n,k,generator=g,device=dev,dtype=torch.float64)*tau
    return (r-b_hat)@W1pinv.t()
X=mkX(16384)
with torch.no_grad(): Y=teacher(X)
Xj=mkX(96)
@torch.no_grad()
def tJ(x,fd=1e-4):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()
Jt=torch.stack([tJ(Xj[i]) for i in range(len(Xj))])
stu=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    stu.layers[0].weight.copy_(W1t); stu.layers[0].bias.copy_(b_hat)
stu.layers[0].weight.requires_grad_(False)
beta=stu.layers[0].bias; down=[p for l in stu.layers[1:] for p in l.parameters()]
def stuJ(xb): return tf.vmap(tf.jacrev(lambda x: stu(x.unsqueeze(0))[0]))(xb)
def loss_fn(nb=1024,nj=24):
    idx=torch.randint(0,len(X),(nb,),generator=g,device=dev)
    Lf=((stu(X[idx])-Y[idx])**2).mean()
    ji=torch.randint(0,len(Xj),(nj,),generator=g,device=dev)
    LJ=((stuJ(Xj[ji])-Jt[ji])**2).mean()
    return Lf+50.0*LJ, Lf, LJ
def berr(): return float((stu.layers[0].bias.detach()-b1t).abs().max())

print(f"init bias err {berr():.3e}")
for rnd in range(8):
    # A: fit downstream (beta frozen)
    beta.requires_grad_(False); optD=torch.optim.Adam(down,lr=3e-3)
    for _ in range(1500):
        l,_,_=loss_fn(); optD.zero_grad(); l.backward(); optD.step()
    # B: fit beta only (downstream frozen)
    beta.requires_grad_(True)
    for p in down: p.requires_grad_(False)
    optB=torch.optim.Adam([beta],lr=5e-3)
    for _ in range(400):
        l,_,_=loss_fn(); optB.zero_grad(); l.backward(); optB.step()
        with torch.no_grad(): beta.clamp_(b_hat-0.08,b_hat+0.08)
    for p in down: p.requires_grad_(True)
    _,Lf,LJ=loss_fn()
    print(f"  round {rnd}: Lf {float(Lf):.2e} LJ {float(LJ):.2e}  bias max_err {berr():.3e}")
