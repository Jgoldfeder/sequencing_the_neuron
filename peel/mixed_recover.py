"""End-to-end MIXED-jet bias recovery for a WIDE second layer (2->2->1, needs order 4).
Measure the order-4 mixed jet from the black box. Recover biases by matching the jet
over (biases, downstream) starting from a BIAS GUESS + RANDOM downstream. If biases
land <1e-4 even from random downstream, the jet-matching landscape beats the window-fit
floor that killed the unwarp scan. Compare jet-matching vs the failed approach."""
import torch, numpy as np
from scipy.optimize import least_squares
torch.set_default_dtype(torch.float64)

def net_params(seed):
    g=torch.Generator().manual_seed(seed)
    a=torch.tensor([1.4,1.1]); b=torch.tensor([0.35,-0.20])
    W2=(2*torch.rand(2,2,generator=g)-1)*1.3; c2=(2*torch.rand(2,generator=g)-1)*0.6
    v=(2*torch.rand(1,2,generator=g)-1)*1.3;  e=(2*torch.rand(1,generator=g)-1)*0.5
    return a,b,W2,c2,v,e

def jet(a,b,W2,c2,v,e,t1v,t2v,n=4):
    t1=torch.tensor(t1v,requires_grad=True); t2=torch.tensor(t2v,requires_grad=True)
    s1=torch.sigmoid(a[0]*t1+b[0]); s2=torch.sigmoid(a[1]*t2+b[1])
    u=torch.sigmoid(W2@torch.stack([s1,s2]).reshape(2,1)+c2.reshape(2,1))
    f=(v@u+e.reshape(1,1)).squeeze()
    cur={(0,0):f}
    for order in range(1,n+1):
        nxt={}
        for (i,j),val in cur.items():
            if i+j==order-1:
                d1=torch.autograd.grad(val,t1,create_graph=True,retain_graph=True)[0]
                d2=torch.autograd.grad(val,t2,create_graph=True,retain_graph=True)[0]
                nxt[(i+1,j)]=d1; nxt[(i,j+1)]=d2
        cur.update(nxt)
    return torch.stack([cur[(i,order-i)] for order in range(1,n+1) for i in range(order,-1,-1)])

PROBES=[(0.2,-0.3),(-0.4,0.25),(0.5,0.1)]
def full_jet(a,b,W2,c2,v,e):
    return torch.cat([jet(a,b,W2,c2,v,e,p[0],p[1]) for p in PROBES])

def recover(seed):
    a,b,W2,c2,v,e=net_params(seed)
    Jmeas=full_jet(a,b,W2,c2,v,e).detach().numpy()
    scale=np.abs(Jmeas).max()
    gg=torch.Generator().manual_seed(seed+1000)
    b0=b+ (2*torch.rand(2,generator=gg)-1)*0.06          # bias GUESS (~0.06 off)
    # downstream RANDOM init (no downstream knowledge)
    p0=np.concatenate([b0.numpy(),
        (np.random.default_rng(seed).standard_normal(9))*0.8])
    def resid(p):
        bb=torch.tensor(p[:2]); W=torch.tensor(p[2:6]).reshape(2,2)
        cc=torch.tensor(p[6:8]); vv=torch.tensor(p[8:10]).reshape(1,2); ee=torch.tensor(p[10:11])
        return (full_jet(a,bb,W,cc,vv,ee).detach().numpy()-Jmeas)/scale
    sol=least_squares(resid,p0,method='lm',max_nfev=4000,xtol=1e-15,ftol=1e-15)
    b_err=np.abs(sol.x[:2]-b.numpy()).max()
    return b_err, float(np.abs(b0.numpy()-b.numpy()).max()), sol.cost
res=[recover(s) for s in range(12)]
errs=np.array([r[0] for r in res]); guess=np.array([r[1] for r in res])
print("MIXED-jet bias recovery, 2->2->1 (order 4), from bias-guess + RANDOM downstream:")
print(f"  initial bias-guess error : median {np.median(guess):.2e}")
print(f"  RECOVERED bias error     : median {np.median(errs):.2e}  best {errs.min():.2e}  worst {errs.max():.2e}")
print(f"  fraction reaching <1e-4  : {(errs<1e-4).mean():.0%}   <1e-6: {(errs<1e-6).mean():.0%}")
