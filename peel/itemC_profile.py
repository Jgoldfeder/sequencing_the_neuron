"""ITEM C step 5: profile loss L(beta)=min_downstream ||jet_model - jet_meas||^2 over
the CLEAN identifying subset (H_off + T_distinct). Two questions:
 (1) at TRUE beta, does the downstream fit from RANDOM init reach ~0? (if it floors,
     fitting is out -> elimination must be algebraic/spectral)
 (2) does best-restart L(beta) have its minimum at true beta on a 1-D slice?"""
import torch, itertools, numpy as np
from torch.func import jacfwd
torch.set_default_dtype(torch.float64)
k,W,m=6,8,4; P=10
g=torch.Generator().manual_seed(3)
W2=torch.randn(W,k,generator=g)*0.9; b2=(2*torch.rand(W,generator=g)-1)*0.6
W3=torch.randn(m,W,generator=g)*0.7; b3=(2*torch.rand(m,generator=g)-1)*0.5
w4=torch.randn(m,generator=g)*0.8;  b4=(2*torch.rand(1,generator=g)-1)*0.3
beta_true=(2*torch.rand(k,generator=g)-1)*0.5
gp=torch.Generator().manual_seed(11); cs=[(2*torch.rand(k,generator=gp)-1)*1.3 for _ in range(P)]
hoff=[(i,j) for i in range(k) for j in range(i+1,k)]
tdist=[(i,j,l) for i in range(k) for j in range(i+1,k) for l in range(j+1,k)]
def jet(beta, ds, c):                     # ds=(W2,b2,W3,b3,w4,b4)
    W2_,b2_,W3_,b3_,w4_,b4_=ds
    def Ft(t):
        s=torch.sigmoid(c+beta+t); q=torch.sigmoid(W2_@s+b2_); h=torch.sigmoid(W3_@q+b3_)
        return (w4_@h+b4_).squeeze()
    G2=jacfwd(jacfwd(Ft))(torch.zeros(k)); G3=jacfwd(jacfwd(jacfwd(Ft)))(torch.zeros(k))
    return torch.cat([torch.stack([G2[i,j] for (i,j) in hoff]),
                      torch.stack([G3[i,j,l] for (i,j,l) in tdist])])
ds_true=(W2,b2,W3,b3,w4,b4)
with torch.no_grad(): Jm=[jet(beta_true,ds_true,c) for c in cs]
scale=max(float(v.abs().max()) for v in Jm)
def fit(beta, seed, n_adam=1500, n_lbfgs=40):
    gg=torch.Generator().manual_seed(seed)
    ds=[torch.randn(W,k,generator=gg)*0.6, torch.randn(W,generator=gg)*0.5,
        torch.randn(m,W,generator=gg)*0.5, torch.randn(m,generator=gg)*0.5,
        torch.randn(m,generator=gg)*0.5, torch.randn(1,generator=gg)*0.3]
    ds=[p.requires_grad_(True) for p in ds]
    def loss():
        return sum(((jet(beta,ds,cs[p])-Jm[p])/scale).pow(2).sum() for p in range(P))
    opt=torch.optim.Adam(ds,lr=3e-3)
    for _ in range(n_adam): opt.zero_grad(); L=loss(); L.backward(); opt.step()
    o2=torch.optim.LBFGS(ds,lr=1.0,max_iter=100,history_size=50,line_search_fn='strong_wolfe')
    for _ in range(n_lbfgs): o2.step(lambda: (o2.zero_grad(), (l:=loss()), l.backward(), l)[1] if True else l)
    with torch.no_grad(): return float(loss())
def Lprofile(beta, restarts=10):
    return min(fit(beta, s) for s in range(restarts))
print("(1) fit-floor test at TRUE beta (downstream from RANDOM init, 10 restarts):")
L0=Lprofile(beta_true)
print(f"    best L(true beta) = {L0:.2e}   -> {'reaches ~0 (fitting viable)' if L0<1e-8 else 'FLOORS (fitting out; need algebraic elimination)'}")
print("(2) L(beta) on 1-D slice beta_1 = true + delta:")
for d in (-0.10,-0.05,0.0,0.05,0.10):
    b=beta_true.clone(); b[0]+=d
    print(f"    delta={d:+.2f}: best L = {Lprofile(b):.2e}")
