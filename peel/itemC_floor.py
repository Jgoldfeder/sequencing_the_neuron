"""Deep fit-floor test (vectorized): does fitting the downstream to H_off reach ~0 at
true beta from random init? If it floors, deep elimination must be algebraic."""
import torch
torch.set_default_dtype(torch.float64)
k,W,m=6,8,4; P=8
g=torch.Generator().manual_seed(3)
W2=torch.randn(W,k,generator=g)*0.9; b2=(2*torch.rand(W,generator=g)-1)*0.6
W3=torch.randn(m,W,generator=g)*0.7; b3=(2*torch.rand(m,generator=g)-1)*0.5
w4=torch.randn(m,generator=g)*0.8;  b4=(2*torch.rand(1,generator=g)-1)*0.3
beta_true=(2*torch.rand(k,generator=g)-1)*0.5
gp=torch.Generator().manual_seed(11); cs=torch.stack([(2*torch.rand(k,generator=gp)-1)*1.3 for _ in range(P)])
iu=torch.triu_indices(k,k,offset=1)
def germH(ds, sact):                            # sact:(P,k) -> G:(P,k,k), vectorized
    W2_,b2_,W3_,b3_,w4_,b4_=ds
    p=sact@W2_.t()+b2_; sp=torch.sigmoid(p); spd=sp*(1-sp); spdd=spd*(1-2*sp); q=sp
    u=q@W3_.t()+b3_; hh=torch.sigmoid(u); hd=hh*(1-hh); hdd=hd*(1-2*hh)
    rho=(w4_*hd)@W3_
    K=torch.einsum('pn,nr,ns->prs', w4_*hdd, W3_, W3_)
    S=torch.diag_embed(rho*spdd)+spd.unsqueeze(2)*K*spd.unsqueeze(1)
    return torch.einsum('ri,prs,sj->pij', W2_, S, W2_)
def offd(G): return G[:,iu[0],iu[1]]
sact=torch.sigmoid(cs+beta_true)
with torch.no_grad(): Jm=offd(germH((W2,b2,W3,b3,w4,b4),sact)); scale=float(Jm.abs().max())
def fit(seed):
    gg=torch.Generator().manual_seed(seed)
    ds=[(torch.randn(W,k,generator=gg)*0.6).requires_grad_(True),(torch.randn(W,generator=gg)*0.5).requires_grad_(True),
        (torch.randn(m,W,generator=gg)*0.5).requires_grad_(True),(torch.randn(m,generator=gg)*0.5).requires_grad_(True),
        (torch.randn(m,generator=gg)*0.5).requires_grad_(True),(torch.randn(1,generator=gg)*0.3).requires_grad_(True)]
    def loss(): return ((offd(germH(ds,sact))-Jm)/scale).pow(2).sum()
    o=torch.optim.LBFGS(ds,lr=1.0,max_iter=400,history_size=80,line_search_fn='strong_wolfe',tolerance_grad=1e-18,tolerance_change=1e-20)
    def cl():
        o.zero_grad(); L=loss(); L.backward(); return L
    for _ in range(3): o.step(cl)
    with torch.no_grad(): return float(loss())
Ls=sorted(fit(s) for s in range(8))
print(f'deep fit to H_off at TRUE beta, 15 restarts: best={Ls[0]:.2e}')
print(f'  sorted top: {[f"{x:.1e}" for x in Ls[:6]]}')
print(f'  VERDICT: {"reaches ~0 (fitting viable)" if Ls[0]<1e-8 else "FLOORS -> deep elimination must be algebraic, not fitting"}')
