import torch, numpy as np
from torch.func import jacrev
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
def germ_jet(b1,W2,b2,W3,b3,w4,b4,tps):
    s=sig(tps+b1); p2=s@W2.T+b2; q=sig(p2); qd=q*(1-q); qdd=qd*(1-2*q)
    u=q@W3.T+b3; h=sig(u); hd=h*(1-h); hdd=hd*(1-2*h)
    dq=qd[:,:,None]*W2[None,:,:]; du=torch.einsum('nr,prk->pnk',W3,dq)
    G1=torch.einsum('on,pn,pnk->pok',w4,hd,du)
    d2u=torch.einsum('nr,pr,ri,rj->pnij',W3,qdd,W2,W2)
    G2=torch.einsum('on,pn,pni,pnj->poij',w4,hdd,du,du)+torch.einsum('on,pn,pnij->poij',w4,hd,d2u)
    return G1,G2
def make(dims,seed):
    g=torch.Generator().manual_seed(seed); k,W,m,O=dims
    return [(2*torch.rand(k,generator=g)-1)*0.5, torch.randn(W,k,generator=g)/np.sqrt(k),(2*torch.rand(W,generator=g)-1)*0.5,
            torch.randn(m,W,generator=g)/np.sqrt(W),(2*torch.rand(m,generator=g)-1)*0.5,
            torch.randn(O,m,generator=g)/np.sqrt(m),(2*torch.rand(O,generator=g)-1)*0.5]
def unpack(th,dims):
    k,W,m,O=dims; i=0
    b1=th[i:i+k];i+=k; W2=th[i:i+W*k].reshape(W,k);i+=W*k; b2=th[i:i+W];i+=W
    W3=th[i:i+m*W].reshape(m,W);i+=m*W; b3=th[i:i+m];i+=m; w4=th[i:i+O*m].reshape(O,m);i+=O*m; b4=th[i:i+O]
    return (b1,W2,b2,W3,b3,w4,b4)
def flat(pars): return torch.cat([pars[0]]+[p.reshape(-1) for p in pars[1:]])
def raw_jet(pars,tps,k):
    b1=pars[0]; s=sig(tps+b1); d=s*(1-s); sdd=d*(1-2*s)
    G1,G2=germ_jet(*pars,tps); Fi=d[:,None,:]*G1; dd=d[:,:,None]*d[:,None,:]
    Fij=dd[:,None,:,:]*G2 + torch.diag_embed(sdd[:,None,:]*G1); iu=torch.triu_indices(k,k)
    return torch.cat([Fi.reshape(-1), Fij[:,:,iu[0],iu[1]].reshape(-1)])
def resid(th,dims,tps,Fm,sc): return (raw_jet(unpack(th,dims),tps,dims[0])-Fm)/sc
def run_lbfgs(th0,dims,tps,Fm,sc):
    th=th0.clone().requires_grad_(True)
    opt=torch.optim.LBFGS([th],lr=1.0,max_iter=500,history_size=100,line_search_fn='strong_wolfe',tolerance_grad=1e-24,tolerance_change=1e-26)
    def cl(): opt.zero_grad(); L=(resid(th,dims,tps,Fm,sc)**2).sum(); L.backward(); return L
    for _ in range(5): opt.step(cl)
    return th.detach()
def run_lm(th0,dims,tps,Fm,sc,iters=80):
    th=th0.clone(); lam=1e-4; I=torch.eye(th.numel())
    r=resid(th,dims,tps,Fm,sc); c=float(r@r)
    for it in range(iters):
        J=jacrev(lambda t: resid(t,dims,tps,Fm,sc))(th); g=J.T@r; Hm=J.T@J
        ok=False
        for _ in range(40):
            try: step=torch.linalg.solve(Hm+lam*I,-g)
            except Exception: lam*=3; continue
            thn=th+step; rn=resid(thn,dims,tps,Fm,sc); cn=float(rn@rn)
            if cn<c: th=thn;r=rn;c=cn;lam=max(lam*0.4,1e-14);ok=True;break
            lam*=3
            if lam>1e13: break
        if not ok or c<1e-28: break
    return th
print("LBFGS vs LM from control[2] (b=truth+0.05, downstream=truth):",flush=True)
for dims,seeds in ([3,4,3,2],[0,1,2]),([6,8,4,4],[0,1]):
    for sd in seeds:
        tru=make(dims,sd); b1=tru[0]; k=dims[0]
        gp=torch.Generator().manual_seed(sd+9); tps=(2*torch.rand(12,k,generator=gp)-1)*1.2
        with torch.no_grad(): Fm=raw_jet(tru,tps,k); sc=Fm.abs().max()
        rng=np.random.default_rng(sd); th0=flat(tru).clone(); th0[:k]=b1+0.05*torch.tensor(2*rng.random(k)-1)
        thL=run_lbfgs(th0,dims,tps,Fm,sc); thM=run_lm(th0,dims,tps,Fm,sc)
        eL=float((thL[:k]-b1).abs().max()); eM=float((thM[:k]-b1).abs().max())
        rL=float((resid(thL,dims,tps,Fm,sc)**2).sum()); rM=float((resid(thM,dims,tps,Fm,sc)**2).sum())
        print(f"  {dims} s{sd}: LBFGS biaserr={eL:.2e}(r={rL:.1e})  LM biaserr={eM:.2e}(r={rM:.1e})",flush=True)
