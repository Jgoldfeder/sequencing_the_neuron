import torch, numpy as np
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
exec(open('mvfast.py').read().split('def jetvec')[0])   # germ_jet, make, sig
def raw_jet(pars,tps,k):
    b1=pars[0]; s=sig(tps+b1); d=s*(1-s); sdd=d*(1-2*s)
    G1,G2=germ_jet(*pars,tps)                       # (P,O,k),(P,O,k,k) wrt s
    Fi=d[:,None,:]*G1
    dd=d[:,:,None]*d[:,None,:]
    Fij=dd[:,None,:,:]*G2 + torch.diag_embed(sdd[:,None,:]*G1)
    iu=torch.triu_indices(k,k)
    return torch.cat([Fi.reshape(-1), Fij[:,:,iu[0],iu[1]].reshape(-1)])
def fitprob(dims,seed,mode,nprobe=12,nrestart=8):
    k=dims[0]; tru=make(dims,seed); b1=tru[0]
    gp=torch.Generator().manual_seed(seed+9); tps=(2*torch.rand(nprobe,k,generator=gp)-1)*1.2
    with torch.no_grad(): Fm=raw_jet(tru,tps,k); sc=Fm.abs().max()
    rng=np.random.default_rng(seed); bg=b1+0.05*torch.tensor(2*rng.random(k)-1)
    best=(9.,9.)
    for rs in range(nrestart):
        gg=torch.Generator().manual_seed(seed*77+rs+1)
        if mode==1:
            down=[(torch.randn(*x.shape,generator=gg)*0.6).clone().requires_grad_(True) for x in tru[1:]]; getb=lambda: b1; P=down
        elif mode==2:
            bb=bg.clone().requires_grad_(True); down=[x.clone().requires_grad_(True) for x in tru[1:]]; getb=lambda: bb; P=[bb]+down
        else:
            bb=bg.clone().requires_grad_(True); down=[(torch.randn(*x.shape,generator=gg)*0.6).clone().requires_grad_(True) for x in tru[1:]]; getb=lambda: bb; P=[bb]+down
        def loss(): return ((raw_jet([getb()]+down,tps,k)-Fm)/sc).pow(2).sum()
        opt=torch.optim.LBFGS(P,lr=1.0,max_iter=300,history_size=70,line_search_fn='strong_wolfe',tolerance_grad=1e-22,tolerance_change=1e-24)
        def cl(): opt.zero_grad(); L=loss(); L.backward(); return L
        for _ in range(3): opt.step(cl)
        with torch.no_grad(): c=float(loss())
        be=float((getb().detach()-b1).abs().max()) if mode>1 else 0.0
        if c<best[0]: best=(c,be)
    return best
print("RAW-JET fit (non-oracle) 3 controls: [1]true-b/rand-down [2]guess-b/true-down [3]guess-b/rand-down",flush=True)
for dims in ([3,4,3,2],[6,8,4,4]):
    for sd in range(3):
        r1=fitprob(dims,sd,1); r2=fitprob(dims,sd,2); r3=fitprob(dims,sd,3)
        print(f"  {dims} s{sd}: [1] resid={r1[0]:.1e} | [2] resid={r2[0]:.1e} biaserr={r2[1]:.1e} | [3] resid={r3[0]:.1e} biaserr={r3[1]:.1e}",flush=True)
