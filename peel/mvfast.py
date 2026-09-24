import torch, numpy as np
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
def germ_jet(b1,W2,b2,W3,b3,w4,b4,tps):
    s=sig(tps+b1); p2=s@W2.T+b2; q=sig(p2); qd=q*(1-q); qdd=qd*(1-2*q)
    u=q@W3.T+b3; h=sig(u); hd=h*(1-h); hdd=hd*(1-2*h)
    dq=qd[:,:,None]*W2[None,:,:]
    du=torch.einsum('nr,prk->pnk',W3,dq)
    G1=torch.einsum('on,pn,pnk->pok',w4,hd,du)
    d2u=torch.einsum('nr,pr,ri,rj->pnij',W3,qdd,W2,W2)
    G2=torch.einsum('on,pn,pni,pnj->poij',w4,hdd,du,du)+torch.einsum('on,pn,pnij->poij',w4,hd,d2u)
    return G1,G2
def make(dims,seed):
    g=torch.Generator().manual_seed(seed); k,W,m,O=dims
    return ((2*torch.rand(k,generator=g)-1)*0.5, torch.randn(W,k,generator=g)/np.sqrt(k),(2*torch.rand(W,generator=g)-1)*0.5,
            torch.randn(m,W,generator=g)/np.sqrt(W),(2*torch.rand(m,generator=g)-1)*0.5,
            torch.randn(O,m,generator=g)/np.sqrt(m),(2*torch.rand(O,generator=g)-1)*0.5)
def jetvec(pars,tps,k):
    G1,G2=germ_jet(*pars,tps); iu=torch.triu_indices(k,k)
    return torch.cat([G1.reshape(-1), G2[:,:,iu[0],iu[1]].reshape(-1)])
def test(dims,seeds,nprobe=12,nrestart=6):
    k,W,m,O=dims
    for seed in seeds:
        tru=make(dims,seed); b1=tru[0]
        gp=torch.Generator().manual_seed(seed+9); tps=(2*torch.rand(nprobe,k,generator=gp)-1)*1.2
        with torch.no_grad(): Jm=jetvec(tru,tps,k); sc=Jm.abs().max()
        nds=sum(x.numel() for x in tru[1:])
        rng=np.random.default_rng(seed); bg=b1+0.05*torch.tensor(2*rng.random(k)-1)
        best=(9.,9.)
        for rs in range(nrestart):
            gg=torch.Generator().manual_seed(seed*50+rs+1)
            bb=bg.clone().requires_grad_(True)
            P=[bb]+[(torch.randn(*x.shape,generator=gg)*0.5).clone().requires_grad_(True) for x in tru[1:]]
            def loss(): return ((jetvec(P,tps,k)-Jm)/sc).pow(2).sum()
            opt=torch.optim.LBFGS(P,lr=1.0,max_iter=250,history_size=60,line_search_fn='strong_wolfe',tolerance_grad=1e-20,tolerance_change=1e-22)
            def cl(): opt.zero_grad(); L=loss(); L.backward(); return L
            for _ in range(3): opt.step(cl)
            with torch.no_grad(): c=float(loss())
            if c<best[0]: best=(c,float((bb.detach()-b1).abs().max()))
        print(f"  {dims}: downstream={nds}p  bias err={best[1]:.2e}  (resid {best[0]:.1e})",flush=True)
print("MULTIVARIATE closed-form-jet full-net fit (biases guess, downstream random):",flush=True)
test([3,4,3,2],[0,1,2]); test([6,8,4,4],[0,1,2]); test([8,12,8,4],[0,1])
