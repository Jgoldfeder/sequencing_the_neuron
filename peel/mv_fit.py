import torch, numpy as np
from torch.func import jacfwd
torch.set_default_dtype(torch.float64)
sig=torch.sigmoid
def make(dims,seed):
    g=torch.Generator().manual_seed(seed); k=dims[0]
    b1=(2*torch.rand(k,generator=g)-1)*0.5
    Ws=[torch.randn(dims[i+1],dims[i],generator=g)/np.sqrt(dims[i]) for i in range(len(dims)-1)]
    bs=[(2*torch.rand(dims[i+1],generator=g)-1)*0.5 for i in range(len(dims)-1)]
    return b1,Ws,bs,k
def fwd(b1,Ws,bs,t):
    h=sig(t+b1)
    for i in range(len(Ws)):
        h=Ws[i]@h+bs[i]
        if i<len(Ws)-1: h=sig(h)
    return h
def jet(b1,Ws,bs,tp,k):
    def F(t): return fwd(b1,Ws,bs,tp+t)
    G=jacfwd(F)(torch.zeros(k)); Hs=jacfwd(jacfwd(F))(torch.zeros(k))
    iu=torch.triu_indices(k,k)
    return torch.cat([G.reshape(-1), Hs[:,iu[0],iu[1]].reshape(-1)])
def flat(Ws,bs): return torch.cat([W.reshape(-1) for W in Ws]+[b for b in bs])
def unflat(v,shapes):
    Ws=[];bs=[];idx=0
    for sh in shapes:
        n=sh[0]*sh[1]; Ws.append(v[idx:idx+n].reshape(sh)); idx+=n
    for sh in shapes:
        bs.append(v[idx:idx+sh[0]]); idx+=sh[0]
    return Ws,bs
def test(dims,seeds,nprobe=16,nrestart=8):
    k=dims[0]
    for seed in seeds:
        b1,Ws,bs,_=make(dims,seed); shapes=[W.shape for W in Ws]; nds=flat(Ws,bs).numel()
        gp=torch.Generator().manual_seed(seed+9); probes=[(2*torch.rand(k,generator=gp)-1)*1.2 for _ in range(nprobe)]
        with torch.no_grad(): Jm=torch.stack([jet(b1,Ws,bs,pb,k) for pb in probes]); scale=Jm.abs().max()
        rng=np.random.default_rng(seed); bg=b1+0.05*torch.tensor(2*rng.random(k)-1)
        best=(9.,9.)
        for rs in range(nrestart):
            gg=torch.Generator().manual_seed(seed*50+rs+1)
            bb=bg.clone().requires_grad_(True); dd=(torch.randn(nds,generator=gg)*0.5).requires_grad_(True)
            def loss():
                Wsx,bsx=unflat(dd,shapes)
                J=torch.stack([jet(bb,Wsx,bsx,pb,k) for pb in probes]); return ((J-Jm)/scale).pow(2).sum()
            opt=torch.optim.LBFGS([bb,dd],lr=1.0,max_iter=300,history_size=60,line_search_fn='strong_wolfe',tolerance_grad=1e-20,tolerance_change=1e-22)
            def cl(): opt.zero_grad(); L=loss(); L.backward(); return L
            for _ in range(4): opt.step(cl)
            with torch.no_grad(): c=float(loss())
            if c<best[0]: best=(c, float((bb.detach()-b1).abs().max()))
        print(f"  {dims} seed{seed}: downstream={nds}p bias err={best[1]:.2e} (resid {best[0]:.1e})")
print("MULTIVARIATE full-network fit (biases guess, downstream random) scaling:")
test([3,4,3,2],[0,1,2]); test([6,8,4,2],[0,1,2])
