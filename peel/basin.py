import torch, numpy as np
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
    return ((2*torch.rand(k,generator=g)-1)*0.5, torch.randn(W,k,generator=g)/np.sqrt(k),(2*torch.rand(W,generator=g)-1)*0.5,
            torch.randn(m,W,generator=g)/np.sqrt(W),(2*torch.rand(m,generator=g)-1)*0.5,
            torch.randn(O,m,generator=g)/np.sqrt(m),(2*torch.rand(O,generator=g)-1)*0.5)
def raw_jet(pars,tps,k):
    b1=pars[0]; s=sig(tps+b1); d=s*(1-s); sdd=d*(1-2*s)
    G1,G2=germ_jet(*pars,tps); Fi=d[:,None,:]*G1; dd=d[:,:,None]*d[:,None,:]
    Fij=dd[:,None,:,:]*G2 + torch.diag_embed(sdd[:,None,:]*G1); iu=torch.triu_indices(k,k)
    return torch.cat([Fi.reshape(-1), Fij[:,:,iu[0],iu[1]].reshape(-1)])
def fit(bg,down0,Fm,tps,k,sc,joint):
    bb=bg.clone().requires_grad_(True)
    if joint: down=[d.clone().requires_grad_(True) for d in down0]; P=[bb]+down
    else: down=[d.clone() for d in down0]; P=[bb]
    def loss(): return ((raw_jet([bb]+down,tps,k)-Fm)/sc).pow(2).sum()
    opt=torch.optim.LBFGS(P,lr=1.0,max_iter=250,history_size=60,line_search_fn='strong_wolfe',tolerance_grad=1e-22,tolerance_change=1e-24)
    def cl(): opt.zero_grad(); L=loss(); L.backward(); return L
    for _ in range(3): opt.step(cl)
    return float((bb.detach()-b1true).abs().max())
def basin(dims,seed,alphas,ndir=10,nprobe=12):
    global b1true
    k=dims[0]; tru=make(dims,seed); b1true=tru[0]; down_true=list(tru[1:])
    gp=torch.Generator().manual_seed(seed+9); tps=(2*torch.rand(nprobe,k,generator=gp)-1)*1.2
    with torch.no_grad(): Fm=raw_jet(tru,tps,k); sc=Fm.abs().max()
    rng=np.random.default_rng(seed); bg=b1true+0.05*torch.tensor(2*rng.random(k)-1)
    print(f" {dims} seed{seed}: alpha  P_b(frozen)  P_b(joint)",flush=True)
    for al in alphas:
        cf=0; cj=0
        for dd in range(ndir):
            g2=torch.Generator().manual_seed(seed*300+int(al*1e6)+dd+1)
            down0=[W+al*(W.norm()/ (X:=torch.randn(*W.shape,generator=g2)).norm())*X for W in down_true]
            if fit(bg,down0,Fm,tps,k,sc,False)<1e-4: cf+=1
            if fit(bg,down0,Fm,tps,k,sc,True)<1e-4: cj+=1
        print(f"   a={al:.0e}: {cf/ndir:.2f}       {cj/ndir:.2f}",flush=True)
print("BASIN-SIZE: P(bias err<1e-4) vs relative downstream perturbation alpha (b0=truth+0.05)",flush=True)
for sd in range(2): basin([3,4,3,2],sd,[1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.0])
basin([6,8,4,4],0,[1e-3,1e-2,3e-2,1e-1,3e-1])
