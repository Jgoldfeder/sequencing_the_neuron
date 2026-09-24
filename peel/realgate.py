import torch, numpy as np
from torch.func import jvp, vjp
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
exec(open('lm_test.py').read().split('def resid')[0])   # germ_jet, make, unpack, flat, raw_jet
def resid(th,dims,tps,Fm,sc): return (raw_jet(unpack(th,dims),tps,dims[0])-Fm)/sc
def cg(matvec,b,iters=200,tol=1e-16):
    x=torch.zeros_like(b); r=b.clone(); p=r.clone(); rs=r@r
    for _ in range(iters):
        Ap=matvec(p); a=rs/(p@Ap+1e-300); x=x+a*p; r=r-a*Ap; rs2=r@r
        if rs2.sqrt()<tol: break
        p=r+(rs2/rs)*p; rs=rs2
    return x
def gncg(th0,rf,iters=30):
    th=th0.clone(); lam=1e-6; r=rf(th); c=float(r@r)
    for it in range(iters):
        rr,vjpf=vjp(rf,th); g=vjpf(rr)[0]
        ok=False
        for _ in range(15):
            JTJ=lambda v: vjpf(jvp(rf,(th,),(v,))[1])[0]+lam*v
            delta=cg(JTJ,-g); thn=th+delta; rn=rf(thn); cn=float(rn@rn)
            if cn<c: th=thn;c=cn;lam=max(lam*0.5,1e-13);ok=True;break
            lam*=4
            if lam>1e11: break
        if not ok or c<1e-26: break
    return th,c
def trial(dims,seed,alpha,nprobe):
    k=dims[0]; tru=make(dims,seed); b1=tru[0]
    gp=torch.Generator().manual_seed(seed+9); tps=(2*torch.rand(nprobe,k,generator=gp)-1)*1.2
    with torch.no_grad(): Fm=raw_jet(tru,tps,k); sc=Fm.abs().max()
    rng=np.random.default_rng(seed); th0=flat(tru).clone(); th0[:k]=b1+0.05*torch.tensor(2*rng.random(k)-1)
    if alpha>0:
        g2=torch.Generator().manual_seed(seed*13+1); idx=k
        newdown=[]
        for W in tru[1:]:
            X=torch.randn(*W.shape,generator=g2); Wp=W+alpha*(W.norm()/X.norm())*X; newdown.append(Wp)
        th0=flat([th0[:k]]+newdown)
    rf=lambda th: resid(th,dims,tps,Fm,sc)
    thM,c=gncg(th0,rf)
    return float((thM[:k]-b1).abs().max()), c
print("VALIDATE GN-CG-LM on 6->8->4->4 (alpha=0, should be machine):",flush=True)
for sd in range(2):
    e,c=trial([6,8,4,4],sd,0.0,12); print(f"  seed{sd}: bias err={e:.2e} resid={c:.1e}",flush=True)
print("REAL 24->32->16->8 GN-CG-LM, oracle warm-start (b0=truth+0.05, downstream truth+alpha):",flush=True)
for al in (0.0,0.01,0.05,0.10):
    e,c=trial([24,32,16,8],0,al,8); print(f"  alpha={al:.2f}: bias err={e:.2e} resid={c:.1e}",flush=True)
