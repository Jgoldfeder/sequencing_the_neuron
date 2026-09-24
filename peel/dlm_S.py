import torch, numpy as np
from torch.func import jacfwd
torch.set_default_dtype(torch.float64)
exec(open('realgate2.py').read().split('print("JOINT')[0])   # germ_deriv, raw_jet_S, make, flat, unpack
def lm(th0,rf,iters=80):
    th=th0.clone();lam=1e-4;n=th.numel();I=torch.eye(n); r=rf(th);c=float(r@r)
    for it in range(iters):
        J=jacfwd(rf)(th);g=J.T@r;H=J.T@J;ok=False
        for _ in range(30):
            try:d=torch.linalg.solve(H+lam*I,-g)
            except Exception:lam*=3;continue
            thn=th+d;rn=rf(thn);cn=float(rn@rn)
            if cn<c:th=thn;r=rn;c=cn;lam=max(lam*0.4,1e-14);ok=True;break
            lam*=3
            if lam>1e13:break
        if not ok or c<1e-28:break
    return th,c
dims=[6,8,4,4];k=6
print("DIRECT-LM with (S,beta,theta), S0=I, toy 6->8->4->4:",flush=True)
for sd in [0,1]:
    Sstar,b1,down=make(dims,sd);shapes=[tuple(d.shape) for d in down]
    gp=torch.Generator().manual_seed(sd+9);tps=(2*torch.rand(12,k,generator=gp)-1)*1.2
    with torch.no_grad():Fm=raw_jet_S(Sstar,b1,down,tps,k);sc=Fm.abs().max()
    rng=np.random.default_rng(sd);b0=b1+0.05*torch.tensor(2*rng.random(k)-1)
    for al in [0.05,0.10,0.20]:
        eb=[];es=[]
        for dd in range(5):
            g2=torch.Generator().manual_seed(sd*500+int(al*1e5)+dd+1)
            down0=[W+al*(W.norm()/(X:=torch.randn(*W.shape,generator=g2)).norm())*X for W in down]
            th0=flat(torch.eye(k),b0,down0)
            rf=lambda th:(raw_jet_S(*unpack(th,dims,shapes),tps,k)-Fm)/sc
            thM,c=lm(th0,rf);Sh,bh,_=unpack(thM,dims,shapes)
            eb.append(float((bh-b1).abs().max()));es.append(float((Sh-Sstar).abs().max()))
        eb=np.array(eb)
        print(f"  s{sd} a={al}: P(bias<1e-4)={(eb<1e-4).mean():.2f} med-bias={np.median(eb):.1e} max||S-S*||={max(es):.1e}",flush=True)
