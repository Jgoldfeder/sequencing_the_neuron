import torch, numpy as np
from torch.func import jacfwd
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
exec(open('realgate2.py').read().split('print("JOINT')[0])   # germ_deriv, raw_jet_S, make, flat, unpack
def lm(th0,rf,iters=80):
    th=th0.clone();lam=1e-4;n=th.numel();I=torch.eye(n);r=rf(th);c=float(r@r)
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
def preact_err(down0,dtrue,Sstar,b1,tps):
    z1=tps@Sstar.T+b1; s=sig(z1)
    z2t=s@dtrue[0].T+dtrue[1]; qt=sig(z2t); z3t=qt@dtrue[2].T+dtrue[3]
    z2i=s@down0[0].T+down0[1]; qi=sig(z2i); z3i=qi@down0[2].T+down0[3]
    ez2=float((z2i-z2t).pow(2).mean().sqrt()/z2t.pow(2).mean().sqrt())
    ez3=float((z3i-z3t).pow(2).mean().sqrt()/z3t.pow(2).mean().sqrt())
    return ez2,ez3
dims=[6,8,4,4];k=6; pts=[]
for sd in [0,1,2]:
    Sstar,b1,down=make(dims,sd);shapes=[tuple(d.shape) for d in down]
    gp=torch.Generator().manual_seed(sd+9);tps=(2*torch.rand(12,k,generator=gp)-1)*1.2
    with torch.no_grad():Fm=raw_jet_S(Sstar,b1,down,tps,k);sc=Fm.abs().max()
    rng=np.random.default_rng(sd);b0=b1+0.05*torch.tensor(2*rng.random(k)-1)
    for al in [0.02,0.05,0.1,0.15,0.2,0.3]:
        for dd in range(6):
            g2=torch.Generator().manual_seed(sd*900+int(al*1e5)+dd+1)
            down0=[W+al*(W.norm()/(X:=torch.randn(*W.shape,generator=g2)).norm())*X for W in down]
            ez2,ez3=preact_err(down0,down,Sstar,b1,tps)
            th0=flat(torch.eye(k),b0,down0)
            rf=lambda th:(raw_jet_S(*unpack(th,dims,shapes),tps,k)-Fm)/sc
            thM,c=lm(th0,rf); _,bh,_=unpack(thM,dims,shapes)
            pts.append((ez2,ez3,int(float((bh-b1).abs().max())<1e-4)))
pts=np.array(pts)
print("Preactivation calibration: LM success vs functional preact error (108 inits)",flush=True)
for lo,hi in [(0,0.05),(0.05,0.10),(0.10,0.15),(0.15,0.25),(0.25,0.5),(0.5,10)]:
    m=(pts[:,1]>=lo)&(pts[:,1]<hi)
    if m.sum(): print(f"  ez3 in [{lo:.2f},{hi:.2f}): n={m.sum():2d}  success={pts[m,2].mean():.2f}  ez2 range=[{pts[m,0].min():.2f},{pts[m,0].max():.2f}]",flush=True)
succ=pts[pts[:,2]==1]; fail=pts[pts[:,2]==0]
print(f"  SUCCESS: max ez2={succ[:,0].max():.2f} max ez3={succ[:,1].max():.2f}; 95pct ez2={np.percentile(succ[:,0],95):.2f} ez3={np.percentile(succ[:,1],95):.2f}",flush=True)
if len(fail): print(f"  FAIL: min ez2={fail[:,0].min():.2f} min ez3={fail[:,1].min():.2f}",flush=True)
