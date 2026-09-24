import torch, numpy as np
from torch.func import jacfwd
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
exec(open('realgate2.py').read().split('print("JOINT')[0])
try: from scipy.optimize import linear_sum_assignment; HAVE=True
except Exception: HAVE=False
def lm(th0,rf,iters=150):
    th=th0.clone();lam=1e-3;n=th.numel();I=torch.eye(n);r=rf(th);c=float(r@r)
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
def preacts(s,down):
    W2,b2,W3,b3,w4,b4=down; z2=s@W2.T+b2; q=sig(z2); z3=q@W3.T+b3; return z2,z3
def match(As,At):
    d=As.shape[1]; C=np.zeros((d,d)); Sg=np.zeros((d,d))
    for i in range(d):
        a=As[:,i]-As[:,i].mean()
        for j in range(d):
            b=At[:,j]-At[:,j].mean(); cc=float((a@b)/(a.norm()*b.norm()+1e-12))
            Sg[i,j]=1.0 if cc>=0 else -1.0; C[i,j]=-abs(cc)
    ci=linear_sum_assignment(C)[1] if HAVE else list(range(d))
    return ci,torch.tensor([Sg[i,ci[i]] for i in range(d)])
def align_eps(stu,down,s):
    z2t,z3t=preacts(s,down); z2s,z3s=preacts(s,stu)
    p2,g2=match(z2s,z2t); az2=z2s[:,p2]*g2; p3,g3=match(z3s,z3t); az3=z3s[:,p3]*g3
    return (float((az2-z2t).pow(2).mean().sqrt()/z2t.pow(2).mean().sqrt()),
            float((az3-z3t).pow(2).mean().sqrt()/z3t.pow(2).mean().sqrt()))
def fdown(down): return torch.cat([d.reshape(-1) for d in down])
def udown(v,shapes):
    out=[];i=0
    for sh in shapes:
        nn=int(np.prod(sh)); out.append(v[i:i+nn].reshape(sh)); i+=nn
    return out
dims=[6,8,4,4]; k=6
print("Order-2 JET-matching student fit (S=I,beta=guess frozen; random-init, restarts) -> eps_z & joint LM",flush=True)
for sd in [0,1,2]:
    Sstar,b1,down=make(dims,sd); shapes=[tuple(d.shape) for d in down]
    berr=0.01
    rng=np.random.default_rng(sd*10+1); betahat=b1+berr*torch.tensor(2*rng.random(k)-1)
    gf=torch.Generator().manual_seed(sd+21); tps=(2*torch.rand(40,k,generator=gf)-1)*1.3
    with torch.no_grad(): Fm=raw_jet_S(Sstar,b1,down,tps,k); sc=Fm.abs().max()
    rf=lambda dv:(raw_jet_S(torch.eye(k),betahat,udown(dv,shapes),tps,k)-Fm)/sc
    best=None;bestc=1e18
    for r in range(12):
        g=torch.Generator().manual_seed(sd*50+r*3+1)
        d0=[torch.randn(sh,generator=g)/np.sqrt(sh[1]) if len(sh)==2 else (2*torch.rand(sh,generator=g)-1)*0.4 for sh in shapes]
        dM,c=lm(fdown(d0),rf,iters=120)
        if c<bestc: bestc=c; best=dM.clone()
    stu=udown(best,shapes)
    gp=torch.Generator().manual_seed(sd+99); te=(2*torch.rand(400,k,generator=gp)-1)*1.2
    se=sig(te@Sstar.T+b1); e2,e3=align_eps(stu,down,se)
    # joint LM from (I, betahat, jet-fitted student)
    gp2=torch.Generator().manual_seed(sd+3); tps2=(2*torch.rand(12,k,generator=gp2)-1)*1.2
    with torch.no_grad(): Fm2=raw_jet_S(Sstar,b1,down,tps2,k); sc2=Fm2.abs().max()
    rf2=lambda th:(raw_jet_S(*unpack(th,dims,shapes),tps2,k)-Fm2)/sc2
    thM,cj=lm(flat(torch.eye(k),betahat,stu),rf2); _,bh,_=unpack(thM,dims,shapes); bias=float((bh-b1).abs().max())
    tag="MACHINE" if bias<1e-4 else ("stuck~guess" if abs(bias-berr)<0.3*berr else "other")
    print(f"seed{sd}: jetfit_resid^2={bestc:.2e} eps_z2={e2:.3f} eps_z3={e3:.3f} | joint LM bias={bias:.2e} ({tag}) resid^2={cj:.1e}",flush=True)
