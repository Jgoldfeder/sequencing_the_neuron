import torch, numpy as np
from torch.func import jacfwd
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
exec(open('realgate2.py').read().split('print("JOINT')[0])
try: from scipy.optimize import linear_sum_assignment; HAVE=True
except Exception: HAVE=False
def lm(th0,rf,iters=120):
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
def Gfwd(s,down):
    W2,b2,W3,b3,w4,b4=down; q=sig(s@W2.T+b2); h=sig(q@W3.T+b3); return h@w4.T+b4
def preacts(s,down):
    W2,b2,W3,b3,w4,b4=down; z2=s@W2.T+b2; q=sig(z2); z3=q@W3.T+b3; return z2,z3
def train_student(shat,y,shapes,nrest,steps,seed):
    best=None;bestr=1e18
    for r in range(nrest):
        g=torch.Generator().manual_seed(seed*137+r*7+1)
        stu=[]
        for sh in shapes:
            if len(sh)==2: stu.append((torch.randn(sh,generator=g)/np.sqrt(sh[1])).requires_grad_(True))
            else: stu.append(((2*torch.rand(sh,generator=g)-1)*0.3).requires_grad_(True))
        opt=torch.optim.Adam(stu,lr=0.05)
        for it in range(steps):
            opt.zero_grad(); loss=((Gfwd(shat,stu)-y)**2).mean(); loss.backward(); opt.step()
            if it==steps//2:
                for gp in opt.param_groups: gp['lr']=0.01
        rr=float(((Gfwd(shat,stu)-y)**2).mean())
        if rr<bestr: bestr=rr; best=[t.detach().clone() for t in stu]
    return best,bestr
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
    p2,g2=match(z2s,z2t); az2=z2s[:,p2]*g2
    p3,g3=match(z3s,z3t); az3=z3s[:,p3]*g3
    e2=float((az2-z2t).pow(2).mean().sqrt()/z2t.pow(2).mean().sqrt())
    e3=float((az3-z3t).pow(2).mean().sqrt()/z3t.pow(2).mean().sqrt())
    return e2,e3
dims=[6,8,4,4]; k=6
print("Query-access student-fit initializer -> joint LM.  (guess bias off by berr; can LM reach machine?)",flush=True)
for sd in [0,1,2]:
    Sstar,b1,down=make(dims,sd); shapes=[tuple(d.shape) for d in down]
    for berr in [0.01,0.05]:
        rng=np.random.default_rng(sd*10+int(berr*100))
        betahat=b1+berr*torch.tensor(2*rng.random(k)-1); Shat=torch.eye(k)
        gq=torch.Generator().manual_seed(sd+7); tq=(2*torch.rand(1500,k,generator=gq)-1)*1.4
        sstar=sig(tq@Sstar.T+b1); y=Gfwd(sstar,down); shat=sig(tq@Shat.T+betahat)
        stu,res=train_student(shat,y,shapes,nrest=8,steps=4000,seed=sd*3+int(berr*100))
        gp=torch.Generator().manual_seed(sd+99); te=(2*torch.rand(400,k,generator=gp)-1)*1.2
        se=sig(te@Sstar.T+b1); e2,e3=align_eps(stu,down,se)
        gp2=torch.Generator().manual_seed(sd+3); tps=(2*torch.rand(12,k,generator=gp2)-1)*1.2
        with torch.no_grad(): Fm=raw_jet_S(Sstar,b1,down,tps,k); sc=Fm.abs().max()
        th0=flat(torch.eye(k),betahat,stu)
        rf=lambda th:(raw_jet_S(*unpack(th,dims,shapes),tps,k)-Fm)/sc
        thM,c=lm(th0,rf); _,bh,_=unpack(thM,dims,shapes); bias=float((bh-b1).abs().max())
        stuck="STUCK@guess" if abs(bias-berr)<0.3*berr else ("MACHINE" if bias<1e-4 else "other")
        print(f"seed{sd} berr={berr:.2f}: student_res={res:.2e} eps_z2={e2:.3f} eps_z3={e3:.3f} | LM bias={bias:.2e} ({stuck}) resid^2={c:.1e}",flush=True)
