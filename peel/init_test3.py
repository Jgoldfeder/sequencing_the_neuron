import torch, numpy as np
from torch.func import jacfwd
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
exec(open('realgate2.py').read().split('print("JOINT')[0])
from scipy.optimize import linear_sum_assignment
def lm(th0,rf,iters=120):
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
def Gfwd(s,down):
    W2,b2,W3,b3,w4,b4=down; q=sig(s@W2.T+b2); h=sig(q@W3.T+b3); return h@w4.T+b4
def preacts(s,down):
    W2,b2,W3,b3,w4,b4=down; z2=s@W2.T+b2; q=sig(z2); z3=q@W3.T+b3; return z2,z3
def match(As,At):                       # FIXED: return teacher-ordered student index + signs
    d=As.shape[1]; C=np.zeros((d,d)); Sg=np.zeros((d,d))
    for i in range(d):
        a=As[:,i]-As[:,i].mean()
        for j in range(d):
            b=At[:,j]-At[:,j].mean(); cc=float((a@b)/(a.norm()*b.norm()+1e-12))
            Sg[i,j]=1.0 if cc>=0 else -1.0; C[i,j]=-abs(cc)
    ci=np.asarray(linear_sum_assignment(C)[1])   # student i -> teacher ci[i]
    inv=np.argsort(ci)                            # teacher j -> student inv[j]
    signs=np.array([Sg[i,ci[i]] for i in range(d)])
    return inv, torch.tensor(signs[inv])
def align_eps(stu,down,s):
    z2t,z3t=preacts(s,down); z2s,z3s=preacts(s,stu)
    i2,g2=match(z2s,z2t); az2=z2s[:,i2]*g2; i3,g3=match(z3s,z3t); az3=z3s[:,i3]*g3
    return (float((az2-z2t).pow(2).mean().sqrt()/z2t.pow(2).mean().sqrt()),
            float((az3-z3t).pow(2).mean().sqrt()/z3t.pow(2).mean().sqrt()))
def perm_comp_copy(down,seedp):         # exact perm+complement copy (functional identity)
    W2,b2,W3,b3,w4,b4=[d.clone() for d in down]; g=torch.Generator().manual_seed(seedp)
    P2=torch.randperm(8,generator=g); s2=(torch.randint(0,2,(8,),generator=g)*2-1).double()
    W2n=s2[:,None]*W2[P2]; b2n=s2*b2[P2]
    W3n=W3[:,P2]*s2[None,:]; b3n=b3+(W3[:,P2]*((1-s2)/2)[None,:]).sum(1)
    P3=torch.randperm(4,generator=g); s3=(torch.randint(0,2,(4,),generator=g)*2-1).double()
    W3n2=s3[:,None]*W3n[P3]; b3n2=s3*b3n[P3]
    w4n=w4[:,P3]*s3[None,:]; b4n=b4+(w4[:,P3]*((1-s3)/2)[None,:]).sum(1)
    return [W2n,b2n,W3n2,b3n2,w4n,b4n]
def fdown(down): return torch.cat([d.reshape(-1) for d in down])
def udown(v,shapes):
    out=[];i=0
    for sh in shapes:
        nn=int(np.prod(sh)); out.append(v[i:i+nn].reshape(sh)); i+=nn
    return out
def train_io(shat,y,shapes,nrest,steps,seed):
    best=None;bestr=1e18
    for r in range(nrest):
        g=torch.Generator().manual_seed(seed*137+r*7+1)
        stu=[(torch.randn(sh,generator=g)/np.sqrt(sh[1])).requires_grad_(True) if len(sh)==2
             else ((2*torch.rand(sh,generator=g)-1)*0.3).requires_grad_(True) for sh in shapes]
        opt=torch.optim.Adam(stu,lr=0.05)
        for it in range(steps):
            opt.zero_grad(); (((Gfwd(shat,stu)-y)**2).mean()).backward(); opt.step()
            if it==steps//2:
                for gp in opt.param_groups: gp['lr']=0.01
        rr=float(((Gfwd(shat,stu)-y)**2).mean())
        if rr<bestr: bestr=rr; best=[t.detach().clone() for t in stu]
    return best,bestr
def cls(b): return "<1e-10" if b<1e-10 else "<1e-6" if b<1e-6 else "<1e-4" if b<1e-4 else f"FAIL({b:.1e})"
dims=[6,8,4,4]; k=6
# ---- metric self-validation ----
S0,b0,dn0=make(dims,0); gg=torch.Generator().manual_seed(5); ss=sig((2*torch.rand(300,k,generator=gg)-1)@torch.eye(k)+b0)
cp=perm_comp_copy(dn0,3); fid=float((Gfwd(ss,cp)-Gfwd(ss,dn0)).abs().max()); ev=align_eps(cp,dn0,ss)
print(f"METRIC CHECK: exact perm/complement copy -> functional |dF|max={fid:.1e}, aligned eps_z2={ev[0]:.2e} eps_z3={ev[1]:.2e} (both must be ~0)",flush=True)
print("\nCLEANUP: I/O-fit vs JET-fit initializers (fixed alignment, held-out errors, explicit thresholds)",flush=True)
for sd in [0,1,2]:
    Sstar,b1,down=make(dims,sd); shapes=[tuple(d.shape) for d in down]
    berr=0.01; rng=np.random.default_rng(sd*10+1); betahat=b1+berr*torch.tensor(2*rng.random(k)-1)
    gq=torch.Generator().manual_seed(sd+7); tq=(2*torch.rand(1500,k,generator=gq)-1)*1.4
    sstar=sig(tq@Sstar.T+b1); y=Gfwd(sstar,down); shat=sig(tq@torch.eye(k)+betahat)
    gh=torch.Generator().manual_seed(sd+300); th_=(2*torch.rand(1000,k,generator=gh)-1)*1.4     # held-out I/O
    sst_h=sig(th_@Sstar.T+b1); y_h=Gfwd(sst_h,down); shat_h=sig(th_@torch.eye(k)+betahat); yn=y_h.pow(2).mean().sqrt()
    gp=torch.Generator().manual_seed(sd+99); te=(2*torch.rand(400,k,generator=gp)-1)*1.2; se=sig(te@Sstar.T+b1)
    gf=torch.Generator().manual_seed(sd+21); tpf=(2*torch.rand(40,k,generator=gf)-1)*1.3
    with torch.no_grad(): Fmf=raw_jet_S(Sstar,b1,down,tpf,k); scf=Fmf.abs().max()
    ghp=torch.Generator().manual_seed(sd+511); tph=(2*torch.rand(25,k,generator=ghp)-1)*1.3     # held-out jet
    with torch.no_grad(): Fmh=raw_jet_S(Sstar,b1,down,tph,k); sch=Fmh.abs().max()
    gp2=torch.Generator().manual_seed(sd+3); tps2=(2*torch.rand(12,k,generator=gp2)-1)*1.2
    with torch.no_grad(): Fm2=raw_jet_S(Sstar,b1,down,tps2,k); sc2=Fm2.abs().max()
    rf2=lambda th:(raw_jet_S(*unpack(th,dims,shapes),tps2,k)-Fm2)/sc2
    def report(tag,stu):
        e2,e3=align_eps(stu,down,se)
        ioho=float((Gfwd(shat_h,stu)-y_h).pow(2).mean().sqrt()/yn)
        jetho=float(((raw_jet_S(torch.eye(k),betahat,stu,tph,k)-Fmh)/sch).pow(2).mean().sqrt())
        thM,cj=lm(flat(torch.eye(k),betahat,stu),rf2); _,bh,_=unpack(thM,dims,shapes); bias=float((bh-b1).abs().max())
        print(f"  seed{sd} {tag}: eps_z2={e2:.3f} eps_z3={e3:.3f} | heldout_IO={ioho:.2e} heldout_jet={jetho:.2e} | jointLM bias={cls(bias)}",flush=True)
    stu_io,res_io=train_io(shat,y,shapes,6,3000,sd*3+1); report(f"IO  (trainMSE={res_io:.1e})",stu_io)
    best=None;bestc=1e18
    rff=lambda dv:(raw_jet_S(torch.eye(k),betahat,udown(dv,shapes),tpf,k)-Fmf)/scf
    for r in range(8):
        g=torch.Generator().manual_seed(sd*50+r*3+1)
        d0=[torch.randn(sh,generator=g)/np.sqrt(sh[1]) if len(sh)==2 else (2*torch.rand(sh,generator=g)-1)*0.4 for sh in shapes]
        dM,c=lm(fdown(d0),rff,iters=80)
        if c<bestc: bestc=c; best=dM.clone()
    report(f"JET (fitres2={bestc:.1e})",udown(best,shapes))
