"""Change the GEOMETRY, not N (per Judah's critique). Two fixes:
 (1) sample h DIRECTLY across the whole cube (mixture: interior + near-corner) -- all reachable after
     peeling L1 via x=W1^+(logit(h)-b1), not the narrow sigma(W1*U([0,1]^784)) distribution.
 (2) sweep the derivative SPAN m=16,32,64,128 with a nested orthonormal basis; m=128 == full Jacobian.
Decisive: if committee W2-err collapses as m->128 on full-cube h, the '5% floor' was under-excitation.
If it stays ~5% with full cube + full Jacobian + true b2/downstream, it's a real structural result.
Oracle assists kept explicit: W2c non-consensus rows start at truth; Hungarian-aligned; score over the 69 cons only."""
import sys, torch
import torch.nn.functional as F
from torch.func import jvp, vjp
torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
dev="cuda"
T="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt"
teacher=MLP([784,128,80,40,32,10],act="sigmoid").to(dev);teacher.load_state_dict(torch.load(T,map_location=dev,weights_only=False))
Wl=[teacher.layers[i].weight.detach() for i in range(5)];bl=[teacher.layers[i].bias.detach() for i in range(5)]
b2t=bl[1];W2t=Wl[1];nt=W2t.norm(dim=1);sh=W2t.shape
pk=torch.load("peel_committee.pt",map_location="cpu",weights_only=False);W2T=W2t.cpu()
AW=[]
for sd in pk["pop_states"]:
    W2m=sd["layers.1.weight"].double();Cp=torch.cdist(W2m,W2T);Cm=torch.cdist(-W2m,W2T);C=torch.minimum(Cp,Cm)
    r,c=linear_sum_assignment(C.numpy());r=torch.tensor(r);c=torch.tensor(c);sgn=torch.where(Cm[r,c]<Cp[r,c],-1.,1.)
    W2a=torch.zeros_like(W2m);W2a[c]=W2m[r]*sgn[:,None];AW.append(W2a)
AW=torch.stack(AW).to(dev)
W2c=W2t.clone();cons=torch.zeros(80,dtype=torch.bool,device=dev)
for j in range(80):
    med=AW[:,j].median(0).values;cl=((AW[:,j]-med).norm(dim=1)/nt[j])<0.2
    if int(cl.sum())>=5:W2c[j]=AW[cl,j].mean(0);cons[j]=True
def werr(W2):return ((W2[cons]-W2t[cons]).norm(dim=1)/nt[cons])
def G(h,W2):
    q2=torch.sigmoid(h@W2.t()+b2t);q3=torch.sigmoid(q2@Wl[2].t()+bl[2]);q4=torch.sigmoid(q3@Wl[3].t()+bl[3]);return q4@Wl[4].t()+bl[4]
# ---- full-cube h: mixture of interior + near-corner (all reachable after peeling L1) ----
N=600
gz=torch.Generator(device="cpu").manual_seed(0)
scales=torch.tensor([1.5,4.0,10.0]);sidx=torch.randint(0,3,(N,),generator=gz)
z=torch.randn(N,128,generator=gz)*scales[sidx][:,None]
h=torch.sigmoid(z).clamp(1e-5,1-1e-5).to(dev)
print(f"h geometry: cube mixture, h mean {float(h.mean()):.3f}  frac<0.05or>0.95 {float(((h<0.05)|(h>0.95)).float().mean()):.2f}",flush=True)
# nested orthonormal basis for jet directions
Qg=torch.Generator(device="cpu").manual_seed(3);Q,_=torch.linalg.qr(torch.randn(128,128,generator=Qg));Q=Q.to(dev)
def cg(A,b,it=50,tol=1e-14):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rs=r@r
    for _ in range(it):
        Ap=A(p);a=rs/(p@Ap+1e-30);x=x+a*p;r=r-a*Ap;rs2=r@r
        if rs2.sqrt()<tol:break
        p=r+(rs2/rs)*p;rs=rs2
    return x
def refine(w0,resid,iters=50):
    wf=w0.clone();lam=1e-4;r=resid(wf);c=float(r@r)
    for it in range(iters):
        _,vjpf=vjp(resid,wf);Jt=lambda u:vjpf(u)[0];Jv=lambda v:jvp(resid,(wf,),(v,))[1]
        gvec=Jt(r);A=lambda v:Jt(Jv(v))+lam*v;ok=False
        for _ in range(12):
            dwf=cg(A,-gvec);wn=wf+dwf;rn=resid(wn);cn=float(rn@rn)
            if cn<c:wf=wn;r=rn;c=cn;lam=max(lam*0.3,1e-15);ok=True;break
            lam*=5
        if not ok or c<1e-28:break
    return wf,c
valt=G(h,W2t).detach();scv=valt.abs().max()
def make_resid(mm):
    if mm==0:
        return lambda wf: ((G(h,wf.reshape(sh))-valt)/scv).reshape(-1)
    U=Q[:mm]
    def dd(W2):
        f=lambda hb: G(hb,W2)
        return torch.func.vmap(lambda u: jvp(f,(h,),(u.expand(N,128),))[1])(U)   # (mm,N,10)
    ddt=dd(W2t).detach();scd=ddt.abs().max()
    def resid(wf):
        W2=wf.reshape(sh)
        return torch.cat([((G(h,W2)-valt)/scv).reshape(-1),((dd(W2)-ddt)/scd).reshape(-1)])
    return resid
gg=torch.Generator(device="cpu").manual_seed(1);Pd=F.normalize(torch.randn(80,128,generator=gg),dim=1).to(dev)
print(f"committee cons {int(cons.sum())}/80  init W-err mean {float(werr(W2c).mean()):.4f} max {float(werr(W2c).max()):.4f}  (N={N})",flush=True)
ctrl,_=refine((W2t+1e-6*nt[:,None]*Pd).reshape(-1).clone(),make_resid(128),iters=30)
print(f"control(m=128) true+1e-6 -> {float(werr(ctrl.reshape(sh)).mean()):.2e}\n",flush=True)
print(f"{'m (jet span)':>13} {'W mean after':>13} {'W max after':>12} {'<5%':>6} {'<2%':>6} {'<1%':>6} {'ratio':>7}",flush=True)
for mm in [0,16,32,64,128]:
    wf,c=refine(W2c.reshape(-1).clone(),make_resid(mm))
    e1=werr(wf.reshape(sh))
    lab=f"{mm} (value)" if mm==0 else (f"{mm} (fullJac)" if mm==128 else str(mm))
    print(f"{lab:>13} {float(e1.mean()):>13.4f} {float(e1.max()):>12.4f} {int((e1<0.05).sum()):>6} {int((e1<0.02).sum()):>6} {int((e1<0.01).sum()):>6} {float((e1/werr(W2c)).median()):>7.3f}",flush=True)
