"""Can full-cube + full-Jacobian refinement solve the NON-consensus L2 neurons too?
Prior run gave the 11 non-cons rows their TRUE weights (oracle). Here init ALL 80 rows from the
committee: cons rows = consensus avg, NON-cons rows = per-neuron MEDIAN across members (a real, worse
guess -- these are exactly the neurons the committee disagrees on). Refine full-Jacobian, full-cube,
oracle b2+downstream. Score cons(69) vs non-cons(11) separately, before/after."""
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
cons=torch.zeros(80,dtype=torch.bool,device=dev)
W2init=AW.median(0).values.clone()                       # ALL rows = per-neuron committee median (realistic)
for j in range(80):
    med=AW[:,j].median(0).values;cl=((AW[:,j]-med).norm(dim=1)/nt[j])<0.2
    if int(cl.sum())>=5: cons[j]=True; W2init[j]=AW[cl,j].mean(0)   # cons rows = consensus avg
ncons=~cons
def err(W2,mask):return ((W2[mask]-W2t[mask]).norm(dim=1)/nt[mask])
def G(h,W2):
    q2=torch.sigmoid(h@W2.t()+b2t);q3=torch.sigmoid(q2@Wl[2].t()+bl[2]);q4=torch.sigmoid(q3@Wl[3].t()+bl[3]);return q4@Wl[4].t()+bl[4]
N=600
gz=torch.Generator(device="cpu").manual_seed(0);scales=torch.tensor([1.5,4.0,10.0]);sidx=torch.randint(0,3,(N,),generator=gz)
z=torch.randn(N,128,generator=gz)*scales[sidx][:,None];h=torch.sigmoid(z).clamp(1e-5,1-1e-5).to(dev)
Qg=torch.Generator(device="cpu").manual_seed(3);Q,_=torch.linalg.qr(torch.randn(128,128,generator=Qg));Q=Q.to(dev)
valt=G(h,W2t).detach();scv=valt.abs().max()
def dd(W2):
    f=lambda hb: G(hb,W2)
    return torch.func.vmap(lambda u: jvp(f,(h,),(u.expand(N,128),))[1])(Q)   # full Jacobian (128 dirs)
ddt=dd(W2t).detach();scd=ddt.abs().max()
def resid(wf):
    W2=wf.reshape(sh)
    return torch.cat([((G(h,W2)-valt)/scv).reshape(-1),((dd(W2)-ddt)/scd).reshape(-1)])
def cg(A,b,it=60,tol=1e-14):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rs=r@r
    for _ in range(it):
        Ap=A(p);a=rs/(p@Ap+1e-30);x=x+a*p;r=r-a*Ap;rs2=r@r
        if rs2.sqrt()<tol:break
        p=r+(rs2/rs)*p;rs=rs2
    return x
def refine(w0,iters=70):
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
ec0=err(W2init,cons);en0=err(W2init,ncons)
print(f"cons {int(cons.sum())}  non-cons {int(ncons.sum())}  (full Jacobian, full-cube, N={N})",flush=True)
print(f"INIT  cons: mean {float(ec0.mean()):.4f} max {float(ec0.max()):.4f} | non-cons: mean {float(en0.mean()):.4f} max {float(en0.max()):.4f}",flush=True)
wf,c=refine(W2init.reshape(-1).clone())
ec1=err(wf.reshape(sh),cons);en1=err(wf.reshape(sh),ncons)
print(f"AFTER cons: mean {float(ec1.mean()):.5f} max {float(ec1.max()):.5f} <1% {int((ec1<0.01).sum())}/{int(cons.sum())}",flush=True)
print(f"AFTER non-cons: mean {float(en1.mean()):.5f} max {float(en1.max()):.5f} <1% {int((en1<0.01).sum())}/{int(ncons.sum())} <2% {int((en1<0.02).sum())}/{int(ncons.sum())} <5% {int((en1<0.05).sum())}/{int(ncons.sum())}",flush=True)
print("  per non-cons neuron  init->after:  "+"  ".join(f"{float(a):.3f}->{float(b):.3f}" for a,b in zip(en0,en1)),flush=True)
