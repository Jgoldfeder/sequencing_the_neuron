"""Is the committee weight guess NECESSARY or just HELPFUL? Keep the oracle ALIGNMENT (correct neuron
correspondence/perm/complement) but replace the weight VALUES with increasingly-wrong / random rows,
then full-cube + full-Jacobian refine (true b2+downstream). If even random-aligned converges to <1e-4,
the population guess is not necessary -- only correspondence is. Score over all 80 neurons; report loss
so spurious minima (loss>0) are visible. NOTE: <1e-4 is our TARGET, not float64 machine precision."""
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
AW=torch.stack(AW).to(dev);W2comm=AW.median(0).values.clone()
for j in range(80):
    med=AW[:,j].median(0).values;cl=((AW[:,j]-med).norm(dim=1)/nt[j])<0.2
    if int(cl.sum())>=5: W2comm[j]=AW[cl,j].mean(0)
def err(W2):return ((W2-W2t).norm(dim=1)/nt)
def G(h,W2):
    q2=torch.sigmoid(h@W2.t()+b2t);q3=torch.sigmoid(q2@Wl[2].t()+bl[2]);q4=torch.sigmoid(q3@Wl[3].t()+bl[3]);return q4@Wl[4].t()+bl[4]
N=600
gz=torch.Generator(device="cpu").manual_seed(0);scales=torch.tensor([1.5,4.0,10.0]);sidx=torch.randint(0,3,(N,),generator=gz)
z=torch.randn(N,128,generator=gz)*scales[sidx][:,None];h=torch.sigmoid(z).clamp(1e-5,1-1e-5).to(dev)
Qg=torch.Generator(device="cpu").manual_seed(3);Q,_=torch.linalg.qr(torch.randn(128,128,generator=Qg));Q=Q.to(dev)
valt=G(h,W2t).detach();scv=valt.abs().max()
def dd(W2):
    f=lambda hb: G(hb,W2)
    return torch.func.vmap(lambda u: jvp(f,(h,),(u.expand(N,128),))[1])(Q)
ddt=dd(W2t).detach();scd=ddt.abs().max()
def resid(wf):
    W2=wf.reshape(sh);return torch.cat([((G(h,W2)-valt)/scv).reshape(-1),((dd(W2)-ddt)/scd).reshape(-1)])
def cg(A,b,it=60,tol=1e-14):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rs=r@r
    for _ in range(it):
        Ap=A(p);a=rs/(p@Ap+1e-30);x=x+a*p;r=r-a*Ap;rs2=r@r
        if rs2.sqrt()<tol:break
        p=r+(rs2/rs)*p;rs=rs2
    return x
def refine(w0,iters=90):
    wf=w0.clone();lam=1e-4;r=resid(wf);c=float(r@r)
    for it in range(iters):
        _,vjpf=vjp(resid,wf);Jt=lambda u:vjpf(u)[0];Jv=lambda v:jvp(resid,(wf,),(v,))[1]
        gvec=Jt(r);A=lambda v:Jt(Jv(v))+lam*v;ok=False
        for _ in range(14):
            dwf=cg(A,-gvec);wn=wf+dwf;rn=resid(wn);cn=float(rn@rn)
            if cn<c:wf=wn;r=rn;c=cn;lam=max(lam*0.3,1e-15);ok=True;break
            lam*=5
        if not ok or c<1e-30:break
    return wf,c
Dg=torch.Generator(device="cpu").manual_seed(11);Dd=F.normalize(torch.randn(80,128,generator=Dg),dim=1).to(dev)
Dg2=torch.Generator(device="cpu").manual_seed(22);Dd2=F.normalize(torch.randn(80,128,generator=Dg2),dim=1).to(dev)
permg=torch.Generator(device="cpu").manual_seed(33);perm=torch.randperm(80,generator=permg).to(dev)
inits=[("committee", W2comm),
       ("aligned+1.0", W2t+1.0*nt[:,None]*Dd),
       ("aligned+2.0", W2t+2.0*nt[:,None]*Dd),
       ("random(true-norm)", nt[:,None]*Dd2),
       ("random(shuf-norm)", nt[perm][:,None]*Dd2)]
print(f"full-cube + full-Jacobian basin sweep (correspondence kept; true b2+downstream; N={N}). target <1e-4.",flush=True)
print(f"{'init':>18} {'init mean':>10} {'aft mean':>10} {'aft max':>10} {'<1e-4':>7} {'<1e-2':>7} {'loss':>10}",flush=True)
for name,W0 in inits:
    e0=err(W0);wf,c=refine(W0.reshape(-1).clone());e1=err(wf.reshape(sh))
    print(f"{name:>18} {float(e0.mean()):>10.4f} {float(e1.mean()):>10.2e} {float(e1.max()):>10.2e} {int((e1<1e-4).sum()):>5}/80 {int((e1<1e-2).sum()):>5}/80 {c:>10.2e}",flush=True)
