"""Decide: is the W2 stall a flat valley or a weak solver? Heavy GN-LM (3000 samples, CG150, 120 outer,
float64), TRUE b2 + TRUE downstream, refine W2 ONLY. Map refinability vs init distance:
 - control true+1e-6  (solver must reach ~1e-12 -> proves solver works)
 - true+0.02, true+0.05, true+0.10  (synthetic, structured-free)
 - REAL consensus (0.158)
If consensus -> machine precision, weight is refinable (Wall1 = artifact). If it floors at ~2e-5/0.14
while the control hits 1e-12, the last chunk of W2 is genuinely flat (carries ~no functional signal)."""
import sys, torch, numpy as np
from torch.func import jvp, vjp
torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
dev="cpu"
T="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt"
teacher=MLP([784,128,80,40,32,10],act="sigmoid");teacher.load_state_dict(torch.load(T,map_location=dev,weights_only=False))
Wl=[teacher.layers[i].weight.detach() for i in range(5)];bl=[teacher.layers[i].bias.detach() for i in range(5)]
W1t,b1t=Wl[0],bl[0];W2t=Wl[1];b2t=bl[1];nt=W2t.norm(dim=1);sh=W2t.shape
CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
pk=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False)
def match(A,B):
    Cp=torch.cdist(A,B);Cm=torch.cdist(-A,B);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.numpy())
    s=torch.where(Cm[r,c]<Cp[r,c],-1.,1.);return torch.tensor(r),torch.tensor(c),s
def align(sd):
    W1m=sd["layers.0.weight"].double();W2m=sd["layers.1.weight"].double();b2m=sd["layers.1.bias"].double()
    r1,c1,s1=match(W1m,W1t);W2c=torch.zeros_like(W2m);W2c[:,c1]=W2m[:,r1]*s1[None,:]
    r2,c2,s2=match(W2c,W2t);Wa=torch.zeros_like(W2c);ba=torch.zeros_like(b2m);Wa[c2]=W2c[r2]*s2[:,None];ba[c2]=b2m[r2]*s2;return Wa,ba
AW=torch.stack([align(sd)[0] for sd in pk["pop_states"]])
W2c=W2t.clone();cons=torch.zeros(80,dtype=torch.bool)
for j in range(80):
    med=AW[:,j].median(0).values;cl=((AW[:,j]-med).norm(dim=1)/nt[j])<0.2
    if int(cl.sum())>=5:W2c[j]=AW[cl,j].mean(0);cons[j]=True
def G(h,W2):
    q2=torch.sigmoid(h@W2.t()+b2t);q3=torch.sigmoid(q2@Wl[2].t()+bl[2]);q4=torch.sigmoid(q3@Wl[3].t()+bl[3]);return q4@Wl[4].t()+bl[4]
g=torch.Generator(device=dev).manual_seed(0);X=torch.rand(1200,784,generator=g);h=torch.sigmoid(X@W1t.t()+b1t)
Yt=G(h,W2t).detach();sc=Yt.abs().max()
def resid(wf):return ((G(h,wf.reshape(sh))-Yt)/sc).reshape(-1)
def cg(A,b,it=40,tol=1e-13):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rs=r@r
    for _ in range(it):
        Ap=A(p);a=rs/(p@Ap+1e-30);x=x+a*p;r=r-a*Ap;rs2=r@r
        if rs2.sqrt()<tol:break
        p=r+(rs2/rs)*p;rs=rs2
    return x
def werr(W2):return float(((W2[cons]-W2t[cons]).norm(dim=1)/nt[cons]).mean())
def refine(w0,iters=60):
    wf=w0.clone();lam=1e-4;r=resid(wf);c=float(r@r)
    for it in range(iters):
        _,vjpf=vjp(resid,wf);Jt=lambda u:vjpf(u)[0];Jv=lambda v:jvp(resid,(wf,),(v,))[1]
        gvec=Jt(r);A=lambda v:Jt(Jv(v))+lam*v;ok=False
        for _ in range(12):
            dwf=cg(A,-gvec);wn=wf+dwf;rn=resid(wn);cn=float(rn@rn)
            if cn<c:wf=wn;r=rn;c=cn;lam=max(lam*0.3,1e-14);ok=True;break
            lam*=5
        if not ok or c<1e-28:break
    return wf,c
gg=torch.Generator(device=dev).manual_seed(1);Pd=torch.randn(80,128,generator=gg);Pd=Pd/Pd.norm(dim=1,keepdim=True)
print(f"HEAVY GN-LM (3000 samples, CG150, 120 outer). loss@true={float(resid(W2t.reshape(-1))@resid(W2t.reshape(-1))):.1e}")
print(f"{'init':>16} {'W-err before':>12} {'W-err after':>12} {'loss after':>12}")
inits=[("true+1e-6",(W2t+1e-6*nt[:,None]*Pd)),("true+0.02",(W2t+0.02*nt[:,None]*Pd)),
       ("true+0.05",(W2t+0.05*nt[:,None]*Pd)),("true+0.10",(W2t+0.10*nt[:,None]*Pd)),
       ("consensus",W2c)]
for name,W0 in inits:
    wf,c=refine(W0.reshape(-1).clone());print(f"{name:>16} {werr(W0):>12.4f} {werr(wf.reshape(sh)):>12.5f} {c:>12.2e}",flush=True)
