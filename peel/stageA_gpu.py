"""W-ONLY refinability of L2 on GPU, swept over #queries. b2=true, downstream=true, refine W2 only.
Question: does a good weight GUESS refine to machine precision if we throw more queries at it?
 - If true+0.02 / consensus reach ~1e-12 at large N -> floor was under-determination (surmountable by queries).
 - If they floor regardless of N while true+1e-6 hits 1e-15 -> the weight sits in a genuinely flat direction.
Strong solver: CG100, 100 outer, float64."""
import sys, os, torch, numpy as np
from torch.func import jvp, vjp
torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
dev="cuda"
T="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt"
teacher=MLP([784,128,80,40,32,10],act="sigmoid").to(dev);teacher.load_state_dict(torch.load(T,map_location=dev,weights_only=False))
Wl=[teacher.layers[i].weight.detach() for i in range(5)];bl=[teacher.layers[i].bias.detach() for i in range(5)]
W1t,b1t=Wl[0],bl[0];W2t=Wl[1];b2t=bl[1];nt=W2t.norm(dim=1);sh=W2t.shape
CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
pk=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location="cpu",weights_only=False)
def match(A,B):
    Cp=torch.cdist(A,B);Cm=torch.cdist(-A,B);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.numpy())
    s=torch.where(Cm[r,c]<Cp[r,c],-1.,1.);return torch.tensor(r),torch.tensor(c),s
def align(sd):
    W1m=sd["layers.0.weight"].double();W2m=sd["layers.1.weight"].double();b2m=sd["layers.1.bias"].double()
    r1,c1,s1=match(W1m,W1t.cpu());W2c=torch.zeros_like(W2m);W2c[:,c1]=W2m[:,r1]*s1[None,:]
    r2,c2,s2=match(W2c,W2t.cpu());Wa=torch.zeros_like(W2c);Wa[c2]=W2c[r2]*s2[:,None];return Wa
AW=torch.stack([align(sd) for sd in pk["pop_states"]]).to(dev)
W2c=W2t.clone();cons=torch.zeros(80,dtype=torch.bool,device=dev)
for j in range(80):
    med=AW[:,j].median(0).values;cl=((AW[:,j]-med).norm(dim=1)/nt[j])<0.2
    if int(cl.sum())>=5:W2c[j]=AW[cl,j].mean(0);cons[j]=True
def G(h,W2):
    q2=torch.sigmoid(h@W2.t()+b2t);q3=torch.sigmoid(q2@Wl[2].t()+bl[2]);q4=torch.sigmoid(q3@Wl[3].t()+bl[3]);return q4@Wl[4].t()+bl[4]
def cg(A,b,it=100,tol=1e-14):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rs=r@r
    for _ in range(it):
        Ap=A(p);a=rs/(p@Ap+1e-30);x=x+a*p;r=r-a*Ap;rs2=r@r
        if rs2.sqrt()<tol:break
        p=r+(rs2/rs)*p;rs=rs2
    return x
def werr(W2):return float(((W2[cons]-W2t[cons]).norm(dim=1)/nt[cons]).mean())
gg=torch.Generator(device="cpu").manual_seed(1);Pd=torch.randn(80,128,generator=gg);Pd=(Pd/Pd.norm(dim=1,keepdim=True)).to(dev)
inits=[("true+1e-6",W2t+1e-6*nt[:,None]*Pd),("true+0.02",W2t+0.02*nt[:,None]*Pd),
       ("consensus",W2c)]
for N in [30000, 100000]:
    g=torch.Generator(device="cpu").manual_seed(0);X=torch.rand(N,784,generator=g).to(dev);h=torch.sigmoid(X@W1t.t()+b1t)
    Yt=G(h,W2t).detach();sc=Yt.abs().max()
    def resid(wf):return ((G(h,wf.reshape(sh))-Yt)/sc).reshape(-1)
    def refine(w0,iters=100):
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
    print(f"\n=== N={N} queries (residuals {N*10} vs params 10240) ===",flush=True)
    print(f"{'init':>12} {'loss@init':>11} {'W before':>9} {'W after':>9} {'loss after':>11}",flush=True)
    for name,W0 in inits:
        li=float(resid(W0.reshape(-1))@resid(W0.reshape(-1)))
        wf,c=refine(W0.reshape(-1).clone())
        print(f"{name:>12} {li:>11.2e} {werr(W0):>9.4f} {werr(wf.reshape(sh)):>9.5f} {c:>11.2e}",flush=True)
