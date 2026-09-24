"""Refine the REAL current committee's L2 weight consensus (peel_committee.pt, ~70 neurons).
Oracle setup: b2=true, downstream=true, refine W2 only via strong GN-LM (GPU). Report per-neuron
W-err before/after so we see how much the actual committee guess refines (vs the earlier floor)."""
import sys, torch, numpy as np
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
pk=torch.load("peel_committee.pt",map_location="cpu",weights_only=False)
# ---- row-align each member's L2 to true (L1 frozen in true order => columns already aligned) ----
W2T=W2t.cpu()
AW=[]
for sd in pk["pop_states"]:
    W2m=sd["layers.1.weight"].double()
    Cp=torch.cdist(W2m,W2T);Cm=torch.cdist(-W2m,W2T);C=torch.minimum(Cp,Cm)
    r,c=linear_sum_assignment(C.numpy());r=torch.tensor(r);c=torch.tensor(c)
    sgn=torch.where(Cm[r,c]<Cp[r,c],-1.,1.)
    W2a=torch.zeros_like(W2m);W2a[c]=W2m[r]*sgn[:,None];AW.append(W2a)
AW=torch.stack(AW).to(dev)
W2c=W2t.clone();cons=torch.zeros(80,dtype=torch.bool,device=dev)
for j in range(80):
    med=AW[:,j].median(0).values;cl=((AW[:,j]-med).norm(dim=1)/nt[j])<0.2
    if int(cl.sum())>=5:W2c[j]=AW[cl,j].mean(0);cons[j]=True
def werr(W2):return ((W2[cons]-W2t[cons]).norm(dim=1)/nt[cons])
print(f"committee W-consensus: {int(cons.sum())}/80 neurons  init W-err mean {float(werr(W2c).mean()):.4f} max {float(werr(W2c).max()):.4f}",flush=True)
def G(h,W2):
    q2=torch.sigmoid(h@W2.t()+b2t);q3=torch.sigmoid(q2@Wl[2].t()+bl[2]);q4=torch.sigmoid(q3@Wl[3].t()+bl[3]);return q4@Wl[4].t()+bl[4]
def cg(A,b,it=120,tol=1e-14):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rs=r@r
    for _ in range(it):
        Ap=A(p);a=rs/(p@Ap+1e-30);x=x+a*p;r=r-a*Ap;rs2=r@r
        if rs2.sqrt()<tol:break
        p=r+(rs2/rs)*p;rs=rs2
    return x
def refine(w0,resid,iters=120):
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
N=30000
g=torch.Generator(device="cpu").manual_seed(0);X=torch.rand(N,784,generator=g).to(dev);h=torch.sigmoid(X@W1t.t()+b1t)
Yt=G(h,W2t).detach();sc=Yt.abs().max()
def resid(wf):return ((G(h,wf.reshape(sh))-Yt)/sc).reshape(-1)
# control: refine a near-true init on THIS setup (validate solver)
gg=torch.Generator(device="cpu").manual_seed(1);Pd=torch.randn(80,128,generator=gg);Pd=(Pd/Pd.norm(dim=1,keepdim=True)).to(dev)
ctrl,cc=refine((W2t+1e-6*nt[:,None]*Pd).reshape(-1).clone(),resid,iters=60)
print(f"control true+1e-6 -> W-err mean {float(werr(ctrl.reshape(sh)).mean()):.2e} loss {cc:.2e}  (solver check)",flush=True)
# refine the committee consensus
wf,c=refine(W2c.reshape(-1).clone(),resid)
e0=werr(W2c);e1=werr(wf.reshape(sh))
print(f"\nCOMMITTEE consensus refine (N={N}, b2/downstream=true):",flush=True)
print(f"  W-err mean {float(e0.mean()):.4f} -> {float(e1.mean()):.4f}   max {float(e0.max()):.4f} -> {float(e1.max()):.4f}   loss {c:.2e}",flush=True)
print(f"  neurons improved to <2%: {int((e1<0.02).sum())}/{int(cons.sum())} ; <5%: {int((e1<0.05).sum())}/{int(cons.sum())}",flush=True)
print(f"  reduction ratio (after/before, per-neuron median): {float((e1/e0).median()):.3f}",flush=True)
torch.save({"W2_refined":wf.reshape(sh).cpu(),"cons":cons.cpu(),"e_before":e0.cpu(),"e_after":e1.cpu()},"committee_refined.pt")
