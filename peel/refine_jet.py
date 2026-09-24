"""JET-matching refinement of the committee L2 weights. Match not just G(h) values but its
directional derivatives D_u G(h) along m random input directions (1st-order jet) -- many more
constraints per query, targeting the flat directions that value-fitting leaves. Oracle b2/downstream
=true, GN-LM on GPU. Head-to-head vs value-only at matched N/solver on the SAME committee consensus."""
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
W1t,b1t=Wl[0],bl[0];W2t=Wl[1];b2t=bl[1];nt=W2t.norm(dim=1);sh=W2t.shape
pk=torch.load("peel_committee.pt",map_location="cpu",weights_only=False); W2T=W2t.cpu()
AW=[]
for sd in pk["pop_states"]:
    W2m=sd["layers.1.weight"].double()
    Cp=torch.cdist(W2m,W2T);Cm=torch.cdist(-W2m,W2T);C=torch.minimum(Cp,Cm)
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
N=1500; m=24
g=torch.Generator(device="cpu").manual_seed(0);X=torch.rand(N,784,generator=g).to(dev);h=torch.sigmoid(X@W1t.t()+b1t)
Ug=torch.Generator(device="cpu").manual_seed(7);U=F.normalize(torch.randn(m,128,generator=Ug),dim=1).to(dev)  # input dirs
def val_dd(W2):
    def f(hb): return G(hb,W2)
    val=f(h)
    dd_one=lambda u: jvp(f,(h,),(u.expand(N,128),))[1]         # (N,10)
    dds=torch.func.vmap(dd_one)(U)                              # (m,N,10) vectorized over dirs
    return val,dds
valt,ddt=val_dd(W2t); valt=valt.detach(); ddt=ddt.detach()
scv=valt.abs().max(); scd=ddt.abs().max()
def resid_val(wf):
    return ((G(h,wf.reshape(sh))-valt)/scv).reshape(-1)
def resid_jet(wf):
    W2=wf.reshape(sh);val,dds=val_dd(W2)
    return torch.cat([((val-valt)/scv).reshape(-1), ((dds-ddt)/scd).reshape(-1)])
def cg(A,b,it=60,tol=1e-14):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rs=r@r
    for _ in range(it):
        Ap=A(p);a=rs/(p@Ap+1e-30);x=x+a*p;r=r-a*Ap;rs2=r@r
        if rs2.sqrt()<tol:break
        p=r+(rs2/rs)*p;rs=rs2
    return x
def refine(w0,resid,iters=60):
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
gg=torch.Generator(device="cpu").manual_seed(1);Pd=F.normalize(torch.randn(80,128,generator=gg),dim=1).to(dev)
print(f"committee cons {int(cons.sum())}/80  init W-err mean {float(werr(W2c).mean()):.4f} max {float(werr(W2c).max()):.4f}  (N={N}, m={m} jet dirs)",flush=True)
# solver sanity on jet objective
ctrl,_=refine((W2t+1e-6*nt[:,None]*Pd).reshape(-1).clone(),resid_jet,iters=40)
print(f"control(jet) true+1e-6 -> {float(werr(ctrl.reshape(sh)).mean()):.2e}",flush=True)
for name,resid in [("VALUE-only",resid_val),("VALUE+JET",resid_jet)]:
    wf,c=refine(W2c.reshape(-1).clone(),resid)
    e0=werr(W2c);e1=werr(wf.reshape(sh))
    print(f"{name:>11}: W-err mean {float(e0.mean()):.4f}->{float(e1.mean()):.4f}  max {float(e0.max()):.4f}->{float(e1.max()):.4f}  "
          f"<5%:{int((e1<0.05).sum())}/{int(cons.sum())} <2%:{int((e1<0.02).sum())}/{int(cons.sum())}  ratio {float((e1/e0).median()):.3f}",flush=True)
