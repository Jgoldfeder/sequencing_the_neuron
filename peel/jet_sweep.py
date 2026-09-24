"""Does jet-matching BREAK the ~5% floor or just lower the constant? Sweep the jet objective over N,
and add a 2nd-order variant (match directional 2nd derivatives too). Committee L2, oracle b2/downstream
=true, GN-LM GPU. If jet W-err keeps dropping toward 0 with N -> jets solve it; if it also floors -> same wall."""
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
m=16
print(f"committee cons {int(cons.sum())}/80  init W-err mean {float(werr(W2c).mean()):.4f}  (jet dirs m={m})",flush=True)
print(f"{'N':>7} {'objective':>12} {'W mean after':>13} {'<5%':>7} {'<2%':>7} {'ratio':>7}",flush=True)
for N in [1500, 6000, 20000]:
    g=torch.Generator(device="cpu").manual_seed(0);X=torch.rand(N,784,generator=g).to(dev);h=torch.sigmoid(X@W1t.t()+b1t)
    Ug=torch.Generator(device="cpu").manual_seed(7);U=F.normalize(torch.randn(m,128,generator=Ug),dim=1).to(dev)
    def val_dd(W2):
        def f(hb): return G(hb,W2)
        val=f(h); dd=torch.func.vmap(lambda u: jvp(f,(h,),(u.expand(N,128),))[1])(U)   # (m,N,10)
        return val,dd
    def val_dd2(W2):                                  # + directional 2nd derivative
        def f(hb): return G(hb,W2)
        val=f(h)
        def d12(u):
            uu=u.expand(N,128)
            d1=lambda hb: jvp(f,(hb,),(uu,))[1]        # 1st dir-deriv as fn of h
            first=d1(h); _,second=jvp(d1,(h,),(uu,))   # 2nd dir-deriv
            return first,second
        d1,d2=torch.func.vmap(d12)(U); return val,d1,d2
    valt,ddt=val_dd(W2c*0+W2t); valt=valt.detach();ddt=ddt.detach()
    scv=valt.abs().max();scd=ddt.abs().max()
    _,d1t,d2t=val_dd2(W2t); d1t=d1t.detach();d2t=d2t.detach();scd2=d2t.abs().max()
    def r_val(wf): return ((G(h,wf.reshape(sh))-valt)/scv).reshape(-1)
    def r_jet(wf):
        val,dd=val_dd(wf.reshape(sh));return torch.cat([((val-valt)/scv).reshape(-1),((dd-ddt)/scd).reshape(-1)])
    def r_jet2(wf):
        val,d1,d2=val_dd2(wf.reshape(sh))
        return torch.cat([((val-valt)/scv).reshape(-1),((d1-d1t)/scd).reshape(-1),((d2-d2t)/scd2).reshape(-1)])
    for nm,rf in [("value",r_val),("jet1",r_jet),("jet2",r_jet2)]:
        wf,c=refine(W2c.reshape(-1).clone(),rf);e1=werr(wf.reshape(sh))
        print(f"{N:>7} {nm:>12} {float(e1.mean()):>13.4f} {int((e1<0.05).sum()):>7} {int((e1<0.02).sum()):>7} {float((e1/werr(W2c)).median()):>7.3f}",flush=True)
