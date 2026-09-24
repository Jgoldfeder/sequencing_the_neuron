"""Oracle gate (user's design): does the REAL consensus W2 refine to machine precision?
Stage A: consensus W2 + true b2 + TRUE downstream -> GN-LM refine W2. If ->0, weight is in-basin.
Stage B: bias basin. Fix consensus W2, set b2 = true + delta (frozen wrong), refine W2 only, and
         separately refine (W2,b2) jointly, over delta in {0,.01,.03,.05,.1,.2,.5}. Where does it break?
All with TRUE downstream first (isolate the bias question)."""
import sys, torch, numpy as np
from torch.func import jvp, vjp
torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
dev = "cpu"
T = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt"
teacher = MLP([784,128,80,40,32,10], act="sigmoid"); teacher.load_state_dict(torch.load(T, map_location=dev, weights_only=False))
Wl=[teacher.layers[i].weight.detach() for i in range(5)]; bl=[teacher.layers[i].bias.detach() for i in range(5)]
W1t,b1t=Wl[0],bl[0]; W2t=Wl[1]; b2t=bl[1]; nt=W2t.norm(dim=1); sh=W2t.shape
# ---- build real consensus (eps=0.2) ----
CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
pk=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False)
def match(A,B):
    Cp=torch.cdist(A,B);Cm=torch.cdist(-A,B);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.numpy())
    s=torch.where(Cm[r,c]<Cp[r,c],-1.,1.);return torch.tensor(r),torch.tensor(c),s
def align(sd):
    W1m=sd["layers.0.weight"].double();W2m=sd["layers.1.weight"].double();b2m=sd["layers.1.bias"].double()
    r1,c1,s1=match(W1m,W1t);W2c=torch.zeros_like(W2m);W2c[:,c1]=W2m[:,r1]*s1[None,:]
    r2,c2,s2=match(W2c,W2t);Wa=torch.zeros_like(W2c);ba=torch.zeros_like(b2m);Wa[c2]=W2c[r2]*s2[:,None];ba[c2]=b2m[r2]*s2;return Wa,ba
AW=[];AB=[]
for sd in pk["pop_states"]:
    w,b=align(sd);AW.append(w);AB.append(b)
AW=torch.stack(AW);AB=torch.stack(AB)
W2c=W2t.clone(); b2c=b2t.clone(); cons=torch.zeros(80,dtype=torch.bool)
for j in range(80):
    med=AW[:,j].median(0).values; cl=((AW[:,j]-med).norm(dim=1)/nt[j])<0.2
    if int(cl.sum())>=5: W2c[j]=AW[cl,j].mean(0); b2c[j]=AB[cl,j].mean(); cons[j]=True
print(f"consensus neurons (eps0.2): {int(cons.sum())}/80  init W rel-err(cons)={float(((W2c[cons]-W2t[cons]).norm(dim=1)/nt[cons]).mean()):.3f}")
# ---- G with TRUE downstream ----
def G(h,W2,b2):
    q2=torch.sigmoid(h@W2.t()+b2);q3=torch.sigmoid(q2@Wl[2].t()+bl[2]);q4=torch.sigmoid(q3@Wl[3].t()+bl[3]);return q4@Wl[4].t()+bl[4]
g=torch.Generator(device=dev).manual_seed(0); X=torch.rand(1200,784,generator=g); h=torch.sigmoid(X@W1t.t()+b1t)
Yt=G(h,W2t,b2t).detach(); sc=Yt.abs().max()
def cg(A,b,it=30,tol=1e-11):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rs=r@r
    for _ in range(it):
        Ap=A(p);a=rs/(p@Ap+1e-30);x=x+a*p;r=r-a*Ap;rs2=r@r
        if rs2.sqrt()<tol:break
        p=r+(rs2/rs)*p;rs=rs2
    return x
def gnlm(param0, resid, iters=50):
    wf=param0.clone(); lam=1e-3; r=resid(wf); c=float(r@r)
    for it in range(iters):
        _,vjpf=vjp(resid,wf); Jt=lambda u: vjpf(u)[0]; Jv=lambda v: jvp(resid,(wf,),(v,))[1]
        gvec=Jt(r); A=lambda v: Jt(Jv(v))+lam*v; ok=False
        for _ in range(10):
            dwf=cg(A,-gvec); wn=wf+dwf; rn=resid(wn); cn=float(rn@rn)
            if cn<c: wf=wn;r=rn;c=cn;lam=max(lam*0.4,1e-13);ok=True;break
            lam*=4
        if not ok or c<1e-26: break
    return wf,c
def werr(W2): return float(((W2[cons]-W2t[cons]).norm(dim=1)/nt[cons]).mean())
# ==== Stage A: refine W2 (b2=true, downstream=true) from real consensus ====
def residW(wf): return ((G(h,wf.reshape(sh),b2t)-Yt)/sc).reshape(-1)
wf,c = gnlm(W2c.reshape(-1).clone(), residW, iters=50)
print(f"\nSTAGE A  refine W2 (b2=true, downstream=true):  W rel-err {werr(W2c):.3f} -> {werr(wf.reshape(sh)):.4f}   loss {c:.2e}")
# ==== Stage B: bias basin.  W2=consensus fixed init, b2=true+delta ====
print("\nSTAGE B  bias basin (W2 init=consensus, downstream=true):")
print(f"{'delta':>7} {'refine W2 only (b2 frozen wrong)':>34} {'refine (W2,b2) jointly':>26}")
gd=torch.Generator(device=dev).manual_seed(3); dvec=torch.randn(80,generator=gd); dvec=dvec/dvec.norm()*np.sqrt(80)
for delta in [0.0,0.01,0.03,0.05,0.1,0.2,0.5]:
    b2w=b2t+delta*dvec
    def residWo(wf): return ((G(h,wf.reshape(sh),b2w)-Yt)/sc).reshape(-1)
    wfo,co=gnlm(W2c.reshape(-1).clone(),residWo,iters=40); eWo=werr(wfo.reshape(sh))
    def residJ(p): W2=p[:80*128].reshape(sh); b2=p[80*128:]; return ((G(h,W2,b2)-Yt)/sc).reshape(-1)
    p0=torch.cat([W2c.reshape(-1),b2w]); pf,cj=gnlm(p0,residJ,iters=40)
    W2j=pf[:80*128].reshape(sh); b2j=pf[80*128:]; eWj=werr(W2j); ebj=float((b2j[cons]-b2t[cons]).abs().mean())
    print(f"{delta:>7.2f} |  Wonly loss {co:.1e} Werr {eWo:.3f}      |  joint loss {cj:.1e} Werr {eWj:.3f} berr {ebj:.3f}")
