"""SEALED, SELF-CONTAINED black-box L1-weights + b1 solve. Seal = l1_blackbox_verify.py:
solver calls ONLY bb.query(x) on 784-D inputs. Truth loaded only for scoring (sign-aware align).
Recomputes the verified W1 solve inline (no reliance on the stale solved_l1_cache), then solves b1
by isolation + transition-center, then measures whether solving b1 makes true-h synthesis real.
"""
import sys, time, torch, numpy as np
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
torch.set_default_dtype(torch.float64)
dev="cuda"; CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
class BlackBox:
    def __init__(self,net):
        net.eval()
        for p in net.parameters(): p.requires_grad_(False)
        def run(x):
            with torch.no_grad(): return net(x).detach().clone()
        object.__setattr__(self,"_run",run); object.__setattr__(self,"n",0)
    def query(self,x):
        object.__setattr__(self,"n",self.n+(x.shape[0] if x.dim()>1 else 1)); return self._run(x)
    __call__=query
    def __getattr__(self,k): raise AttributeError(f"query-only; '{k}' forbidden")
pop=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False)
merged=torch.load(CD+"mergedbest512_sigmoid__784x128x80x40x32x10__s0_final.pt",map_location=dev,weights_only=False)
dims=pop["dims"]; d,k,O=dims[0],dims[1],dims[-1]
_t=MLP(dims,act="sigmoid").to(dev).double(); _t.load_state_dict(pop["teacher_state"]); _t.eval()
bb=BlackBox(_t); del _t
Wg=merged["state_dict"]["layers.0.weight"].to(dev).double().clone()
bg=merged["state_dict"]["layers.0.bias"].to(dev).double().clone()
# ---------- verified black-box W1 solve (dir SVD + multi-harmonic magnitude) ----------
Wgpinv=Wg.t()@torch.linalg.inv(Wg@Wg.t()); g=torch.Generator(device=dev).manual_seed(1)
def J_at(x,fd=5e-5):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((bb.query(x.unsqueeze(0)+fd*E)-bb.query(x.unsqueeze(0)-fd*E))/(2*fd)).t()
N=torch.empty_like(Wg)
for j in range(k):
    t=torch.full((k,),20.0,device=dev,dtype=torch.float64); t[j]=0.0
    x0=Wgpinv@(t-bg); U,Sv,Vh=torch.linalg.svd(J_at(x0),full_matrices=False)
    nv=Vh[0]; N[j]=nv if float(nv@Wg[j])>0 else -nv
V=N.t()@torch.linalg.inv(N@N.t()); Wdir=N*Wg.norm(dim=1,keepdim=True)
W1rp=Wdir.t()@torch.linalg.inv(Wdir@Wdir.t()); P=6
def fit_mag(tails,a_g):
    ell=np.arange(P+1)[:,None]
    def res(a):
        tot=0.0
        for ts,gs,side in tails:
            A=np.exp(-side*a*ell*ts[None,:]).T; cf,_,_,_=np.linalg.lstsq(A,gs,rcond=None); tot+=float(((A@cf-gs)**2).sum())
        return tot
    lo,hi=0.8*a_g,1.2*a_g
    for _ in range(70):
        m1=hi-(hi-lo)*.618; m2=lo+(hi-lo)*.618
        if res(m1)<res(m2): hi=m2
        else: lo=m1
    return .5*(lo+hi)
a_hat=torch.zeros(k,device=dev,dtype=torch.float64)
for jj in range(k):
    a_g=float(Wg[jj].norm()); vj=V[:,jj]
    TT=(2*torch.rand(40,k,generator=g,device=dev,dtype=torch.float64)-1)*2.0; TT[:,jj]=0.0
    X0=(TT-bg)@W1rp.t()
    sw=(bb.query(X0+(6.0/a_g)*vj)-bb.query(X0-(6.0/a_g)*vj)).norm(dim=1)
    top=torch.topk(sw,8).indices
    tp=torch.linspace(2.5/a_g,7.0/a_g,40,device=dev,dtype=torch.float64)
    tm=torch.linspace(-7.0/a_g,-2.5/a_g,40,device=dev,dtype=torch.float64); tails=[]
    for mi in top.tolist():
        x0=X0[mi]
        Fp=bb.query(x0.unsqueeze(0)+tp.unsqueeze(1)*vj.unsqueeze(0)).cpu().numpy()
        Fm=bb.query(x0.unsqueeze(0)+tm.unsqueeze(1)*vj.unsqueeze(0)).cpu().numpy()
        for r in range(O):
            tails.append((tp.cpu().numpy(),Fp[:,r],+1)); tails.append((tm.cpu().numpy(),Fm[:,r],-1))
    a_hat[jj]=fit_mag(tails,a_g)
W1s=a_hat[:,None]*N; a=a_hat; W1sp=W1s.t()@torch.linalg.inv(W1s@W1s.t()); b1_guess=bg.clone()
# ---------- solve b1 (isolation + transition center) ----------
T=25.0; b1_solved=torch.zeros(k,device=dev)
for j in range(k):
    t=torch.full((k,),T,device=dev); t[j]=0.0
    xb=W1sp@(t-b1_guess); njxb=float(N[j]@xb)
    s_guess=-njxb-float(b1_guess[j]/a[j]); span=10.0/float(a[j])
    S=torch.linspace(s_guess-span,s_guess+span,401,device=dev)
    X=xb.unsqueeze(0)+S.unsqueeze(1)*N[j].unsqueeze(0)
    Oo=bb.query(X); mag=(Oo[2:]-Oo[:-2]).norm(dim=1); ip=int(torch.argmax(mag))+1
    if 1<=ip<len(mag)-1:
        y0,y1,y2=float(mag[ip-1]),float(mag[ip]),float(mag[ip+1]); den=(y0-2*y1+y2)
        delta=0.5*(y0-y2)/den if abs(den)>1e-30 else 0.0
    else: delta=0.0
    s_star=float(S[ip])+delta*float(S[1]-S[0]); b1_solved[j]=-a[j]*(njxb+s_star)
print(f"[solve] W1+b1 done, {bb.n} black-box queries")
# ---------- scoring (sign-aware alignment) ----------
b1t=pop["teacher_state"]["layers.0.bias"].to(dev).double()
W1t=pop["teacher_state"]["layers.0.weight"].to(dev).double(); nt=W1t.norm(dim=1)
Cp=torch.cdist(W1s,W1t); Cm=torch.cdist(-W1s,W1t); C=torch.minimum(Cp,Cm)
ri,ci=linear_sum_assignment(C.cpu().numpy()); ri=torch.tensor(ri,device=dev); ci=torch.tensor(ci,device=dev)
sgn=torch.where(Cm[ri,ci]<Cp[ri,ci],-1.0,1.0)
perm=torch.empty(k,dtype=torch.long,device=dev); perm[ri]=ci
rowerr=(C[ri,ci]/nt[ci])   # sign-aware min distance, like l1_blackbox_verify.aligned()
noflip=(Cp[ri,ci]/nt[ci]); flip=(Cm[ri,ci]/nt[ci])
real_flips=int(((sgn<0)&(noflip>0.5)).sum())   # matched via flip AND no-flip dist is large => genuine flip
print(f"[check] recomputed W1 aligned row rel-err (sign-aware): mean {float(rowerr.mean()):.3e} max {float(rowerr.max()):.3e}")
print(f"[check] sign flips flagged: {int((sgn<0).sum())};  GENUINE flips (no-flip dist>0.5): {real_flips};  no-flip-only mean err: {float(noflip.mean()):.3e} max {float(noflip.max()):.3e}")
# complement-aware "truth in OUR sign convention": sgn folds the flipped neurons correctly
W1t_our=sgn[:,None]*W1t[perm]     # should match W1s
b1t_our=sgn*b1t[perm]            # what b1_solved should equal
rowerr2=((W1s-W1t_our).norm(dim=1)/nt[perm])
print(f"[check] W1 err in our convention (complement-folded): mean {float(rowerr2.mean()):.3e} max {float(rowerr2.max()):.3e}")
def stat(v): return f"mean {float(v.mean()):.3e}  max {float(v.max()):.3e}"
print(f"\ntrue b1: mean|b1*| {float(b1t.abs().mean()):.3e}  max {float(b1t.abs().max()):.3e}  (a_j mean {float(a.mean()):.2f})")
print(f"b1 GUESS  abs err: {stat((b1_guess-b1t_our).abs())}   err/a_j: {stat(((b1_guess-b1t_our)/a).abs())}")
print(f"b1 SOLVED abs err: {stat((b1_solved-b1t_our).abs())}   err/a_j: {stat(((b1_solved-b1t_our)/a).abs())}")
# ---------- true-h synthesis fidelity, complement-aware, decomposed ----------
W1tp_our=W1t_our.t()@torch.linalg.inv(W1t_our@W1t_our.t())   # true pinv in our convention (isolates b-only)
gen=torch.Generator(device=dev).manual_seed(0)
def hfid(Wpinv,b, std):
    zt=torch.randn(2000,k,generator=gen,device=dev)*std
    hs=torch.sigmoid(zt).clamp(1e-4,1-1e-4); lg=torch.log(hs/(1-hs))
    x=(lg-b)@Wpinv.t(); h=torch.sigmoid(x@W1t_our.t()+b1t_our); e=(h-hs).abs(); return float(e.mean())
print(f"\ntrue-h synthesis fidelity mean|h_actual-h*|, decomposed (complement-aware):")
print(f"{'target std':>10} {'(W1t,b1t)ctrl':>13} {'(W1t,b1guess)':>13} {'(W1s,b1t)':>11} {'(W1s,b1guess)':>13}")
for std in [0.5,1.0,1.5,2.5]:
    c1=hfid(W1tp_our,b1t_our,std); c2=hfid(W1tp_our,b1_guess,std); c3=hfid(W1sp,b1t_our,std); c4=hfid(W1sp,b1_guess,std)
    print(f"{std:>10.1f} {c1:>13.2e} {c2:>13.2e} {c3:>11.2e} {c4:>13.2e}")
print("cols: control(should~0) | b1-error only | W1-residual only | both(realistic)")
