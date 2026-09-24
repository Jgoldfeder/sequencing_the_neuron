"""FULL layer-1 refinement via dual-direction exact isolation:
 direction = SVD; magnitude = multi-harmonic tail variable-projection over a;
 bias = q-transform: log|D_l| grows with slope c, b = c - a*(n.x0). Layer-1 guess
 only, no downstream. Reports max/mean/median for magnitude, bias, full weight."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch, numpy as np
from scipy.interpolate import AAA
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]; O=dims[-1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True); g=torch.Generator(device=dev).manual_seed(1)
@torch.no_grad()
def J_at(x,fd=5e-5):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()
Wg=torch.empty(k,d,device=dev,dtype=torch.float64); bg=torch.empty(k,device=dev,dtype=torch.float64)
for j in range(k):
    u=tn[j]; v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
    Wg[j]=(math.cos(math.radians(5))*u+math.sin(math.radians(5))*v)*W1t[j].norm()*(1+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64)-1))
    bg[j]=b1t[j]+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64).item()-1)
Wgpinv=Wg.t()@torch.linalg.inv(Wg@Wg.t())
N=torch.empty_like(Wg)
for j in range(k):
    t=torch.full((k,),20.0,device=dev,dtype=torch.float64); t[j]=0.0
    x0=Wgpinv@(t-bg); U,Sv,Vh=torch.linalg.svd(J_at(x0),full_matrices=False)
    nv=Vh[0]; N[j]=nv if float(nv@Wg[j])>0 else -nv
V=N.t()@torch.linalg.inv(N@N.t())
W1rec=N*Wg.norm(dim=1,keepdim=True); W1rp=W1rec.t()@torch.linalg.inv(W1rec@W1rec.t())

P=6; nctx=8; ncand=40; ell=np.arange(P+1)
def fit_a(tails,a_g):
    def res(a):
        tot=0.0
        for ts,gs,side in tails:
            A=np.exp(-side*a*ell[:,None]*ts[None,:]).T
            coef,_,_,_=np.linalg.lstsq(A,gs,rcond=None); tot+=float(((A@coef-gs)**2).sum())
        return tot
    lo,hi=0.85*a_g,1.15*a_g
    for _ in range(70):
        m1=hi-(hi-lo)*0.618; m2=lo+(hi-lo)*0.618
        if res(m1)<res(m2): hi=m2
        else: lo=m1
    return 0.5*(lo+hi)
from numpy.polynomial.chebyshev import chebfit
def fit_c(trans,a,Q=3.0):                                 # Bernstein: Cheb-coef decay rate -> q*=-R
    cs=[]
    for ts,Fmat in trans:
        q=np.exp(a*ts); m=(q>1e-3)&(q<=Q)
        if m.sum()<25: continue
        x=2*q[m]/Q-1
        for r in range(Fmat.shape[1]):
            try: coef=chebfit(x,Fmat[m,r],30)
            except Exception: continue
            n=np.arange(len(coef)); ac=np.abs(coef); ok=(n>=6)&(n<=22)&(ac>1e-13)
            if ok.sum()<5: continue
            slope=np.polyfit(n[ok],np.log(ac[ok]),1)[0]; rho=math.exp(-slope)
            if rho<=1.0: continue
            xstar=-0.5*(rho+1/rho); R=-Q*(xstar+1)/2       # q*=Q(x*+1)/2=-R
            if 0.6<R<1.5: cs.append(-math.log(R))
    return float(np.median(cs)) if cs else 0.0

merr=[]; berr=[]; werr=[]
for j in range(k):
    a_g=float(Wg[j].norm()); vj=V[:,j]
    TT=(2*torch.rand(ncand,k,generator=g,device=dev,dtype=torch.float64)-1)*2.0; TT[:,j]=0.0
    X0=(TT-bg)@W1rp.t()
    with torch.no_grad(): swing=(teacher(X0+(6/a_g)*vj)-teacher(X0-(6/a_g)*vj)).norm(dim=1)
    top=torch.topk(swing,nctx).indices
    tp=torch.linspace(2.5/a_g,7.0/a_g,40,device=dev,dtype=torch.float64)
    tm=torch.linspace(-7.0/a_g,-2.5/a_g,40,device=dev,dtype=torch.float64)
    ttr=torch.tensor(np.cos(np.pi*(np.arange(90)+0.5)/90)*(5.0/a_g),device=dev,dtype=torch.float64)  # transition
    tails=[]; trans=[]
    for mi in top.tolist():
        x0=X0[mi]
        with torch.no_grad():
            Fp=teacher(x0.unsqueeze(0)+tp.unsqueeze(1)*vj.unsqueeze(0)).cpu().numpy()
            Fm=teacher(x0.unsqueeze(0)+tm.unsqueeze(1)*vj.unsqueeze(0)).cpu().numpy()
            Ft=teacher(x0.unsqueeze(0)+ttr.unsqueeze(1)*vj.unsqueeze(0)).cpu().numpy()
        for r in range(O):
            tails.append((tp.cpu().numpy(),Fp[:,r],+1)); tails.append((tm.cpu().numpy(),Fm[:,r],-1))
        trans.append((ttr.cpu().numpy(),Ft))
    a_hat=fit_a(tails,a_g); c_hat=fit_c(trans,a_hat)
    nx=-float(bg[j])/a_g                                   # n_j . x0 (exact, context-invariant)
    b_hat=c_hat-a_hat*nx
    merr.append(abs(a_hat-float(W1t[j].norm()))); berr.append(abs(b_hat-float(b1t[j])))
    s=1.0 if float(N[j]@tn[j])>0 else -1.0
    werr.append(max(float((s*a_hat*N[j]-W1t[j]).abs().max()), abs(b_hat-float(b1t[j]))))
me=torch.tensor(merr); be=torch.tensor(berr); we=torch.tensor(werr)
print("DUAL-DIRECTION full refinement (layer-1 guess only):")
print(f"  magnitude: max {me.max():.2e} mean {me.mean():.2e} median {me.median():.2e}")
print(f"  bias:      max {be.max():.2e} mean {be.mean():.2e} median {be.median():.2e}")
print(f"  full(w,b): max {we.max():.2e} mean {we.mean():.2e} median {we.median():.2e}   (before: max 8e-2)")
