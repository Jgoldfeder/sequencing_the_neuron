"""Gate A, ORDERING FIXED. meas_jet and model jet both use i-major [i,o] gradient.
Diagnostic: model-at-truth must match measured jet to ~machine (else bug remains).
Test BOTH: (1) bias-only (first layer fixed at W1_hat) and (2) joint (S,beta) refine."""
import sys, math, itertools
sys.path.insert(0,'/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code')
import torch, numpy as np
from numpy.polynomial import chebyshev as C
from torch.func import jacrev, jacfwd
from nets import MLP
torch.set_default_dtype(torch.float64)
dims=[128,24,32,16,8]; d,k=dims[0],dims[1]; O=dims[-1]
teacher=MLP(dims,act='sigmoid').double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True); g=torch.Generator().manual_seed(1)
@torch.no_grad()
def J_at(x,fd=5e-5):
    E=torch.eye(d,dtype=torch.float64); return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()
Wg=torch.empty(k,d); bg=torch.empty(k)
for j in range(k):
    u=tn[j]; v=torch.randn(d,generator=g); v=v-(v@u)*u; v=v/v.norm()
    Wg[j]=(math.cos(math.radians(5))*u+math.sin(math.radians(5))*v)*W1t[j].norm()*(1+0.08*(2*torch.rand(1,generator=g)-1))
    bg[j]=b1t[j]+0.08*(2*torch.rand(1,generator=g).item()-1)
Wgpinv=Wg.t()@torch.linalg.inv(Wg@Wg.t()); N=torch.empty_like(Wg)
for j in range(k):
    t=torch.full((k,),20.0); t[j]=0.0; x0=Wgpinv@(t-bg)
    U,Sv,Vh=torch.linalg.svd(J_at(x0),full_matrices=False); nv=Vh[0]; N[j]=nv if float(nv@Wg[j])>0 else -nv
mag_err=8e-5*(2*torch.rand(k,generator=g)-1); a_hat=W1t.norm(dim=1)*(1+mag_err)
What1=N*a_hat[:,None]; Vhat=What1.t()@torch.linalg.inv(What1@What1.t())
print(f'leakage ||W1 Vhat-I||_max={float((W1t@Vhat-torch.eye(k)).abs().max()):.2e}')
Ws=[teacher.layers[l].weight.detach() for l in range(1,4)]; bs=[teacher.layers[l].bias.detach() for l in range(1,4)]
Wsh=[w.shape for w in Ws]; th_ds=torch.cat([w.reshape(-1) for w in Ws]+[b for b in bs])
S_true=(W1t@Vhat)
x0=torch.zeros(d); gp=torch.Generator().manual_seed(5); probes=[(2*torch.rand(k,generator=gp)-1)*1.0 for _ in range(3)]
pairs=list(itertools.combinations(range(k),2))
def Fbatch(pb,T):
    X=x0+(pb+T)@Vhat.t()
    with torch.no_grad(): return torch.stack([teacher(X[r]) for r in range(X.shape[0])]).numpy()
def dderiv(pb,u,rho=0.7,Nn=44,deg=14):
    x=np.cos((np.arange(Nn)+0.5)*np.pi/Nn)*rho; vals=Fbatch(pb,torch.tensor(np.outer(x,u)))
    cf=C.chebfit(x/rho,vals,deg); return C.chebval(0.0,C.chebder(cf))/rho, C.chebval(0.0,C.chebder(C.chebder(cf)))/rho**2
def meas_jet(pb):
    Fi=np.zeros((k,O)); Fii=np.zeros((k,O))
    for i in range(k):
        u=np.zeros(k); u[i]=1.0; Fi[i],Fii[i]=dderiv(pb,u)
    Fij=np.zeros((len(pairs),O))
    for p,(i,j) in enumerate(pairs):
        u=np.zeros(k); u[i]=1.0; u[j]=1.0; _,d2=dderiv(pb,u); Fij[p]=(d2-Fii[i]-Fii[j])/2
    return np.concatenate([Fi.reshape(-1),Fii.reshape(-1),Fij.reshape(-1)])       # i-major
def jetvec(fwd,pb):
    def Fs(t): return fwd(pb+t)
    t0=torch.zeros(k); G=jacrev(Fs)(t0); H=jacfwd(jacrev(Fs))(t0)
    return torch.cat([G.t().reshape(-1),                                          # FIX: [i,o]
                      torch.stack([H[:,i,i] for i in range(k)]).reshape(-1),
                      torch.stack([H[:,i,j] for (i,j) in pairs]).reshape(-1)])
def downstream(h, theta_ds):
    idx=0; Wd=[]
    for sh in Wsh:
        n=sh[0]*sh[1]; Wd.append(theta_ds[idx:idx+n].reshape(sh)); idx+=n
    bd=[]
    for sh in Wsh: bd.append(theta_ds[idx:idx+sh[0]]); idx+=sh[0]
    for l,(W,bb) in enumerate(zip(Wd,bd)):
        z=W@h+bb; h=torch.sigmoid(z) if l<len(Wd)-1 else z
    return h
# ---------- (1) bias-only: first layer fixed at W1_hat (S=I in probe coords) ----------
def mk_bias(theta):    # theta=[b1(k), ds]
    def fwd(tin): return downstream(torch.sigmoid(tin+theta[:k]), theta[k:])   # What1@Vhat=I -> S=I
    return fwd
th_bias=torch.cat([b1t,th_ds])
# ---------- (2) joint (S,beta) ----------
def mk_joint(theta):   # theta=[vec(S)(k*k), beta(k), ds]
    S=theta[:k*k].reshape(k,k); beta=theta[k*k:k*k+k]
    def fwd(tin): return downstream(torch.sigmoid(S@tin+beta), theta[k*k+k:])
    return fwd
th_joint=torch.cat([S_true.reshape(-1),b1t,th_ds])
def run(name, theta, mk, ntarget, betaslice):
    rows=[]; dJ=[]
    for pb in probes:
        Jm=meas_jet(pb); Jmod=jetvec(mk(theta),pb).detach().numpy()
        Jc=jacrev(lambda th: jetvec(mk(th),pb))(theta).detach().numpy(); rows.append(Jc); dJ.append(Jm-Jmod)
    M=np.concatenate(rows,0); dJ=np.concatenate(dJ)
    A=M[:,:ntarget]; B=M[:,ntarget:]; Xb,_,_,_=np.linalg.lstsq(B,A,rcond=None); Aeff=A-B@Xb
    sv=np.linalg.svd(Aeff,compute_uv=False)
    r=dJ-B@np.linalg.lstsq(B,dJ,rcond=None)[0]; dtar=np.linalg.pinv(Aeff)@r; dbeta=dtar[betaslice]
    print(f'[{name}] model-at-truth vs measured: max|dJ|={np.abs(dJ).max():.2e}  (should be ~machine if model=truth)')
    print(f'   A_eff sigma_min={sv.min():.2e} cond={sv.max()/sv.min():.1f} rank={int((sv>sv.max()*1e-9).sum())}/{ntarget}')
    verdict="PASS" if np.abs(dbeta).max()<1e-4 else "FAIL"
    print(f'   IMPLIED bias err: max={np.abs(dbeta).max():.2e} median={np.median(np.abs(dbeta)):.2e} -> {verdict}')
run('bias-only', th_bias, mk_bias, k, slice(0,k))
run('joint(S,beta)', th_joint, mk_joint, k*k+k, slice(k*k,k*k+k))
