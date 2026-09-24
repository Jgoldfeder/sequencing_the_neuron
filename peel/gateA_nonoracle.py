"""GATE A (fully non-oracle measurement gate). Recover W1_hat by the REAL pipeline
(saturation+SVD directions, dual-tail magnitudes) from a realistic guess. Build the
dual basis V_hat FROM W1_hat (never true W1). Make every black-box jet query via V_hat.
Recompute A_eff in these approximate coordinates (model uses W1_hat as fixed 1st layer,
biases+downstream as theta). Implied bias error = A_eff^+ (I-BB^+) (J_meas - J_model)
folds in BOTH measurement noise AND the W1_hat!=W1 mismatch. Report ||W1 V_hat - I||_max."""
import sys, math, itertools
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch, numpy as np
from numpy.polynomial import chebyshev as C
from torch.func import jacrev, jacfwd
from nets import MLP
torch.set_default_dtype(torch.float64); dev="cpu"
dims=[128,24,32,16,8]; d,k=dims[0],dims[1]; O=dims[-1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True); g=torch.Generator().manual_seed(1)
@torch.no_grad()
def J_at(x,fd=5e-5):
    E=torch.eye(d,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()
# ---- realistic guess: 5deg direction, 8% magnitude, bias off ----
Wg=torch.empty(k,d,dtype=torch.float64); bg=torch.empty(k,dtype=torch.float64)
for j in range(k):
    u=tn[j]; v=torch.randn(d,generator=g,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
    Wg[j]=(math.cos(math.radians(5))*u+math.sin(math.radians(5))*v)*W1t[j].norm()*(1+0.08*(2*torch.rand(1,generator=g)-1))
    bg[j]=b1t[j]+0.08*(2*torch.rand(1,generator=g).item()-1)
Wgpinv=Wg.t()@torch.linalg.inv(Wg@Wg.t())
# ---- directions via saturation + SVD ----
N=torch.empty_like(Wg)
for j in range(k):
    t=torch.full((k,),20.0,dtype=torch.float64); t[j]=0.0
    x0=Wgpinv@(t-bg); U,Sv,Vh=torch.linalg.svd(J_at(x0),full_matrices=False)
    nv=Vh[0]; N[j]=nv if float(nv@Wg[j])>0 else -nv
Vn=N.t()@torch.linalg.inv(N@N.t()); W1rp=(N*Wg.norm(dim=1,keepdim=True)).t()@torch.linalg.inv((N*Wg.norm(dim=1,keepdim=True))@(N*Wg.norm(dim=1,keepdim=True)).t())
# ---- magnitudes via dual-tail multi-harmonic ----
P=6; nctx=8; ncand=40; ell=np.arange(P+1)[:,None]
def fit_mag(tails,a_g):
    def res(a):
        tot=0.0
        for ts,gs,side in tails:
            A=np.exp(-side*a*ell*ts[None,:]).T; coef,_,_,_=np.linalg.lstsq(A,gs,rcond=None); tot+=float(((A@coef-gs)**2).sum())
        return tot
    lo,hi=0.85*a_g,1.15*a_g
    for _ in range(70):
        m1=hi-(hi-lo)*0.618; m2=lo+(hi-lo)*0.618
        if res(m1)<res(m2): hi=m2
        else: lo=m1
    return 0.5*(lo+hi)
a_hat=torch.empty(k,dtype=torch.float64)
for j in range(k):
    a_g=float(Wg[j].norm()); vj=Vn[:,j]
    TT=(2*torch.rand(ncand,k,generator=g,dtype=torch.float64)-1)*2.0; TT[:,j]=0.0
    X0=(TT-bg)@W1rp.t()
    with torch.no_grad(): swing=(teacher(X0+(6.0/a_g)*vj)-teacher(X0-(6.0/a_g)*vj)).norm(dim=1)
    top=torch.topk(swing,nctx).indices
    tp=torch.linspace(2.5/a_g,7.0/a_g,40,dtype=torch.float64); tm=torch.linspace(-7.0/a_g,-2.5/a_g,40,dtype=torch.float64)
    tails=[]
    for mi in top.tolist():
        x0=X0[mi]
        with torch.no_grad():
            Fp=teacher(x0.unsqueeze(0)+tp.unsqueeze(1)*vj.unsqueeze(0)).cpu().numpy()
            Fm=teacher(x0.unsqueeze(0)+tm.unsqueeze(1)*vj.unsqueeze(0)).cpu().numpy()
        for r in range(O): tails.append((tp.cpu().numpy(),Fp[:,r],+1)); tails.append((tm.cpu().numpy(),Fm[:,r],-1))
    a_hat[j]=fit_mag(tails,a_g)
What1=N*a_hat[:,None]                                  # RECOVERED first layer
ang=torch.acos((N*tn).sum(1).clamp(-1,1)).max()*180/math.pi
print(f"recovery: max direction err {float(ang):.2e} deg, max magnitude rel err {float(((a_hat-W1t.norm(dim=1))/W1t.norm(dim=1)).abs().max()):.2e}")
# ---- dual basis from RECOVERED W1_hat; leakage ----
Vhat=What1.t()@torch.linalg.inv(What1@What1.t())
leak=(W1t@Vhat-torch.eye(k,dtype=torch.float64)).abs().max()
print(f"||W1 @ Vhat - I||_max (cross-neuron leakage) = {float(leak):.2e}")
# ---- model with W1_hat fixed; theta=(b1, downstream) ----
Ws=[teacher.layers[l].weight.detach() for l in range(1,4)]; bs=[teacher.layers[l].bias.detach() for l in range(1,4)]
Wsh=[w.shape for w in Ws]
theta0=torch.cat([b1t]+[w.reshape(-1) for w in Ws]+[b for b in bs])
def model_fwd(theta,x):
    b1_=theta[:k]; idx=k; Wd=[]
    for sh in Wsh:
        n=sh[0]*sh[1]; Wd.append(theta[idx:idx+n].reshape(sh)); idx+=n
    bd=[]
    for sh in Wsh: bd.append(theta[idx:idx+sh[0]]); idx+=sh[0]
    h=torch.sigmoid(What1@x+b1_)
    for l,(W,bb) in enumerate(zip(Wd,bd)):
        z=W@h+bb; h=torch.sigmoid(z) if l<len(Wd)-1 else z
    return h
x0=torch.zeros(d); gp=torch.Generator().manual_seed(5)
probes=[(2*torch.rand(k,generator=gp)-1)*1.0 for _ in range(3)]
pairs=list(itertools.combinations(range(k),2))
def Fbatch(pb,T):                                     # BLACK-BOX true teacher via Vhat
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
    return np.concatenate([Fi.reshape(-1),Fii.reshape(-1),Fij.reshape(-1)])
def model_jetvec(theta,pb):
    t0=torch.zeros(k)
    def Fs(t): return model_fwd(theta, x0+(pb+t)@Vhat.t())
    G=jacrev(Fs)(t0); H=jacfwd(jacrev(Fs))(t0)
    return torch.cat([G.reshape(-1), torch.stack([H[:,i,i] for i in range(k)]).reshape(-1),
                      torch.stack([H[:,i,j] for (i,j) in pairs]).reshape(-1)])
rows=[]; dJ=[]
for pb in probes:
    Jm=meas_jet(pb)                                   # black-box (true net, Vhat)
    Jmod=model_jetvec(theta0,pb).detach().numpy()     # model at TRUE theta (W1_hat 1st layer)
    Jc=jacrev(lambda th: model_jetvec(th,pb))(theta0).detach().numpy()
    rows.append(Jc); dJ.append(Jm-Jmod)               # includes measurement AND W1_hat mismatch
M=np.concatenate(rows,0); dJ=np.concatenate(dJ)
A=M[:,:k]; B=M[:,k:]; Xb,_,_,_=np.linalg.lstsq(B,A,rcond=None); Aeff=A-B@Xb
r=dJ-B@np.linalg.lstsq(B,dJ,rcond=None)[0]; db=np.linalg.pinv(Aeff)@r
sv=np.linalg.svd(Aeff,compute_uv=False)
print(f"order-2 A_eff (in approx coords): sigma_min={sv.min():.2e} cond={sv.max()/sv.min():.1f}")
print(f"total jet residual (meas + W1_hat mismatch): max|dJ|={np.abs(dJ).max():.2e}")
print(f"IMPLIED bias error (fully non-oracle): max|db|={np.abs(db).max():.2e} median={np.median(np.abs(db)):.2e}")
print(f"  -> {'PASS' if np.abs(db).max()<1e-4 else 'FAIL'} (<1e-4)")
