"""ITEM 2 (measurement gate): can BLACK-BOX queries measure the order-2 mixed jet on
the REAL 128->24->32->16->8 net accurately enough to resolve delta_b ~ 1e-4?
Measure gradient F_i and Hessian F_ij (dual coords) by 1-D Chebyshev fits along axis
(e_i) and diagonal (e_i+e_j) directions; F_ij=(dir2_{i+j}-F_ii-F_jj)/2. Compare to
autodiff truth. Then PROPAGATE the measured jet error through the order-2 identifiability
operator: implied delta_b = pinv(A_eff) (I-BB^+) delta_J.  Target: implied |db| <~ 1e-4."""
import torch, itertools, numpy as np
from numpy.polynomial import chebyshev as C
from torch.func import jacrev, jacfwd
torch.set_default_dtype(torch.float64)

dims=[128,24,32,16,8]; d,k=dims[0],dims[1]
g=torch.Generator().manual_seed(1)
Ws=[(torch.randn(dims[i+1],dims[i],generator=g))/np.sqrt(dims[i]) for i in range(len(dims)-1)]
bs=[(2*torch.rand(dims[i+1],generator=g)-1)*0.6 for i in range(len(dims)-1)]
W1=Ws[0]; b1=bs[0]; V=W1.t()@torch.linalg.inv(W1@W1.t()); x0=torch.zeros(d)
theta0=torch.cat([b1]+[Ws[l].reshape(-1) for l in range(1,len(Ws))]+[bs[l] for l in range(1,len(bs))])
Wsh=[Ws[l].shape for l in range(1,len(Ws))]
def forward(theta,x):
    b1_=theta[:k]; idx=k; Wd=[]
    for sh in Wsh:
        n=sh[0]*sh[1]; Wd.append(theta[idx:idx+n].reshape(sh)); idx+=n
    bd=[]
    for sh in Wsh: bd.append(theta[idx:idx+sh[0]]); idx+=sh[0]
    h=torch.sigmoid(W1@x+b1_)
    for l,(W,bb) in enumerate(zip(Wd,bd)):
        z=W@h+bb; h=torch.sigmoid(z) if l<len(Wd)-1 else z
    return h
gp=torch.Generator().manual_seed(99)
probes=[(2*torch.rand(k,generator=gp)-1)*1.2 for _ in range(3)]

def Fbatch(pb, T):                       # T:(n,k) -> (n,8) black-box teacher evals
    X=x0+ (pb+T)@V.t()
    with torch.no_grad(): return torch.stack([forward(theta0,X[r]) for r in range(X.shape[0])]).numpy()
def dderiv(pb, u, rho=0.7, N=44, deg=14):     # F''and F' along direction u (per output)
    x=np.cos((np.arange(N)+0.5)*np.pi/N)*rho
    T=torch.tensor(np.outer(x,u))
    vals=Fbatch(pb,T)                         # (N,8)
    cf=C.chebfit(x/rho, vals, deg)
    d1=C.chebval(0.0,C.chebder(cf))/rho; d2=C.chebval(0.0,C.chebder(C.chebder(cf)))/rho**2
    return d1,d2                               # (8,),(8,)

pairs=list(itertools.combinations(range(k),2))
def measure_jet(pb):
    Fi=np.zeros((k,8)); Fii=np.zeros((k,8))
    for i in range(k):
        u=np.zeros(k); u[i]=1.0; Fi[i],Fii[i]=dderiv(pb,u)
    Fij=np.zeros((len(pairs),8))
    for p,(i,j) in enumerate(pairs):
        u=np.zeros(k); u[i]=1.0; u[j]=1.0
        _,dir2=dderiv(pb,u); Fij[p]=(dir2-Fii[i]-Fii[j])/2.0
    # stack: order-1 (k*8) then order-2 unique (i<=j). diag first then pairs, per output
    return Fi, Fii, Fij

def true_jet(pb):
    t0=torch.zeros(k)
    def Fs(t): return forward(theta0, x0+(pb+t)@V.t())
    G=jacrev(Fs)(t0)                          # (8,k)
    H=jacfwd(jacrev(Fs))(t0)                  # (8,k,k)
    Fi=G.t().numpy(); Fii=np.stack([H[:,i,i].numpy() for i in range(k)])
    Fij=np.stack([H[:,i,j].numpy() for (i,j) in pairs])
    return Fi,Fii,Fij

# ---- measure vs truth ----
errs=[]
allmeas=[]; alltrue=[]
for pb in probes:
    Fi,Fii,Fij=measure_jet(pb); tFi,tFii,tFij=true_jet(pb)
    e=max(np.abs(Fi-tFi).max(),np.abs(Fii-tFii).max(),np.abs(Fij-tFij).max()); errs.append(e)
    allmeas.append((Fi,Fii,Fij)); alltrue.append((tFi,tFii,tFij))
print(f"black-box order-2 jet vs autodiff: max abs err over 3 probes = {max(errs):.2e}")
print(f"  (per-probe max err: {', '.join(f'{e:.1e}' for e in errs)})")

# ---- build A,B (order-2 identifiability) and propagate the measured error ----
def jetvec_fn(theta, pb):
    t0=torch.zeros(k)
    def Fs(t): return forward(theta, x0+(pb+t)@V.t())
    G=jacrev(Fs)(t0); H=jacfwd(jacrev(Fs))(t0)
    parts=[G.reshape(-1)]
    parts.append(torch.stack([H[:,i,i] for i in range(k)]).reshape(-1))
    parts.append(torch.stack([H[:,i,j] for (i,j) in pairs]).reshape(-1))
    return torch.cat(parts)
rows=[]; dJ=[]
for pb,(Fi,Fii,Fij),(tFi,tFii,tFij) in zip(probes,allmeas,alltrue):
    Jc=jacrev(lambda th: jetvec_fn(th,pb))(theta0).detach().numpy()
    rows.append(Jc)
    dJ.append(np.concatenate([(Fi-tFi).reshape(-1),(Fii-tFii).reshape(-1),(Fij-tFij).reshape(-1)]))
M=np.concatenate(rows,0); dJ=np.concatenate(dJ)
A=M[:,:k]; B=M[:,k:]
Xb,_,_,_=np.linalg.lstsq(B,A,rcond=None); Aeff=A-B@Xb
# implied bias error from measured jet error: db = pinv(Aeff) (I-BB^+) dJ
r=dJ - B@np.linalg.lstsq(B,dJ,rcond=None)[0]
db=np.linalg.pinv(Aeff)@r
sv=np.linalg.svd(Aeff,compute_uv=False)
print(f"order-2 A_eff: sigma_min={sv.min():.2e} cond={sv.max()/sv.min():.1f}")
print(f"IMPLIED bias error from black-box jet measurement: max|db|={np.abs(db).max():.2e}  median={np.median(np.abs(db)):.2e}")
print(f"  -> measurement {'IS' if np.abs(db).max()<1e-4 else 'is NOT'} accurate enough for 1e-4 biases")
