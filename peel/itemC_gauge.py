"""ITEM C decisive test: gauge-projected A_eff on clean jets (H_off + T_dist), 6->8->4->1.
A = d(clean jet)/dbeta (6), B = d(clean jet)/d(downstream) (97),
C = per-probe infinitesimal diagonal-gauge tangents: delta F_ij=(eps_i+eps_j)F_ij,
    delta F_ijl=(eps_i+eps_j+eps_l)F_ijl  (P*k columns, block-diagonal by probe).
A_eff       = (I-BB^+)A          [known: identifies b]
A_eff_gauge = (I-P_[B,C])A       [does bias info SURVIVE quotienting the diagonal warp?]
sigma_min(A_eff_gauge)~0 => clean jets identify b only via the gauge (need repeated-index).
sigma_min(A_eff_gauge)>0 => genuine cross-probe architecture info -> spectral invariant justified."""
import torch, itertools, numpy as np
from torch.func import jacrev, jacfwd
torch.set_default_dtype(torch.float64)
k,W,m=6,8,4; P=16
g=torch.Generator().manual_seed(3)
W2=torch.randn(W,k,generator=g)*0.9; b2=(2*torch.rand(W,generator=g)-1)*0.6
W3=torch.randn(m,W,generator=g)*0.7; b3=(2*torch.rand(m,generator=g)-1)*0.5
w4=torch.randn(m,generator=g)*0.8;  b4=(2*torch.rand(1,generator=g)-1)*0.3
beta=(2*torch.rand(k,generator=g)-1)*0.5
theta0=torch.cat([beta,W2.reshape(-1),b2,W3.reshape(-1),b3,w4,b4]); nb=k
def unpack(th):
    i=k; W2_=th[i:i+W*k].reshape(W,k); i+=W*k; b2_=th[i:i+W]; i+=W
    W3_=th[i:i+m*W].reshape(m,W); i+=m*W; b3_=th[i:i+m]; i+=m
    w4_=th[i:i+m]; i+=m; b4_=th[i:i+1]; return W2_,b2_,W3_,b3_,w4_,b4_
gp=torch.Generator().manual_seed(11); cs=[(2*torch.rand(k,generator=gp)-1)*1.3 for _ in range(P)]
hoff=[(i,j) for i in range(k) for j in range(i+1,k)]
tdist=[(i,j,l) for i in range(k) for j in range(i+1,k) for l in range(j+1,k)]
def jetclean(th,c):
    beta_=th[:k]; W2_,b2_,W3_,b3_,w4_,b4_=unpack(th)
    def Ft(t):
        s=torch.sigmoid(c+beta_+t); q=torch.sigmoid(W2_@s+b2_); h=torch.sigmoid(W3_@q+b3_); return (w4_@h+b4_).squeeze()
    G2=jacfwd(jacfwd(Ft))(torch.zeros(k)); G3=jacfwd(jacfwd(jacfwd(Ft)))(torch.zeros(k))
    return torch.cat([torch.stack([G2[i,j] for (i,j) in hoff]),torch.stack([G3[i,j,l] for (i,j,l) in tdist])])
D=len(hoff)+len(tdist)
Jvals=[]; Jacs=[]
for c in cs:
    Jvals.append(jetclean(theta0,c).detach().numpy())
    Jacs.append(jacrev(lambda th: jetclean(th,c))(theta0).detach().numpy())
Jvals=np.array(Jvals); Jac=np.stack(Jacs)          # (P,D),(P,D,ntheta)
A=Jac[:,:,:nb].reshape(P*D,nb); B=Jac[:,:,nb:].reshape(P*D,Jac.shape[2]-nb)
# gauge tangents C: (P*D, P*k)
C=np.zeros((P*D, P*k))
for p in range(P):
    for i in range(k):
        col=np.zeros(D)
        for idx,(a,b) in enumerate(hoff):
            if i in (a,b): col[idx]=Jvals[p,idx]
        for idx,(a,b,l) in enumerate(tdist):
            if i in (a,b,l): col[len(hoff)+idx]=Jvals[p,len(hoff)+idx]
        C[p*D:(p+1)*D, p*k+i]=col
def sigmin(A,Nuis):
    X,_,_,_=np.linalg.lstsq(Nuis,A,rcond=None); Ae=A-Nuis@X
    sv=np.linalg.svd(Ae,compute_uv=False); return sv.min(),sv.max()/sv.min(),int((sv>sv.max()*1e-9).sum())
s0,c0,r0=sigmin(A,B)
BC=np.concatenate([B,C],axis=1)
s1,c1,r1=sigmin(A,BC)
print(f"rows={P*D}  B cols={B.shape[1]}  gauge C cols={C.shape[1]}")
print(f"A_eff        (project out downstream only):   sigma_min={s0:.3e} cond={c0:.1f} rank={r0}/{nb}")
print(f"A_eff_gauge  (project out downstream + gauge): sigma_min={s1:.3e} cond={c1:.1f} rank={r1}/{nb}")
print(f"  ratio sigma_min(gauge)/sigma_min(plain) = {s1/s0:.3e}")
print(f"  VERDICT: {'gauge absorbs the signal -> need repeated-index/recurrence' if s1<1e-9 else 'bias info SURVIVES gauge -> genuine cross-probe architecture invariant exists'}")
