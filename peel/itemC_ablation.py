"""ITEM C diagnostic: jet-subset A_eff ablation on 6->8->4->1.
Target = 6 first-layer biases beta; nuisance = ALL downstream (W2,b2,W3,b3,w4,b4).
For each subset of jet coordinates, A=d(subset)/dbeta, B=d(subset)/dnuisance,
A_eff=(I-BB^+)A. Find the SMALLEST subset that gives rank 6 with good conditioning.
Key question: do the CLEAN entries (H_off + T_distinct, no bias-contaminated
repeated indices) already identify beta? -> then infer nuisance first, use H_ii last
(item-B-like). If it needs T_repeated -> the sigmoid recurrence is essential."""
import torch, itertools, numpy as np
from torch.func import jacrev, jacfwd
torch.set_default_dtype(torch.float64)
k,W,m=6,8,4; P=12
g=torch.Generator().manual_seed(3)
def R(): return torch.randn if True else None
W2=torch.randn(W,k,generator=g)*0.9; b2=(2*torch.rand(W,generator=g)-1)*0.6
W3=torch.randn(m,W,generator=g)*0.7; b3=(2*torch.rand(m,generator=g)-1)*0.5
w4=torch.randn(m,generator=g)*0.8;  b4=(2*torch.rand(1,generator=g)-1)*0.3
beta=(2*torch.rand(k,generator=g)-1)*0.5
theta0=torch.cat([beta,W2.reshape(-1),b2,W3.reshape(-1),b3,w4,b4])
sl={}   # param slices
i=0
for nm,t in [('beta',beta),('W2',W2),('b2',b2),('W3',W3),('b3',b3),('w4',w4),('b4',b4)]:
    n=t.numel(); sl[nm]=slice(i,i+n); i+=n
def unpack(th):
    return (th[sl['W2']].reshape(W,k), th[sl['b2']], th[sl['W3']].reshape(m,W),
            th[sl['b3']], th[sl['w4']], th[sl['b4']])
gp=torch.Generator().manual_seed(11); cs=[(2*torch.rand(k,generator=gp)-1)*1.3 for _ in range(P)]
# typed index lists over k
o1=[(i,) for i in range(k)]
hoff=[(i,j) for i in range(k) for j in range(i+1,k)]
hdiag=[(i,i) for i in range(k)]
tdist=[(i,j,l) for i in range(k) for j in range(i+1,k) for l in range(j+1,k)]
tiij=[(i,i,j) for i in range(k) for j in range(k) if j!=i]
tiii=[(i,i,i) for i in range(k)]
def jetvec(th, c):
    beta_=th[sl['beta']]; W2_,b2_,W3_,b3_,w4_,b4_=unpack(th)
    def Ft(t):
        s=torch.sigmoid(c+beta_+t); q=torch.sigmoid(W2_@s+b2_); h=torch.sigmoid(W3_@q+b3_)
        return (w4_@h+b4_).squeeze()
    G1=jacfwd(Ft)(torch.zeros(k))
    G2=jacfwd(jacfwd(Ft))(torch.zeros(k))
    G3=jacfwd(jacfwd(jacfwd(Ft)))(torch.zeros(k))
    parts=[torch.stack([G1[i] for (i,) in o1]),
           torch.stack([G2[i,j] for (i,j) in hoff]),
           torch.stack([G2[i,i] for (i,ii) in hdiag]),
           torch.stack([G3[i,j,l] for (i,j,l) in tdist]),
           torch.stack([G3[i,i,j] for (i,ii,j) in tiij]),
           torch.stack([G3[i,i,i] for (i,ii,iii) in tiii])]
    return torch.cat(parts)
# per-probe Jacobian d jetvec / d theta
Ms=[jacrev(lambda th: jetvec(th,c))(theta0).detach().numpy() for c in cs]
M=np.stack(Ms)   # (P, 83, 103)
# component-type slices within the 83
lens=[len(o1),len(hoff),len(hdiag),len(tdist),len(tiij),len(tiii)]
names=['o1','Hoff','Hdiag','Tdist','Tiij','Tiii']; off=np.cumsum([0]+lens)
idx={names[i]:list(range(off[i],off[i+1])) for i in range(len(names))}
def aeff(rowtypes):
    rows=[i for nm in rowtypes for i in idx[nm]]
    Msub=M[:,rows,:].reshape(-1,103); A=Msub[:,:k]; B=Msub[:,k:]
    Xb,_,_,_=np.linalg.lstsq(B,A,rcond=None); Ae=A-B@Xb
    sv=np.linalg.svd(Ae,compute_uv=False); rank=int((sv>sv.max()*1e-9).sum())
    return Msub.shape[0], rank, sv.min(), (sv.max()/sv.min() if sv.min()>0 else np.inf)
print("subset                          rows  rank/6   sigma_min    cond")
for sub in [['Hoff'],['Hdiag'],['Hoff','Hdiag'],['Tdist'],['Hoff','Tdist'],
            ['Hoff','Tiij'],['Hoff','Tdist','Tiij','Tiii'],['Hoff','Hdiag','Tdist'],
            ['Hoff','Hdiag','Tdist','Tiij','Tiii'],['o1','Hoff','Hdiag','Tdist','Tiij','Tiii']]:
    r,rk,smin,cond=aeff(sub)
    print(f"  {'+'.join(sub):30s} {r:4d}   {rk}/6    {smin:.2e}   {cond:.1f}")
