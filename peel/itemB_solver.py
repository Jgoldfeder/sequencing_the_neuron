"""ITEM B solver: rank-1-subspace recovery (NON-iterative in the hard part).
Germ Hessian off-diagonals across many probes span an 8-dim subspace of the 15-dim
off-diagonal space. Its rank-1 elements are {offdiag(w_r w_r^T)}. Find each w_r by
minimizing a 6-DIM objective ||(I-P)offdiag(w w^T)||^2 (many cheap restarts) -> 8
directions. Then per-probe linear-solve coeffs d_r, reconstruct H_ii=sum d_r w_ri^2,
and 1-2 s_i=(F_ii - H_ii sdot_i^2)/F_i. Root/branch by approximate bias only."""
import numpy as np, itertools, torch
from scipy.optimize import minimize
torch.set_default_dtype(torch.float64)
k,W=6,8; P=20
g_=torch.Generator().manual_seed(3)
W2=(torch.randn(W,k,generator=g_))*0.9; b2=(2*torch.rand(W,generator=g_)-1)*0.6
w3=(torch.randn(W,generator=g_))*0.9; b3=(2*torch.rand(1,generator=g_)-1)*0.3
b1=(2*torch.rand(k,generator=g_)-1)*0.5
def Ffun(t,z0):
    s=torch.sigmoid(z0+t); q=torch.sigmoid(W2@s+b2); return (w3@q+b3).squeeze()
gp=torch.Generator().manual_seed(8); z0s=[(2*torch.rand(k,generator=gp)-1)*1.4+b1 for _ in range(P)]
oi=[(i,j) for i in range(k) for j in range(i+1,k)]                 # 15 off-diag pairs
def jet(z0):
    t=torch.zeros(k,requires_grad=True); f=Ffun(t,z0); g1=torch.autograd.grad(f,t,create_graph=True)[0]
    H=torch.zeros(k,k)
    for i in range(k): H[i]=torch.autograd.grad(g1[i],t,retain_graph=True)[0]
    return g1.detach().numpy(), H.detach().numpy()
data=[dict(zip(['g','H','z0'],(*jet(z0),z0.numpy()))) for z0 in z0s]
def offvec(A): return np.array([A[i,j] for (i,j) in oi])
def rank1_off(w): return np.array([w[i]*w[j] for (i,j) in oi])

def solve(s_list):
    # (1) subspace from germ-Hessian off-diagonals
    Hoff=[]
    for p in range(P):
        sd=s_list[p]*(1-s_list[p]); A=data[p]['H'].copy()
        Hoff.append(np.array([A[i,j]/(sd[i]*sd[j]) for (i,j) in oi]))
    Hoff=np.array(Hoff)                                            # (P,15)
    U,S,Vt=np.linalg.svd(Hoff,full_matrices=False); r=int((S>S.max()*1e-9).sum())
    basis=Vt[:r]                                                   # (r,15)
    Pperp=np.eye(15)-basis.T@basis
    def f(v):
        w=v/np.linalg.norm(v); o=rank1_off(w); return float(o@Pperp@o)
    # (2) find rank-1 directions by many 6-dim restarts
    found=[]
    rng=np.random.default_rng(0)
    for _ in range(120):
        v0=rng.standard_normal(k); res=minimize(f,v0,method='BFGS',options=dict(maxiter=300,gtol=1e-14))
        if res.fun<1e-12:
            w=res.x/np.linalg.norm(res.x)
            if not any(abs(w@u)>0.999 for u in found): found.append(w)
    Wr=np.array(found)                                             # (nfound,k)
    # (3) per-probe: linear-solve coeffs d_r from off-diagonals, reconstruct diagonal, biases
    G=np.array([rank1_off(w) for w in Wr]).T                       # (15, nfound)
    b_rec=np.zeros((P,k))
    for p in range(P):
        sd=s_list[p]*(1-s_list[p]); Hoff_p=np.array([data[p]['H'][i,j]/(sd[i]*sd[j]) for (i,j) in oi])
        d,_,_,_=np.linalg.lstsq(G,Hoff_p,rcond=None)               # H_off = sum d_r rank1_off(w_r)
        Hii=np.array([ (d*Wr[:,i]**2).sum() for i in range(k)])    # reconstruct diagonal
        mu=(np.diag(data[p]['H']) - Hii*sd**2)/data[p]['g']        # 1-2 s_i
        s_new=np.clip((1-mu)/2,1e-6,1-1e-6)
        b_rec[p]=np.log(s_new/(1-s_new)) - (data[p]['z0']-b1.numpy())
    return len(found), b_rec

# (1) mechanism: true-s conversion
s_true=[torch.sigmoid(torch.tensor(d['z0'])).numpy() for d in data]
nf,b_rec=solve(s_true); err=np.abs(b_rec-b1.numpy()[None,:])
print(f'[mechanism true-s] rank-1 elements found={nf} (expect {W})')
print(f'  bias err: max={err.max():.2e} median={np.median(err):.2e}  cross-probe std max={b_rec.std(0).max():.2e}')
# (2) non-oracle: guess biases, iterate the conversion
rng=np.random.default_rng(1); b_est=b1.numpy()+0.05*(2*rng.random(k)-1)
for it in range(6):
    s_cur=[torch.sigmoid(torch.tensor(d['z0']-b1.numpy()+b_est)).numpy() for d in data]
    nf,b_rec=solve(s_cur); b_est=b_rec.mean(0)
err2=np.abs(b_est-b1.numpy())
print(f'[non-oracle guess+iterate] found={nf}, final bias err max={err2.max():.2e} median={np.median(err2):.2e} (guess ~5e-2)')
