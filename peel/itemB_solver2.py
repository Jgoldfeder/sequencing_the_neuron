"""ITEM B solver v2: disambiguate rank-1s using BOTH orders (cross-order structure).
Find w with offdiag(w w^T) in the W-dim Hessian subspace AND offdist(w^3) in the W-dim
3rd-order subspace. Intersection = the true W second-layer directions (removes the
spurious rank-1s of the off-diagonal-only problem). Subspace dims fixed to W (known
architecture). Then linear-solve coeffs, reconstruct H_ii, extract biases."""
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
oi=[(i,j) for i in range(k) for j in range(i+1,k)]                     # 15 pairs
ti=[(i,j,l) for i in range(k) for j in range(i+1,k) for l in range(j+1,k)]  # 20 triples
def jet(z0):
    t=torch.zeros(k,requires_grad=True); f=Ffun(t,z0); g1=torch.autograd.grad(f,t,create_graph=True)[0]
    H=torch.zeros(k,k); T=torch.zeros(k,k,k)
    gi=[torch.autograd.grad(g1[i],t,create_graph=True)[0] for i in range(k)]
    for i in range(k):
        H[i]=gi[i].detach()
        for j in range(k): T[i,j]=torch.autograd.grad(gi[i][j],t,retain_graph=True)[0]
    return g1.detach().numpy(), H.detach().numpy(), T.detach().numpy()
data=[dict(zip(['g','H','T','z0'],(*jet(z0),z0.numpy()))) for z0 in z0s]
def r1o(w): return np.array([w[i]*w[j] for (i,j) in oi])
def r1t(w): return np.array([w[i]*w[j]*w[l] for (i,j,l) in ti])
def perp(vecs, dim):
    U,S,Vt=np.linalg.svd(np.array(vecs),full_matrices=False); B=Vt[:dim]; return np.eye(B.shape[1])-B.T@B

def solve(s_list):
    Hoff=[]; Tdist=[]
    for p in range(P):
        sd=s_list[p]*(1-s_list[p]); A=data[p]['H']; Tt=data[p]['T']
        Hoff.append(np.array([A[i,j]/(sd[i]*sd[j]) for (i,j) in oi]))
        Tdist.append(np.array([Tt[i,j,l]/(sd[i]*sd[j]*sd[l]) for (i,j,l) in ti]))
    PpH=perp(Hoff,W); PpT=perp(Tdist,W)
    def f(v):
        w=v/np.linalg.norm(v); o2=r1o(w); o3=r1t(w)
        return float(o2@PpH@o2 + o3@PpT@o3)
    found=[]; rng=np.random.default_rng(0)
    for _ in range(200):
        res=minimize(f,rng.standard_normal(k),method='BFGS',options=dict(maxiter=400,gtol=1e-15))
        if res.fun<1e-12:
            w=res.x/np.linalg.norm(res.x)
            if not any(abs(w@u)>0.999 for u in found): found.append(w)
    Wr=np.array(found); G=np.array([r1o(w) for w in Wr]).T                # (15,nf)
    b_rec=np.zeros((P,k))
    for p in range(P):
        sd=s_list[p]*(1-s_list[p]); Hoff_p=np.array([data[p]['H'][i,j]/(sd[i]*sd[j]) for (i,j) in oi])
        d,_,_,_=np.linalg.lstsq(G,Hoff_p,rcond=None)
        Hii=np.array([(d*Wr[:,i]**2).sum() for i in range(k)])
        mu=(np.diag(data[p]['H'])-Hii*sd**2)/data[p]['g']
        s_new=np.clip((1-mu)/2,1e-6,1-1e-6)
        b_rec[p]=np.log(s_new/(1-s_new))-(data[p]['z0']-b1.numpy())
    return len(found), b_rec
s_true=[torch.sigmoid(torch.tensor(d['z0'])).numpy() for d in data]
nf,b_rec=solve(s_true); err=np.abs(b_rec-b1.numpy()[None,:])
print(f'[mechanism true-s] rank-1 found={nf} (expect {W}); bias err max={err.max():.2e} median={np.median(err):.2e} cross-probe std {b_rec.std(0).max():.2e}')
rng=np.random.default_rng(1); b_est=b1.numpy()+0.05*(2*rng.random(k)-1)
for it in range(6):
    s_cur=[torch.sigmoid(torch.tensor(d['z0']-b1.numpy()+b_est)).numpy() for d in data]
    nf,b_rec=solve(s_cur); b_est=b_rec.mean(0)
err2=np.abs(b_est-b1.numpy())
print(f'[non-oracle guess+iterate] found={nf}, final bias err max={err2.max():.2e} median={np.median(err2):.2e} (guess ~5e-2)')
