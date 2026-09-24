"""ITEM B solver v3 (working design). Break the circularity with the GUESS:
- approximate germ diagonal H_ii ~ (F_ii - g_i sddot_i)/sdot_i^2 using guessed biases,
  off-diagonals exact -> full-matrix subspace (rank W) has CLEAN 8-rank-1 locus.
- recover the W directions w_r (unit) by minimizing ||(I-P)vech(w w^T)||^2 (6-dim, cheap).
- reconstruct EXACT diagonal from BIAS-FREE off-diagonals: d_r=lstsq, H_ii=sum d_r w_ri^2.
- biases: (F_ii - H_ii sdot_i^2)/F_i = 1-2 s_i.  Iterate. Roots chosen by approx bias."""
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
def jet(z0):
    t=torch.zeros(k,requires_grad=True); f=Ffun(t,z0); g1=torch.autograd.grad(f,t,create_graph=True)[0]
    H=torch.zeros(k,k)
    for i in range(k): H[i]=torch.autograd.grad(g1[i],t,retain_graph=True)[0]
    return g1.detach().numpy(), H.detach().numpy()
data=[dict(zip(['g','H','z0'],(*jet(z0),z0.numpy()))) for z0 in z0s]
iu=np.triu_indices(k); sc=np.where(iu[0]==iu[1],1.0,np.sqrt(2))
def vech(A): return A[iu]*sc
oi=[(i,j) for i in range(k) for j in range(i+1,k)]
def r1o(w): return np.array([w[i]*w[j] for (i,j) in oi])

def solve(s_list):
    # approximate full germ Hessians (guess diagonal, exact off-diagonal)
    Ms=[]
    for p in range(P):
        s=s_list[p]; sd=s*(1-s); sdd=sd*(1-2*s); Hj=data[p]['H']; g=data[p]['g']; H=np.zeros((k,k))
        for i in range(k):
            for j in range(k):
                if i!=j: H[i,j]=Hj[i,j]/(sd[i]*sd[j])
            H[i,i]=(Hj[i,i]-(g[i]/sd[i])*sdd[i])/sd[i]**2      # g_germ=F_i/sdot
        Ms.append(vech(H))
    Uu,S,Vt=np.linalg.svd(np.array(Ms),full_matrices=False); B=Vt[:W]; Pp=np.eye(len(iu[0]))-B.T@B
    def f(v):
        w=v/np.linalg.norm(v); o=vech(np.outer(w,w)); return float(o@Pp@o)
    cands=[]; rng=np.random.default_rng(0)
    for _ in range(200):
        res=minimize(f,rng.standard_normal(k),method='BFGS',options=dict(maxiter=400,gtol=1e-15))
        cands.append((res.fun, res.x/np.linalg.norm(res.x)))
    clusters=[]                                                    # take W lowest-f DISTINCT dirs
    for fv,w in sorted(cands,key=lambda z:z[0]):
        if not any(abs(w@c)>0.99 for c in clusters): clusters.append(w)
        if len(clusters)>=W: break
    Wr=np.array(clusters); G=np.array([r1o(w) for w in Wr]).T
    b_rec=np.zeros((P,k))
    for p in range(P):
        s=s_list[p]; sd=s*(1-s); Hoff=np.array([data[p]['H'][i,j]/(sd[i]*sd[j]) for (i,j) in oi])
        d,_,_,_=np.linalg.lstsq(G,Hoff,rcond=None)                  # exact off-diag solve (bias-free)
        Hii=np.array([(d*Wr[:,i]**2).sum() for i in range(k)])
        mu=(np.diag(data[p]['H'])-Hii*sd**2)/data[p]['g']
        s_new=np.clip((1-mu)/2,1e-6,1-1e-6)
        b_rec[p]=np.log(s_new/(1-s_new))-(data[p]['z0']-b1.numpy())
    return len(Wr), b_rec

# mechanism check: TRUE biases
s_true=[torch.sigmoid(torch.tensor(d['z0'])).numpy() for d in data]
nf,b_rec=solve(s_true); err=np.abs(b_rec-b1.numpy()[None,:])
print(f"[mechanism true-s] found={nf} bias err max={err.max():.2e} median={np.median(err):.2e} cross-probe std={b_rec.std(0).max():.2e}")
# non-oracle: start from guess biases (~5e-2), iterate
rng=np.random.default_rng(1); b_est=b1.numpy()+0.05*(2*rng.random(k)-1)
print(f"start guess err: max={np.abs(b_est-b1.numpy()).max():.2e}")
for it in range(8):
    s_cur=[torch.sigmoid(torch.tensor(d['z0']-b1.numpy()+b_est)).numpy() for d in data]
    nf,b_rec=solve(s_cur); b_new=b_rec.mean(0)
    print(f"  iter {it}: found={nf}  bias err max={np.abs(b_new-b1.numpy()).max():.2e} median={np.median(np.abs(b_new-b1.numpy())):.2e}  cross-probe std={b_rec.std(0).max():.2e}")
    b_est=b_new
