"""ITEM B (vectorized): overcomplete 6->8->1. Structured extraction via shared-w
across probes + per-probe (c,rho), recurrence tau=(3rho^2-1)/2, fit to clean
observables g,H_offdiag,T_distinct. Reconstruct H_ii -> biases. Root selection by
approximate bias only. Mechanism check (true-s convert) then non-oracle (guess+iterate)."""
import numpy as np, itertools, torch
from scipy.optimize import least_squares
torch.set_default_dtype(torch.float64)
k,W=6,8; P=5
g_=torch.Generator().manual_seed(3)
W2=(torch.randn(W,k,generator=g_))*0.9; b2=(2*torch.rand(W,generator=g_)-1)*0.6
w3=(torch.randn(W,generator=g_))*0.9; b3=(2*torch.rand(1,generator=g_)-1)*0.3
b1=(2*torch.rand(k,generator=g_)-1)*0.5
def F(t,z0):
    s=torch.sigmoid(z0+t); q=torch.sigmoid(W2@s+b2); return (w3@q+b3).squeeze()
gp=torch.Generator().manual_seed(8); z0s=[(2*torch.rand(k,generator=gp)-1)*1.2 + b1 for _ in range(P)]
pairs2=list(itertools.combinations(range(k),2)); trip=list(itertools.combinations(range(k),3))
ii=np.array([a for a,b in pairs2]); jj=np.array([b for a,b in pairs2])
ta=np.array([a for a,b,c in trip]); tb=np.array([b for a,b,c in trip]); tc=np.array([c for a,b,c in trip])
def jets(z0):
    t=torch.zeros(k,requires_grad=True); f=F(t,z0)
    g1=torch.autograd.grad(f,t,create_graph=True)[0]; H=torch.zeros(k,k); T=torch.zeros(k,k,k)
    for i in range(k):
        gi=torch.autograd.grad(g1[i],t,create_graph=True)[0]; H[i]=gi
        for j in range(i,k): T[i,j]=torch.autograd.grad(gi[j],t,retain_graph=True)[0]
    return g1.detach().numpy(),H.detach().numpy(),T.detach().numpy()
data=[]
for z0 in z0s:
    g1,H,T=jets(z0); s=torch.sigmoid(z0).numpy()
    data.append(dict(g1=g1,H=H,T=T,s=s,z0=z0.numpy()))
def obs_from(s_list):
    obs=[]
    for p in range(P):
        d=data[p]; sd=s_list[p]*(1-s_list[p])
        g=d['g1']/sd
        Hoff=d['H'][ii,jj]/(sd[ii]*sd[jj])
        Tdist=d['T'][ta,tb,tc]/(sd[ta]*sd[tb]*sd[tc])
        obs.append((g,Hoff,Tdist))
    return obs
def fit_germ(obs, restarts=25):
    def resid(x):
        w=x[:W*k].reshape(W,k); c=x[W*k:W*k+W*P].reshape(P,W); rho=x[W*k+W*P:].reshape(P,W)
        tau=(3*rho**2-1)/2; cr=c*rho; ct=c*tau
        g_pred=c@w
        H_pred=np.einsum('pr,ri,rj->pij',cr,w,w)
        T_pred=np.einsum('pr,ri,rj,rl->pijl',ct,w,w,w)
        r=[]
        for p in range(P):
            r.append(g_pred[p]-obs[p][0]); r.append(H_pred[p][ii,jj]-obs[p][1]); r.append(T_pred[p][ta,tb,tc]-obs[p][2])
        return np.concatenate(r)
    best=None
    for rs in range(restarts):
        rng=np.random.default_rng(rs)
        x0=np.concatenate([rng.standard_normal(W*k)*0.7, rng.standard_normal(W*P)*0.5, (2*rng.random(W*P)-1)*0.8])
        sol=least_squares(resid,x0,method='lm',max_nfev=3000)
        if best is None or sol.cost<best.cost: best=sol
    return best
def recover_bias(x, s_convert):
    w=x[:W*k].reshape(W,k); c=x[W*k:W*k+W*P].reshape(P,W); rho=x[W*k+W*P:].reshape(P,W)
    b_rec=np.zeros((P,k))
    for p in range(P):
        Hii=(c[p][:,None]*rho[p][:,None]*w**2).sum(0)           # sum_r c rho w_ri^2
        sd=s_convert[p]*(1-s_convert[p])
        mu=(np.diag(data[p]['H']) - Hii*sd**2)/data[p]['g1']    # = 1-2 s_i
        s_new=np.clip((1-mu)/2,1e-6,1-1e-6)
        b_rec[p]=np.log(s_new/(1-s_new)) - (data[p]['z0']-b1.numpy())
    return b_rec
# ---- (1) mechanism: convert with TRUE s ----
s_true=[data[p]['s'] for p in range(P)]
best=fit_germ(obs_from(s_true))
b_rec=recover_bias(best.x, s_true); err=np.abs(b_rec-b1.numpy()[None,:])
print(f"[mechanism true-s] germ-fit residual cost={best.cost:.2e}")
print(f"  bias err max={err.max():.2e} median={np.median(err):.2e}; cross-probe std max={b_rec.std(0).max():.2e}")
# ---- (2) non-oracle: start from guess biases, iterate ----
rng=np.random.default_rng(0); b_guess=b1.numpy()+0.05*(2*rng.random(k)-1)
s_cur=[torch.sigmoid(torch.tensor(data[p]['z0']-b1.numpy()+b_guess)).numpy() for p in range(P)]
for it in range(5):
    best=fit_germ(obs_from(s_cur), restarts=15)
    b_rec=recover_bias(best.x, s_cur); b_est=b_rec.mean(0)
    s_cur=[torch.sigmoid(torch.tensor(data[p]['z0']-b1.numpy()+b_est)).numpy() for p in range(P)]
err2=np.abs(b_est-b1.numpy())
print(f"[non-oracle guess+iterate] final bias err max={err2.max():.2e} median={np.median(err2):.2e} (guess was ~5e-2)")
