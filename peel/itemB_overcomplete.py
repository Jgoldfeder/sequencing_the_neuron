"""ITEM B: overcomplete 6->8->1 (W=8>k=6). Structured extraction:
w_r shared across probes (W2 fixed), (c_r,rho_r) per probe. Fit shared-w + per-probe
(c,rho) to CLEAN observables g,H_offdiag,T_distinct with recurrence tau=(3rho^2-1)/2.
Then reconstruct H_ii = sum_r c_r rho_r w_ri^2 and get 1-2 s_i = (F_ii - H_ii sdot_i^2)/F_i.
Root/branch selection uses ONLY the approximate bias. Report bias recovery error."""
import numpy as np, itertools, torch
from scipy.optimize import least_squares
torch.set_default_dtype(torch.float64)
k,W=6,8; P=4
g_=torch.Generator().manual_seed(3)
W2=(torch.randn(W,k,generator=g_))*0.9; b2=(2*torch.rand(W,generator=g_)-1)*0.6
w3=(torch.randn(W,generator=g_))*0.9; b3=(2*torch.rand(1,generator=g_)-1)*0.3
b1=(2*torch.rand(k,generator=g_)-1)*0.5                       # TRUE first-layer biases
def F(t, z0):                                                 # dual coords: z_i = z0_i + t_i
    s=torch.sigmoid(z0+t); q=torch.sigmoid(W2@s+b2); return (w3@q+b3).squeeze()
# probes: base preactivations z0^(p)
gp=torch.Generator().manual_seed(8); z0s=[(2*torch.rand(k,generator=gp)-1)*1.2 + b1 for _ in range(P)]
pairs2=list(itertools.combinations(range(k),2)); trip=list(itertools.combinations(range(k),3))
def jets(z0):
    t=torch.zeros(k,requires_grad=True); f=F(t,z0)
    g1=torch.autograd.grad(f,t,create_graph=True)[0]
    H=torch.zeros(k,k); T=torch.zeros(k,k,k)
    for i in range(k):
        gi=torch.autograd.grad(g1[i],t,create_graph=True)[0]; H[i]=gi
        for j in range(k):
            if j>=i: T[i,j]=torch.autograd.grad(gi[j],t,retain_graph=True)[0]
    return g1.detach().numpy(), H.detach().numpy(), T.detach().numpy(), f
# ---- measured jets ----
data=[]
for z0 in z0s:
    g1,H,T,_=jets(z0); s=torch.sigmoid(z0).numpy(); sd=s*(1-s); sdd=sd*(1-2*s)
    data.append(dict(g1=g1,H=H,T=T,s=s,sd=sd,sdd=sdd,z0=z0.numpy()))
# ---- clean germ observables (use guess biases to convert) ----
def clean(d, s_use):
    sd=s_use*(1-s_use)
    g=d['g1']/sd
    Hoff=np.array([d['H'][i,j]/(sd[i]*sd[j]) for (i,j) in pairs2])
    Tdist=np.array([d['T'][min(i,j,l),sorted([i,j,l])[1],max(i,j,l)]/(sd[i]*sd[j]*sd[l]) for (i,j,l) in trip])
    return g,Hoff,Tdist
def extract(s_guess_list, restarts=6):
    obs=[clean(data[p], s_guess_list[p]) for p in range(P)]
    def resid(x):
        w=x[:W*k].reshape(W,k); c=x[W*k:W*k+W*P].reshape(P,W); rho=x[W*k+W*P:].reshape(P,W)
        tau=(3*rho**2-1)/2; r=[]
        for p in range(P):
            gp_=(c[p][:,None]*w).sum(0)                                   # sum_r c_r w_r
            Hp=np.array([ (c[p]*rho[p]*w[:,i]*w[:,j]).sum() for (i,j) in pairs2])
            Tp=np.array([ (c[p]*tau[p]*w[:,i]*w[:,j]*w[:,l]).sum() for (i,j,l) in trip])
            r+=[gp_-obs[p][0], Hp-obs[p][1], Tp-obs[p][2]]
        return np.concatenate(r)
    best=None
    for rs in range(restarts):
        rng=np.random.default_rng(rs)
        x0=np.concatenate([rng.standard_normal(W*k)*0.6, rng.standard_normal(W*P)*0.5, (2*rng.random(W*P)-1)*0.5])
        sol=least_squares(resid,x0,method='lm',max_nfev=4000)
        if best is None or sol.cost<best.cost: best=sol
    x=best.x; w=x[:W*k].reshape(W,k); c=x[W*k:W*k+W*P].reshape(P,W); rho=x[W*k+W*P:].reshape(P,W)
    # reconstruct H_ii per probe and recover biases
    b_rec=np.zeros((P,k))
    for p in range(P):
        Hii=np.array([ (c[p]*rho[p]*w[:,i]**2).sum() for i in range(k)])
        sd=data[p]['s']*(1-data[p]['s'])              # NOTE uses s_guess in practice; here true for mechanism check
        mu=(np.diag(data[p]['H']) - Hii*sd**2)/data[p]['g1']*sd   # (F_ii - H_ii sd^2)/F_i ; F_i=g1
        # F_i = g1_i (already dF/dt). (F_ii-H_ii sd^2)/F_i = sdd/sd = 1-2s
        mu=(np.diag(data[p]['H']) - Hii*sd**2)/data[p]['g1']
        s_new=np.clip((1-mu)/2,1e-6,1-1e-6)
        b_rec[p]=np.log(s_new/(1-s_new)) - (data[p]['z0']-b1.numpy())  # z0 = (z0-b1)+b1 ; recover b1
    return best.cost, b_rec
# (1) mechanism check: convert with TRUE biases
s_true=[data[p]['s'] for p in range(P)]
cost,b_rec=extract(s_true)
err=np.abs(b_rec-b1.numpy()[None,:])
print(f"[mechanism, true-s convert] germ-fit residual={cost:.2e}")
print(f"  bias recovery err (per probe, should agree): max={err.max():.2e} median={np.median(err):.2e}")
print(f"  cross-probe consistency of recovered b: std over probes max={b_rec.std(0).max():.2e}")
