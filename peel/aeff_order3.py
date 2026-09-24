"""ACTUAL order-3 local-identifiability test on the real architecture.
Directions+magnitudes assumed exact (first-layer weights fixed). theta=(first-layer
biases b_1..b_k ; ALL downstream params eta). Dual coords t_i shift z_i only.
Mixed jet J3={F_i,F_ij,F_ijk} at P probe points. A=dJ3/db, B=dJ3/deta.
A_eff=(I-BB^+)A. sigma_min(A_eff)>0 with full rank k => order-3 jet locally
identifies ALL biases in the actual network. Measured, not extrapolated."""
import torch, itertools, numpy as np
from torch.func import jacrev, jacfwd
torch.set_default_dtype(torch.float64)

def build(dims, seed):
    g=torch.Generator().manual_seed(seed)
    Ws=[]; bs=[]
    for i in range(len(dims)-1):
        Ws.append((torch.randn(dims[i+1],dims[i],generator=g))/np.sqrt(dims[i]))
        bs.append((2*torch.rand(dims[i+1],generator=g)-1)*0.5)
    d=dims[0]; k=dims[1]
    W1=Ws[0]; b1=bs[0]
    # dual directions: W1 @ V = I  => moving along V[:,i] shifts z_i by t_i only
    V=W1.t()@torch.linalg.inv(W1@W1.t())               # (d,k)
    # probe base: choose x0 giving moderate first-layer preacts, moderate downstream
    x0=torch.zeros(d)
    # theta = [b1 (k), then downstream weights+biases flattened]
    eta=torch.cat([Ws[l].reshape(-1) for l in range(1,len(Ws))]+[bs[l] for l in range(1,len(bs))])
    theta0=torch.cat([b1, eta]); nb=k
    Wshapes=[Ws[l].shape for l in range(1,len(Ws))]
    def forward(theta, x):
        b1_=theta[:k]; idx=k
        Wd=[];
        for sh in Wshapes:
            n=sh[0]*sh[1]; Wd.append(theta[idx:idx+n].reshape(sh)); idx+=n
        bd=[]
        for sh in Wshapes:
            bd.append(theta[idx:idx+sh[0]]); idx+=sh[0]
        h=torch.sigmoid(W1@x+b1_)
        for l,(W,bb) in enumerate(zip(Wd,bd)):
            z=W@h+bb
            h=torch.sigmoid(z) if l<len(Wd)-1 else z       # linear output layer
        return h
    return theta0, nb, V, x0, forward, W1

def jet_jac(dims, seed, nprobe=3, nproj=None):
    theta0, nb, V, x0, forward, W1 = build(dims, seed)
    d=dims[0]; k=dims[1]; O=dims[-1]
    gp=torch.Generator().manual_seed(seed+7)
    # output projections (capture full O-dim output row space): use O random projections
    R = O if nproj is None else nproj
    projs=[torch.randn(O,generator=gp) for _ in range(R)]
    probes=[ (2*torch.rand(k,generator=gp)-1)*1.5 for _ in range(nprobe)]   # dual-coord probe offsets
    # unique multi-index lists
    idx1=[(i,) for i in range(k)]
    idx2=list(itertools.combinations_with_replacement(range(k),2))
    idx3=list(itertools.combinations_with_replacement(range(k),3))
    rows1=[]; rows2=[]; rows3=[]
    for w in projs:
        for pb in probes:
            def Fp(t, theta):
                x=x0 + V@(pb+t)                    # shift z by (pb+t); t=0 is the probe
                return torch.dot(w, forward(theta, x))
            g=jacrev(Fp, argnums=1)                # d/dtheta -> (Ntheta,)
            t0=torch.zeros(k)
            D1=jacfwd(g,argnums=0)(t0,theta0)                          # (Ntheta,k)
            D2=jacfwd(jacfwd(g,argnums=0),argnums=0)(t0,theta0)        # (Ntheta,k,k)
            D3=jacfwd(jacfwd(jacfwd(g,argnums=0),argnums=0),argnums=0)(t0,theta0)  # (Ntheta,k,k,k)
            for (i,) in idx1: rows1.append(D1[:,i])
            for (i,j) in idx2: rows2.append(D2[:,i,j])
            for (i,j,l) in idx3: rows3.append(D3[:,i,j,l])
    return [torch.stack(rows1).numpy(), torch.stack(rows2).numpy(), torch.stack(rows3).numpy()], nb

def aeff_sigma(M, nb):
    A=M[:,:nb]; B=M[:,nb:]
    X,_,_,_=np.linalg.lstsq(B, A, rcond=None)     # B^+ A
    Aeff=A - B@X
    return np.linalg.svd(Aeff,compute_uv=False), np.linalg.svd(A,compute_uv=False)

for tag,dims,seeds in [("VALIDATE 16->6->8->6->3",[16,6,8,6,3],[0]),
                       ("REAL     128->24->32->16->8",[128,24,32,16,8],[1,2])]:
    for seed in seeds:
        Ms,nb=jet_jac(dims,seed)
        print(f"\n{tag} (seed {seed}): biases {nb}, downstream {Ms[0].shape[1]-nb}")
        for hi,label in [(1,"order<=1"),(2,"order<=2"),(3,"order<=3")]:
            M=np.concatenate(Ms[:hi],0)
            sv,svA=aeff_sigma(M,nb)
            rank=int((sv>sv.max()*1e-9).sum())
            print(f"   {label}: rows={M.shape[0]:6d}  rank(A_eff)={rank:2d}/{nb}  "
                  f"sigma_min(A_eff)={sv.min():.2e}  cond={sv.max()/sv.min():.1f}  "
                  f"retention sigma_min(A_eff)/sigma_min(A)={sv.min()/svA.min():.2f}")
