"""ITEM C: hierarchical Hessian-subspace diagnostic (target-only, spectral).
Model: germ Hessian G^(p,o) = A^(p,o) + L^(p,o), A^(p,o) in shared A=span{w_r w_r^T}
(dim W, ALL p,o), L^(p,o) in L_p=span{z_pn z_pn^T} (dim m, shared across outputs o at
probe p). Fit (A,{L_p}) by pure alternating SVD projection (NO net fit). Reconstruct
full germ Hessian at candidate b from black-box F_i,F_ij (user's formulas). Test residual
at true b, then perturbed b (delta parabola). KEY: discriminates only if #outputs > m."""
import numpy as np
sig=lambda x:1/(1+np.exp(-x))
def build(k,Wd,md,O,seed):
    rng=np.random.default_rng(seed)
    return dict(W2=rng.standard_normal((Wd,k))*0.9,b2=(2*rng.random(Wd)-1)*0.6,
                W3=rng.standard_normal((md,Wd))*0.7,b3=(2*rng.random(md)-1)*0.5,
                W4=rng.standard_normal((O,md))*0.8,b4=(2*rng.random(O)-1)*0.3,
                beta=(2*rng.random(k)-1)*0.5,k=k,Wd=Wd,md=md,O=O)
def grad_hess(net,s):
    W2,b2,W3,b3,W4=net['W2'],net['b2'],net['W3'],net['b3'],net['W4']; O=net['O']; k=net['k']
    p2=W2@s+b2; sp=sig(p2); spd=sp*(1-sp); spdd=spd*(1-2*sp); q=sp
    u=W3@q+b3; hh=sig(u); hd=hh*(1-hh); hdd=hd*(1-2*hh)
    Gg=np.zeros((O,k)); Gh=np.zeros((O,k,k))
    for o in range(O):
        rho=W3.T@(W4[o]*hd); K=W3.T@np.diag(W4[o]*hdd)@W3
        Gg[o]=W2.T@(rho*spd); S=np.diag(rho*spdd)+np.diag(spd)@K@np.diag(spd); Gh[o]=W2.T@S@W2
    return Gg,Gh
def make_jet(net,cs):
    Fi=[];FH=[]
    for c in cs:
        s=sig(c+net['beta']); d=s*(1-s); sdd=d*(1-2*s); Gg,Gh=grad_hess(net,s)
        fi=Gg*d[None,:]; fh=Gh*np.outer(d,d)[None,:,:]
        for o in range(net['O']): np.fill_diagonal(fh[o], np.diag(Gh[o])*d**2+Gg[o]*sdd)
        Fi.append(fi); FH.append(fh)
    return np.array(Fi),np.array(FH)     # (P,O,k),(P,O,k,k)
def reconstruct(net,b,cs,Fi,FH):
    P,O,k=Fi.shape; iu=np.triu_indices(k); sc=np.where(iu[0]==iu[1],1.0,np.sqrt(2))
    gs=np.zeros((P,O,len(iu[0])))
    for p in range(P):
        s=sig(cs[p]+b); d=s*(1-s); r=1-2*s
        for o in range(O):
            G=FH[p,o]/np.outer(d,d)
            np.fill_diagonal(G,(np.diag(FH[p,o])-Fi[p,o]*r)/d**2)
            gs[p,o]=G[iu]*sc
    return gs
def altproj(gs,Wdim,mdim,iters=400):
    P,O,D=gs.shape; A=np.linalg.svd(gs.reshape(P*O,D).T,full_matrices=False)[0][:,:Wdim]
    Ls=[None]*P
    for it in range(iters):
        Pa=A@A.T
        for p in range(P):
            R=gs[p].T-Pa@gs[p].T; Ls[p]=np.linalg.svd(R,full_matrices=False)[0][:,:mdim]
        cols=[gs[p].T-Ls[p]@Ls[p].T@gs[p].T for p in range(P)]
        A=np.linalg.svd(np.concatenate(cols,axis=1),full_matrices=False)[0][:,:Wdim]
    tot=0.0
    for p in range(P):
        B=np.concatenate([A,Ls[p]],axis=1); Pb=B@np.linalg.pinv(B)
        for o in range(O): rr=gs[p,o]-Pb@gs[p,o]; tot+=rr@rr
    return tot/(P*O)
def run(k,Wd,md,O,tag):
    net=build(k,Wd,md,O,3); rng=np.random.default_rng(11); cs=[(2*rng.random(k)-1)*1.3 for _ in range(15)]
    Fi,FH=make_jet(net,cs); bt=net['beta']
    print(f"{tag}: k={k} W={Wd} m={md} O={O}  (discriminates iff O>m={md})")
    for d in (0.0,1e-4,1e-3,1e-2):
        b=bt.copy(); b[0]+=d; res=altproj(reconstruct(net,b,cs,Fi,FH),Wd,md)
        print(f"   delta={d:+.0e}: residual = {res:.3e}")
run(6,8,4,4,"6->8->4->4 (O=m, user's toy)")
run(6,8,2,4,"6->8->2->4 (O>m, non-degenerate)")
