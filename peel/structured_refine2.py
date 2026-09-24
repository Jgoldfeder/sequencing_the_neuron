import torch, numpy as np
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
exec(open('realgate2.py').read().split('print("JOINT')[0])   # make, germ_deriv
from scipy.optimize import linear_sum_assignment
def cp_als(T,R,iters=800,restarts=12):
    Q,k,_=T.shape; normT=T.norm(); best=None; bestfit=-1e9
    for rs in range(restarts):
        g=torch.Generator().manual_seed(rs*7+1)
        A=torch.randn(Q,R,generator=g); B=torch.randn(k,R,generator=g); C=torch.randn(k,R,generator=g)
        for it in range(iters):
            A=torch.einsum('qij,ir,jr->qr',T,B,C)@torch.linalg.pinv((B.T@B)*(C.T@C))
            B=torch.einsum('qij,qr,jr->ir',T,A,C)@torch.linalg.pinv((A.T@A)*(C.T@C)); B=B/B.norm(dim=0,keepdim=True)
            C=torch.einsum('qij,qr,ir->jr',T,A,B)@torch.linalg.pinv((A.T@A)*(B.T@B)); C=C/C.norm(dim=0,keepdim=True)
        fit=1-float((T-torch.einsum('qr,ir,jr->qij',A,B,C)).norm()/normT)
        if fit>bestfit: bestfit=fit; best=(A,B.clone(),C.clone())
    return best,bestfit
def cosmatch(B,W2):
    Bn=B/B.norm(dim=0,keepdim=True); Wn=W2/W2.norm(dim=1,keepdim=True)
    Cc=(Wn@Bn).abs().numpy(); ri,ci=linear_sum_assignment(-Cc); return float(Cc[ri,ci].min()), float(Cc[ri,ci].mean())
dims=[6,8,4,4]; k,W,m,O=dims; Sstar,b1,down=make(dims,0); W2t=down[0]
gp=torch.Generator().manual_seed(4); Tp=(2*torch.rand(60,k,generator=gp)-1)*1.6
s_all=sig(Tp@Sstar.T+b1)                                  # (P,k)
G1t,G2t=germ_deriv(s_all,*down); s1=G1t.abs().max(); s2=G2t.abs().max()
Hs=[G2t[p,o] for p in range(Tp.shape[0]) for o in range(O)]
(A,Bc,Cc),fit=cp_als(torch.stack(Hs),W); mi0,me0=cosmatch(Bc,W2t)
print(f"CP-8 init: recovery min-cos={mi0:.4f} mean-cos={me0:.4f}")
def fit_downstream(W2init,seed,steps=6000):
    g=torch.Generator().manual_seed(seed)
    P=[W2init.clone().requires_grad_(True),
       (0.1*torch.randn(W,generator=g)).requires_grad_(True),
       (torch.randn(m,W,generator=g)/np.sqrt(W)).requires_grad_(True),
       (0.1*torch.randn(m,generator=g)).requires_grad_(True),
       (torch.randn(O,m,generator=g)/np.sqrt(m)).requires_grad_(True),
       torch.zeros(O,requires_grad=True)]
    opt=torch.optim.Adam(P,lr=0.02)
    for it in range(steps):
        opt.zero_grad(); G1,G2=germ_deriv(s_all,*P)
        L=((G1-G1t)**2).mean()/s1**2+((G2-G2t)**2).mean()/s2**2; L.backward(); opt.step()
        if it==steps//2:
            for gr in opt.param_groups: gr['lr']=0.005
    return P[0].detach(),float(L)
# warm start from CP directions (scale learned), vs random directions
for label,W2i in [("CP-warm",(Bc.T/Bc.T.norm(dim=1,keepdim=True))*0.41),
                  ("random ",torch.randn(W,k,generator=torch.Generator().manual_seed(0))/np.sqrt(k))]:
    best=None
    for sd in range(3):
        W2f,L=fit_downstream(W2i,sd); mi,me=cosmatch(W2f.T,W2t)
        if best is None or L<best[0]: best=(L,mi,me)
    print(f"  {label} -> germ-jet fit best: loss={best[0]:.2e} W2 recovery min-cos={best[1]:.4f} mean-cos={best[2]:.4f}")
