import torch, numpy as np
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
exec(open('realgate2.py').read().split('print("JOINT')[0])
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
dims=[6,8,4,4]; k,W,m,O=dims; Sstar,b1,down=make(dims,0); W2t,b2t,W3t,b3t,w4,b4=down
gp=torch.Generator().manual_seed(4); Tp=(2*torch.rand(40,k,generator=gp)-1)*1.6
Hs=[]; svs=[]
for p in range(Tp.shape[0]):
    s=sig(Sstar@Tp[p]+b1); p2=W2t@s+b2t; sp=sig(p2); spd=sp*(1-sp); spdd=spd*(1-2*sp); q=sp
    u=W3t@q+b3t; hh=sig(u); hd=hh*(1-hh); hdd=hd*(1-2*hh)
    for o in range(O):
        rho=W3t.T@(w4[o]*hd); K=W3t.T@torch.diag(w4[o]*hdd)@W3t
        S=torch.diag(rho*spdd)+torch.diag(spd)@K@torch.diag(spd)
        Hs.append(W2t.T@S@W2t); svs.append(s)
H=torch.stack(Hs); Sv=torch.stack(svs)   # (Q,k,k),(Q,k)
(A,Bc,Cc),fit=cp_als(H,W); mi0,me0=cosmatch(Bc,W2t)
print(f"CP-8 on REAL tensor: fit={fit:.4f} recovery min-cos={mi0:.4f} mean-cos={me0:.4f}  (init for refinement)")
iu=torch.triu_indices(k,k); sc=torch.where(iu[0]==iu[1],1.0,np.sqrt(2))
def vech(M): return M[...,iu[0],iu[1]]*sc
tgt=vech(H)                                            # (Q,21)
def resid(W2,b2,W3):
    d2=sig(Sv@W2.T+b2)*(1-sig(Sv@W2.T+b2))             # (Q,W)  layer-2 gates per slice
    wout=torch.einsum('ri,rj->rij',W2,W2)              # (W,k,k) w_r w_r^T  (shared)
    V=torch.einsum('nr,qr,ri->qni',W3,d2,W2)           # (Q,m,k)  v_{q,n}=(W3 D2 W2)_n
    vout=torch.einsum('qni,qnj->qnij',V,V)             # (Q,m,k,k)
    basis=torch.cat([wout.unsqueeze(0).expand(H.shape[0],-1,-1,-1),vout],dim=1)  # (Q,W+m,k,k)
    D=vech(basis).transpose(1,2)                       # (Q,21,W+m)
    G=D.transpose(1,2)@D + 1e-9*torch.eye(W+m); coef=torch.linalg.solve(G, D.transpose(1,2)@tgt.unsqueeze(-1))
    r=tgt.unsqueeze(-1)-D@coef
    return (r.squeeze(-1)**2).sum()
W2=Bc.T.clone().requires_grad_(True)                   # CP init (rows = directions)
b2=torch.zeros(W,requires_grad=True); W3=(0.3*torch.randn(m,W,generator=torch.Generator().manual_seed(2))).requires_grad_(True)
opt=torch.optim.Adam([W2,b2,W3],lr=0.02)
for it in range(4000):
    opt.zero_grad(); L=resid(W2,b2,W3); L.backward(); opt.step()
    if it%1000==0 or it==3999:
        mi,me=cosmatch(W2.detach().T,W2t); print(f"  refine it{it}: loss={float(L):.3e} recovery min-cos={mi:.4f} mean-cos={me:.4f}")
