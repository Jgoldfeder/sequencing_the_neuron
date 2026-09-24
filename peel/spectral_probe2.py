import torch, numpy as np
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
exec(open('realgate2.py').read().split('print("JOINT')[0])
from scipy.optimize import linear_sum_assignment
def S_H(down,s,o):
    W2,b2,W3,b3,w4,b4=down
    p2=W2@s+b2; sp=sig(p2); spd=sp*(1-sp); spdd=spd*(1-2*sp); q=sp
    u=W3@q+b3; hh=sig(u); hd=hh*(1-hh); hdd=hd*(1-2*hh)
    rho=W3.T@(w4[o]*hd); K=W3.T@torch.diag(w4[o]*hdd)@W3
    Sf=torch.diag(rho*spdd)+torch.diag(spd)@K@torch.diag(spd)
    off=Sf-torch.diag(torch.diag(Sf)); cont=float(off.norm()/(torch.diag(Sf).norm()+1e-12))
    return W2.T@Sf@W2, cont
def cosmatch(B,W2):     # rows of B vs rows of W2, up to sign/perm
    Bn=B/B.norm(dim=1,keepdim=True); Wn=W2/W2.norm(dim=1,keepdim=True)
    C=(Wn@Bn.T).abs().numpy(); ri,ci=linear_sum_assignment(-C)
    return float(C[ri,ci].min()), float(C[ri,ci].mean())
def joint_diag(Hs,k,iters=4000,lr=0.03):
    A=torch.eye(k,requires_grad=True); opt=torch.optim.Adam([A],lr=lr)
    Hs=torch.stack(Hs)
    for it in range(iters):
        opt.zero_grad()
        N=torch.einsum('ai,tij,jb->tab',A.T,Hs,A)            # A^T H A per matrix
        dg=torch.diagonal(N,dim1=1,dim2=2)
        offsq=(N.pow(2).sum(dim=(1,2))-dg.pow(2).sum(1))
        loss=(offsq/(dg.pow(2).sum(1)+1e-9)).sum()
        loss.backward(); opt.step()
    return torch.linalg.inv(A.detach())                       # rows ~ w2 rows (perm/scale)
dims=[6,6,4,4]; k=6; Sstar,b1,down=make(dims,0); W2=down[0]
gp=torch.Generator().manual_seed(4); T=(2*torch.rand(80,k,generator=gp)-1)*1.6
Hs=[]; conts=[]
for p in range(T.shape[0]):
    s=sig(Sstar@T[p]+b1)
    for o in range(4):
        H,c=S_H(down,s,o); Hs.append(H); conts.append(c)
conts=np.array(conts); order=np.argsort(conts)
print(f"square [6,6,4,4]: {len(Hs)} Hessians (probes x outputs), contamination med={np.median(conts):.2f}")
for ntop in [20,40,80]:
    idx=order[:ntop]; B=joint_diag([Hs[i] for i in idx],k)
    mi,me=cosmatch(B,W2); print(f"  JOINT-DIAG over {ntop} lowest-contam Hessians: min-cos={mi:.4f} mean-cos={me:.4f}")
B=joint_diag(Hs,k); mi,me=cosmatch(B,W2); print(f"  JOINT-DIAG over ALL {len(Hs)} Hessians: min-cos={mi:.4f} mean-cos={me:.4f}")
