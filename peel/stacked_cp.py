import torch, numpy as np
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
exec(open('realgate2.py').read().split('print("JOINT')[0])   # make
from scipy.optimize import linear_sum_assignment
def cp_als(T,R,iters=800,restarts=12):
    Q,k,_=T.shape; normT=T.norm(); best=None; bestfit=-1e9
    for rs in range(restarts):
        g=torch.Generator().manual_seed(rs*7+1)
        A=torch.randn(Q,R,generator=g); B=torch.randn(k,R,generator=g); C=torch.randn(k,R,generator=g)
        for it in range(iters):
            M=torch.einsum('qij,ir,jr->qr',T,B,C); A=M@torch.linalg.pinv((B.T@B)*(C.T@C))
            M=torch.einsum('qij,qr,jr->ir',T,A,C); B=M@torch.linalg.pinv((A.T@A)*(C.T@C)); B=B/B.norm(dim=0,keepdim=True)
            M=torch.einsum('qij,qr,ir->jr',T,A,B); C=M@torch.linalg.pinv((A.T@A)*(B.T@B)); C=C/C.norm(dim=0,keepdim=True)
        Trec=torch.einsum('qr,ir,jr->qij',A,B,C); fit=1-float((T-Trec).norm()/normT)
        if fit>bestfit: bestfit=fit; best=(A,B.clone(),C.clone())
    return best,bestfit
def cosmatch(B,W2):
    Bn=B/B.norm(dim=0,keepdim=True); Wn=W2/W2.norm(dim=1,keepdim=True)
    Cc=(Wn@Bn).abs().numpy(); ri,ci=linear_sum_assignment(-Cc)
    return float(Cc[ri,ci].min()), float(Cc[ri,ci].mean())
def build(dims,ideal):
    k,W,m,O=dims; Sstar,b1,down=make(dims,0); W2_,b2,W3,b3,w4,b4=down
    gp=torch.Generator().manual_seed(4); Tp=(2*torch.rand(40,k,generator=gp)-1)*1.6
    sl=[]
    for p in range(Tp.shape[0]):
        s=sig(Sstar@Tp[p]+b1); p2=W2_@s+b2; sp=sig(p2); spd=sp*(1-sp); spdd=spd*(1-2*sp); q=sp
        u=W3@q+b3; hh=sig(u); hd=hh*(1-hh); hdd=hd*(1-2*hh)
        for o in range(O):
            rho=W3.T@(w4[o]*hd); a=rho*spdd
            if ideal: S=torch.diag(a)
            else:
                K=W3.T@torch.diag(w4[o]*hdd)@W3; S=torch.diag(a)+torch.diag(spd)@K@torch.diag(spd)
            sl.append(W2_.T@S@W2_)
    return torch.stack(sl), W2_
for dims in [[6,8,4,4],[6,6,4,4]]:
    k,W=dims[0],dims[1]
    T,W2=build(dims,ideal=True); (A,B,C),fit=cp_als(T,W)
    mi,me=cosmatch(B,W2)
    print(f"dims={dims} W={W}>k={k}? {'OVERCOMPLETE' if W>k else 'square'} | IDEAL diag tensor {tuple(T.shape)} rank-{W} CP: fit={fit:.5f} recovery min-cos={mi:.4f} mean-cos={me:.4f}")
    T,W2=build(dims,ideal=False); (A,B,C),fit=cp_als(T,W)
    mi,me=cosmatch(B,W2)
    print(f"           REAL (with layer-3 curvature) rank-{W} CP: fit={fit:.5f} recovery min-cos={mi:.4f} mean-cos={me:.4f}")
