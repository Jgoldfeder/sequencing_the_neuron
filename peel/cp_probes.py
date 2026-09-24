import torch, numpy as np
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
exec(open('/tmp/claude-1001/-home-judah/480010f4-8e92-4a50-8319-7f2f544ea513/scratchpad/realgate2.py').read().split('print("JOINT')[0])
from scipy.optimize import linear_sum_assignment
def cp_als(T,R,w=None,iters=1000,restarts=14):
    Q,k,_=T.shape; normT=T.norm(); best=None; bestfit=-1e9
    if w is None: w=torch.ones(Q)
    Tw=T*w[:,None,None]
    for rs in range(restarts):
        g=torch.Generator().manual_seed(rs*7+1)
        A=torch.randn(Q,R,generator=g); B=torch.randn(k,R,generator=g); C=torch.randn(k,R,generator=g)
        for it in range(iters):
            A=torch.einsum('qij,ir,jr->qr',T,B,C)@torch.linalg.pinv((B.T@B)*(C.T@C))
            Aw=A*w[:,None]
            B=torch.einsum('qij,qr,jr->ir',Tw,A,C)@torch.linalg.pinv((Aw.T@A)*(C.T@C)); B=B/B.norm(dim=0,keepdim=True)
            C=torch.einsum('qij,qr,ir->jr',Tw,A,B)@torch.linalg.pinv((Aw.T@A)*(B.T@B)); C=C/C.norm(dim=0,keepdim=True)
        fit=1-float((T-torch.einsum('qr,ir,jr->qij',A,B,C)).norm()/normT)
        if fit>bestfit: bestfit=fit; best=(A,B.clone(),C.clone())
    return best,bestfit
def cosmatch(B,W2):
    Bn=B/B.norm(dim=0,keepdim=True); Wn=W2/W2.norm(dim=1,keepdim=True)
    Cc=(Wn@Bn).abs().numpy(); ri,ci=linear_sum_assignment(-Cc); return float(Cc[ri,ci].min()), float(Cc[ri,ci].mean())
dims=[6,8,4,4]; k,W,m,O=dims; Sstar,b1,down=make(dims,0); W2t,b2t,W3t,b3t,w4,b4=down
def tensor(nprobes,width,seed):
    gp=torch.Generator().manual_seed(seed); Tp=(2*torch.rand(nprobes,k,generator=gp)-1)*width
    Hs=[]; conts=[]
    for p in range(nprobes):
        s=sig(Sstar@Tp[p]+b1); p2=W2t@s+b2t; sp=sig(p2); spd=sp*(1-sp); spdd=spd*(1-2*sp); q=sp
        u=W3t@q+b3t; hh=sig(u); hd=hh*(1-hh); hdd=hd*(1-2*hh)
        for o in range(O):
            rho=W3t.T@(w4[o]*hd); K=W3t.T@torch.diag(w4[o]*hdd)@W3t
            S=torch.diag(rho*spdd)+torch.diag(spd)@K@torch.diag(spd)
            off=(S-torch.diag(torch.diag(S))).norm()/(torch.diag(S).norm()+1e-9)
            Hs.append(W2t.T@S@W2t); conts.append(float(off))
    return torch.stack(Hs), torch.tensor(conts)
for npr,wd in [(60,1.6),(200,2.0),(400,2.4)]:
    T,c=tensor(npr,wd,4); (_,B,_),fit=cp_als(T,W); mi,me=cosmatch(B,W2t)
    print(f"probes={npr} width={wd}: {T.shape[0]} slices, contam med={float(c.median()):.2f} | CP-8 min-cos={mi:.4f} mean-cos={me:.4f}",flush=True)
    w=(c<c.median()).double()
    (_,B,_),fit=cp_als(T,W,w=w); mi,me=cosmatch(B,W2t)
    print(f"           low-contam-weighted CP-8: min-cos={mi:.4f} mean-cos={me:.4f}",flush=True)
