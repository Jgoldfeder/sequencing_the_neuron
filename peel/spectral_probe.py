import torch, numpy as np
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
exec(open('realgate2.py').read().split('print("JOINT')[0])   # make, etc.
from scipy.optimize import linear_sum_assignment
def S_H(down,s,o):
    W2,b2,W3,b3,w4,b4=down
    p2=W2@s+b2; sp=sig(p2); spd=sp*(1-sp); spdd=spd*(1-2*sp); q=sp
    u=W3@q+b3; hh=sig(u); hd=hh*(1-hh); hdd=hd*(1-2*hh)
    rho=W3.T@(w4[o]*hd); K=W3.T@torch.diag(w4[o]*hdd)@W3
    Sdiag=torch.diag(rho*spdd); Sfull=Sdiag+torch.diag(spd)@K@torch.diag(spd)
    off=Sfull-torch.diag(torch.diag(Sfull))
    cont=float(off.norm()/(torch.diag(Sfull).norm()+1e-12))
    return (W2.T@Sdiag@W2), (W2.T@Sfull@W2), cont
def recover(H1,H2,W2):
    M=H1@torch.linalg.inv(H2); ev,V=torch.linalg.eig(M); V=V.real
    Vn=V/(V.norm(dim=0,keepdim=True)+1e-12); Wn=W2/W2.norm(dim=1,keepdim=True)
    C=(Wn@Vn).abs().numpy(); ri,ci=linear_sum_assignment(-C)
    return float(C[ri,ci].min()), float(C[ri,ci].mean())
def run(dims,tag):
    k,W,m,O=dims; Sstar,b1,down=make(dims,0); W2=down[0]
    gp=torch.Generator().manual_seed(4); T=(2*torch.rand(60,k,generator=gp)-1)*1.6
    conts=[]; Hd=[]; Hf=[]
    for p in range(T.shape[0]):
        s=sig(Sstar@T[p]+b1)
        hd0,hf0,c=S_H(down,s,0); conts.append(c); Hd.append(hd0); Hf.append(hf0)
    conts=np.array(conts)
    print(f"\n{tag}  dims={dims} (W {'=' if W==k else '>'} k => {'matrix-Jennrich OK' if W<=k else 'OVER-COMPLETE: needs order-3 tensor'})")
    print(f"  contamination ||offdiag(S)||/||diag(S)|| across 60 probes: median={np.median(conts):.2f} min={conts.min():.2f} max={conts.max():.2f}")
    if W<=k:
        order=np.argsort(conts)
        # ideal (diag) recovery from a low-contam pair
        a,b=order[0],order[1]
        mi,me=recover(Hd[a],Hd[b],W2); print(f"  IDEAL (diag-only Hessians) direction recovery: min-cos={mi:.4f} mean-cos={me:.4f}  (expect ~1)")
        # real recovery: lowest-contam pair vs a random/high-contam pair
        mi2,me2=recover(Hf[a],Hf[b],W2); print(f"  REAL  (full Hessians, LOWEST-contam pair {conts[a]:.2f},{conts[b]:.2f}): min-cos={mi2:.4f} mean-cos={me2:.4f}")
        hi=order[-1]; mi3,me3=recover(Hf[order[-1]],Hf[order[-2]],W2); print(f"  REAL  (full Hessians, HIGHEST-contam pair {conts[order[-1]]:.2f},{conts[order[-2]]:.2f}): min-cos={mi3:.4f} mean-cos={me3:.4f}")
        # averaged over many low-contam pairs
        got=[]
        for ii in range(6):
            for jj in range(ii+1,6):
                got.append(recover(Hf[order[ii]],Hf[order[jj]],W2)[1])
        print(f"  REAL  mean-cos over 15 low-contam pairs: {np.mean(got):.4f} +/- {np.std(got):.4f}")
run([6,6,4,4],"SQUARE anchor")
run([6,8,4,4],"OVER-COMPLETE (real-style)")
