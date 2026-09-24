import torch, numpy as np
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
exec(open('realgate2.py').read().split('print("JOINT')[0])
from scipy.optimize import linear_sum_assignment
def S_H(down,s,o):
    W2,b2,W3,b3,w4,b4=down
    p2=W2@s+b2; sp=sig(p2); spd=sp*(1-sp); spdd=spd*(1-2*sp); q=sp
    u=W3@q+b3; hh=sig(u); hd=hh*(1-hh); hdd=hd*(1-2*hh)
    z3max=float(u.abs().min())                       # how saturated is the LEAST-saturated L3 unit
    rho=W3.T@(w4[o]*hd); K=W3.T@torch.diag(w4[o]*hdd)@W3
    Sf=torch.diag(rho*spdd)+torch.diag(spd)@K@torch.diag(spd)
    off=Sf-torch.diag(torch.diag(Sf)); cont=float(off.norm()/(torch.diag(Sf).norm()+1e-12))
    return W2.T@Sf@W2, cont, z3max
def recover(H1,H2,W2):
    M=H1@torch.linalg.inv(H2); ev,V=torch.linalg.eig(M); V=V.real
    Vn=V/(V.norm(dim=0,keepdim=True)+1e-12); Wn=W2/W2.norm(dim=1,keepdim=True)
    C=(Wn@Vn).abs().numpy(); ri,ci=linear_sum_assignment(-C)
    return float(C[ri,ci].min()), float(C[ri,ci].mean())
dims=[6,6,4,4]; k=6; Sstar,b1,down=make(dims,0); W2=down[0]
gp=torch.Generator().manual_seed(4); T=(2*torch.rand(3000,k,generator=gp)-1)*2.2   # wide -> some saturate L3
recs=[]
for p in range(T.shape[0]):
    s=sig(Sstar@T[p]+b1); H,c,z=S_H(down,s,0); recs.append((c,z,H))
conts=np.array([r[0] for r in recs]); order=np.argsort(conts)
print(f"square [6,6,4,4], 3000 wide probes: contamination min={conts.min():.3f} med={np.median(conts):.3f} max={conts.max():.3f}")
print("recovery (pairwise Jennrich) vs contamination-selection threshold:")
for thr in [0.02,0.04,0.06,0.10,0.20]:
    sel=[i for i in order if conts[i]<thr][:12]
    if len(sel)<2: print(f"  contam<{thr}: only {len(sel)} probes"); continue
    ms=[]; mns=[]
    for a in range(len(sel)):
        for b in range(a+1,len(sel)):
            mi,me=recover(recs[sel[a]][2],recs[sel[b]][2],W2); ms.append(mi); mns.append(me)
    print(f"  contam<{thr}: n={len(sel)} probes, {len(ms)} pairs -> min-cos={np.mean(ms):.4f} mean-cos={np.mean(mns):.4f} (best pair min-cos={np.max(ms):.4f})")
