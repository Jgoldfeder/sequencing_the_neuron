"""Does probing MORE first-layer coordinates lower the required mixed order for a
FIXED deep downstream? First layer = k neurons (t_i controls z_i). Downstream fixed
depth. Find min mixed order for ALL k biases to be identifiable, as k grows."""
import torch, numpy as np, itertools
torch.set_default_dtype(torch.float64)

def make(k, hidden, seed=0):
    g=torch.Generator().manual_seed(seed)
    a=1.0+0.4*torch.rand(k,generator=g); b=(2*torch.rand(k,generator=g)-1)*0.5
    sizes=[k]+hidden+[1]; Ws=[];bs=[]
    for i in range(len(sizes)-1):
        Ws.append((2*torch.rand(sizes[i+1],sizes[i],generator=g)-1)*1.2)
        bs.append((2*torch.rand(sizes[i+1],generator=g)-1)*0.5)
    theta=torch.cat([b]+[W.reshape(-1) for W in Ws]+[bb for bb in bs])
    meta=(k,[ (W.shape) for W in Ws])
    return theta, meta, a

def fwd(theta, meta, a, t):
    k,Wsh=meta; b=theta[:k]; idx=k
    h=torch.sigmoid(a*t+b).reshape(k,1)
    Ws=[]
    for sh in Wsh:
        n=sh[0]*sh[1]; Ws.append(theta[idx:idx+n].reshape(sh)); idx+=n
    bs=[]
    for sh in Wsh:
        bs.append(theta[idx:idx+sh[0]]); idx+=sh[0]
    for i,(W,bb) in enumerate(zip(Ws,bs)):
        h=W@h+bb.reshape(-1,1)
        if i<len(Ws)-1: h=torch.sigmoid(h)
    return h.squeeze()

def jet(theta, meta, a, tv, n):
    k=meta[0]; t=torch.tensor(tv,requires_grad=True)
    f=fwd(theta,meta,a,t)
    cur={tuple([0]*k):f}
    for order in range(1,n+1):
        nxt={}
        for idx,val in list(cur.items()):
            if sum(idx)==order-1:
                for c in range(k):
                    ni=list(idx); ni[c]+=1; ni=tuple(ni)
                    if ni not in nxt and ni not in cur:
                        gr=torch.autograd.grad(val,t,create_graph=True,retain_graph=True)[0]
                        for cc in range(k):
                            key=list(idx); key[cc]+=1; nxt[tuple(key)]=gr[cc]
                break_flag=True
        # simpler: recompute per node (above grad returns full gradient vector)
        cur.update(nxt)
    comps=[cur[idx] for idx in sorted(cur) if 1<=sum(idx)<=n]
    return torch.stack(comps)

def min_order(k, hidden, seed=0):
    theta,meta,a=make(k,hidden,seed)
    for n in (2,3,4):
        th=theta.clone().requires_grad_(True)
        tv=torch.linspace(0.15,-0.25,k)
        J=torch.autograd.functional.jacobian(lambda t: jet(t,meta,a,tv,n), th, vectorize=True).detach().numpy()
        S=np.linalg.svd(J,compute_uv=False) if J.shape[0]>=J.shape[1] else None
        U,Sv,Vt=np.linalg.svd(J)
        tol=Sv.max()*1e-9
        null=[Vt[i] for i in range(Vt.shape[0]) if i>=len(Sv) or Sv[i]<tol]
        nb=np.array(null) if null else np.zeros((0,J.shape[1]))
        if nb.shape[0]==0 or np.abs(nb[:,:k]).max()<1e-6:
            return n, J.shape
    return '>4', None

print("Min mixed order for ALL biases identifiable, vs #coordinates k (fixed deep downstream):")
print("  downstream [4] (width4, 1 hidden):")
for k in (2,3,4,5,6):
    mo,sh=min_order(k,[4]); print(f"    k={k}: min order={mo}   (jet size x params = {sh})")
print("  downstream [4,4] (deep, 2 hidden):")
for k in (2,3,4,5,6):
    mo,sh=min_order(k,[4,4]); print(f"    k={k}: min order={mo}   (jet size x params = {sh})")
print("  downstream [6,4] (deeper/wider):")
for k in (3,4,5,6,7):
    mo,sh=min_order(k,[6,4]); print(f"    k={k}: min order={mo}   (jet size x params = {sh})")
