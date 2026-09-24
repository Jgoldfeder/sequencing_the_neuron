"""Dual-direction EXACT isolation + multi-harmonic tail fit for MAGNITUDE.
After SVD gives directions N, probe along v_j (columns of V=N^T(NN^T)^-1, so
N v_j = e_j): moving along v_j changes ONLY z_j, all other z_k EXACTLY constant.
Then g(t)=H(sigma(a t + c)); in the tails g = sum_l C_l exp(l a t). One nonlinear
unknown a -> variable projection. Does magnitude beat 1.7e-3, reach 1e-4?"""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch, numpy as np
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]; O=dims[-1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True); g=torch.Generator(device=dev).manual_seed(1)
@torch.no_grad()
def J_at(x,fd=5e-5):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()
Wg=torch.empty(k,d,device=dev,dtype=torch.float64); bg=torch.empty(k,device=dev,dtype=torch.float64)
for j in range(k):
    u=tn[j]; v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
    Wg[j]=(math.cos(math.radians(5))*u+math.sin(math.radians(5))*v)*W1t[j].norm()*(1+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64)-1))
    bg[j]=b1t[j]+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64).item()-1)
Wgpinv=Wg.t()@torch.linalg.inv(Wg@Wg.t())
# directions via SVD
N=torch.empty_like(Wg)
for j in range(k):
    t=torch.full((k,),20.0,device=dev,dtype=torch.float64); t[j]=0.0
    x0=Wgpinv@(t-bg); U,Sv,Vh=torch.linalg.svd(J_at(x0),full_matrices=False)
    nv=Vh[0]; N[j]=nv if float(nv@Wg[j])>0 else -nv
# dual directions: V = N^T (N N^T)^-1, columns v_j, N v_j = e_j
NNt=N@N.t(); print(f"cond(N N^T) = {float(torch.linalg.cond(NNt)):.1f}")
V=(N.t()@torch.linalg.inv(NNt))                      # (d,k)
W1rec=N*Wg.norm(dim=1,keepdim=True); W1rp=W1rec.t()@torch.linalg.inv(W1rec@W1rec.t())

# verify exact isolation: moving along v_j, how much do other z_k change (TRUE)?
j=0; vj=V[:,j]; chg=(W1t@vj); chg[j]=0
print(f"exact-isolation check (neuron 0): max |dz_k/dt| for k!=0 = {float(chg.abs().max()):.2e} (want ~0)\n")

P=6; nctx=8; ncand=40
def fit_mag(tails, a_g):
    ell=np.arange(P+1)[:,None]
    def res(a):
        tot=0.0
        for ts,gs,side in tails:
            A=np.exp(-side*a*ell*ts[None,:]).T
            coef,_,_,_=np.linalg.lstsq(A,gs,rcond=None); tot+=float(((A@coef-gs)**2).sum())
        return tot
    lo,hi=0.85*a_g,1.15*a_g                                    # NARROW window (guess within 8%)
    for _ in range(70):
        m1=hi-(hi-lo)*0.618; m2=lo+(hi-lo)*0.618
        if res(m1)<res(m2): hi=m2
        else: lo=m1
    return 0.5*(lo+hi)

merr=[]
for jj in range(k):
    j=jj; a_g=float(Wg[j].norm()); vj=V[:,j]
    # visibility-select contexts: swing = ||g(+T)-g(-T)|| (neuron-j transition amplitude)
    TT=(2*torch.rand(ncand,k,generator=g,device=dev,dtype=torch.float64)-1)*2.0; TT[:,j]=0.0
    X0=(TT-bg)@W1rp.t()
    with torch.no_grad():
        swing=(teacher(X0+(6.0/a_g)*vj)-teacher(X0-(6.0/a_g)*vj)).norm(dim=1)
    top=torch.topk(swing,nctx).indices
    tp=torch.linspace(2.5/a_g,7.0/a_g,40,device=dev,dtype=torch.float64)
    tm=torch.linspace(-7.0/a_g,-2.5/a_g,40,device=dev,dtype=torch.float64)
    tails=[]
    for mi in top.tolist():
        x0=X0[mi]
        with torch.no_grad():
            Fp=teacher(x0.unsqueeze(0)+tp.unsqueeze(1)*vj.unsqueeze(0)).cpu().numpy()
            Fm=teacher(x0.unsqueeze(0)+tm.unsqueeze(1)*vj.unsqueeze(0)).cpu().numpy()
        for r in range(O):
            tails.append((tp.cpu().numpy(),Fp[:,r],+1)); tails.append((tm.cpu().numpy(),Fm[:,r],-1))
    a_hat=fit_mag(tails,a_g); merr.append(abs(a_hat-float(W1t[j].norm())))
merr=torch.tensor(merr)
print(f"MAGNITUDE via dual-direction + multi-harmonic tail (P={P}, {nctx} contexts):")
print(f"  err: max {merr.max():.3e}  mean {merr.mean():.3e}  median {merr.median():.3e}   (old AAA: 1.7e-3)")
