"""ITEM C normal-space mining: reverse-engineer the invariant from the oracle.
A_perp=(I-P_[B,C])A has rank 6. Its left singular vectors u_r are the local architecture
invariants (u_r^T B=0, u_r^T C=0, u_r^T A!=0). Reshape each u_r into probe x Hoff (P,15)
and probe x Tdist (P,20); SVD to test separability/rank; inspect the index pattern v_ij.
Whiten H,T by RMS. Repeat over downstream nets -> look for a STABLE structural pattern."""
import torch, itertools, numpy as np
from torch.func import jacrev, jacfwd
torch.set_default_dtype(torch.float64)
k,W,m=6,8,4; P=16
hoff=[(i,j) for i in range(k) for j in range(i+1,k)]
tdist=[(i,j,l) for i in range(k) for j in range(i+1,k) for l in range(j+1,k)]
nH,nT=len(hoff),len(tdist); D=nH+nT
def build(seed):
    g=torch.Generator().manual_seed(seed)
    W2=torch.randn(W,k,generator=g)*0.9; b2=(2*torch.rand(W,generator=g)-1)*0.6
    W3=torch.randn(m,W,generator=g)*0.7; b3=(2*torch.rand(m,generator=g)-1)*0.5
    w4=torch.randn(m,generator=g)*0.8;  b4=(2*torch.rand(1,generator=g)-1)*0.3
    beta=(2*torch.rand(k,generator=g)-1)*0.5
    th=torch.cat([beta,W2.reshape(-1),b2,W3.reshape(-1),b3,w4,b4])
    gp=torch.Generator().manual_seed(seed+50); cs=[(2*torch.rand(k,generator=gp)-1)*1.3 for _ in range(P)]
    return th,cs
def unpack(th):
    i=k; W2_=th[i:i+W*k].reshape(W,k); i+=W*k; b2_=th[i:i+W]; i+=W
    W3_=th[i:i+m*W].reshape(m,W); i+=m*W; b3_=th[i:i+m]; i+=m; w4_=th[i:i+m]; i+=m; b4_=th[i:i+1]
    return W2_,b2_,W3_,b3_,w4_,b4_
def jetclean(th,c):
    beta_=th[:k]; W2_,b2_,W3_,b3_,w4_,b4_=unpack(th)
    def Ft(t):
        s=torch.sigmoid(c+beta_+t); q=torch.sigmoid(W2_@s+b2_); h=torch.sigmoid(W3_@q+b3_); return (w4_@h+b4_).squeeze()
    G2=jacfwd(jacfwd(Ft))(torch.zeros(k)); G3=jacfwd(jacfwd(jacfwd(Ft)))(torch.zeros(k))
    return torch.cat([torch.stack([G2[i,j] for (i,j) in hoff]),torch.stack([G3[i,j,l] for (i,j,l) in tdist])])
def analyze(seed):
    th,cs=build(seed)
    Jv=np.array([jetclean(th,c).detach().numpy() for c in cs])           # (P,D)
    Jac=np.stack([jacrev(lambda t: jetclean(t,c))(th).detach().numpy() for c in cs])  # (P,D,ntheta)
    # whiten by RMS per block
    wH=1.0/np.sqrt((Jv[:,:nH]**2).mean()); wT=1.0/np.sqrt((Jv[:,nH:]**2).mean())
    wt=np.concatenate([np.full(nH,wH),np.full(nT,wT)])
    Jvw=Jv*wt[None,:]; Jacw=Jac*wt[None,:,None]
    A=Jacw[:,:,:k].reshape(P*D,k); B=Jacw[:,:,k:].reshape(P*D,Jac.shape[2]-k)
    C=np.zeros((P*D,P*k))
    for p in range(P):
        for i in range(k):
            col=np.zeros(D)
            for idx,(a,b) in enumerate(hoff):
                if i in (a,b): col[idx]=Jvw[p,idx]
            for idx,(a,b,l) in enumerate(tdist):
                if i in (a,b,l): col[nH+idx]=Jvw[p,nH+idx]
            C[p*D:(p+1)*D,p*k+i]=col
    N=np.concatenate([B,C],1); Aperp=A-N@np.linalg.lstsq(N,A,rcond=None)[0]
    U,S,Vt=np.linalg.svd(Aperp,full_matrices=False)
    print(f' seed {seed}: Aperp singular values = {np.array2string(S,precision=2)}')
    ranks=[]
    for r in range(k):
        ur=U[:,r].reshape(P,D); UH=ur[:,:nH]; UT=ur[:,nH:]
        svH=np.linalg.svd(UH,compute_uv=False); svT=np.linalg.svd(UT,compute_uv=False)
        rH=int((svH>0.05*svH.max()).sum()); rT=int((svT>0.05*svT.max()).sum())
        ranks.append((rH,rT))
    print(f'          (rank U_H, rank U_T) per singular vector: {ranks}')
    # inspect the index pattern of the strongest separable component of u_0
    ur=U[:,0].reshape(P,D); UH=ur[:,:nH]
    uh,sh,vh=np.linalg.svd(UH,full_matrices=False); vij=vh[0]            # dominant Hoff index pattern (15,)
    M=np.zeros((k,k))
    for idx,(a,b) in enumerate(hoff): M[a,b]=vij[idx]; M[b,a]=vij[idx]
    svM=np.linalg.svd(M,compute_uv=False)
    print(f'          u_0 dominant Hoff index-matrix singular values = {np.array2string(svM,precision=2)} (rank1 => v_ij=x_i x_j)')
    return ranks
print('normal-space mining across downstream networks:')
for sd in (3,7,11): analyze(sd)
