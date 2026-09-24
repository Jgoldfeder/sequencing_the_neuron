import torch, numpy as np, itertools
from torch.func import jacfwd
torch.set_default_dtype(torch.float64)
k,W,m=6,8,4
g=torch.Generator().manual_seed(3)
W2=torch.randn(W,k,generator=g)*0.9; b2=(2*torch.rand(W,generator=g)-1)*0.6
W3=torch.randn(m,W,generator=g)*0.7; b3=(2*torch.rand(m,generator=g)-1)*0.5
w4=torch.randn(m,generator=g)*0.8; b4=(2*torch.rand(1,generator=g)-1)*0.3
def G(s):
    q=torch.sigmoid(W2@s+b2); h=torch.sigmoid(W3@q+b3); return (w4@h+b4).squeeze()
def sv(M): return np.linalg.svd(M,compute_uv=False)
for trial in range(3):
    s0=torch.rand(k,generator=g)*0.6+0.2
    G2=jacfwd(jacfwd(G))(s0).numpy()
    G3=jacfwd(jacfwd(jacfwd(G)))(s0).numpy()
    G4=jacfwd(jacfwd(jacfwd(jacfwd(G))))(s0).numpy()
    s2=sv(G2); 
    M3=G3.reshape(k,k*k); s3=sv(M3)
    M4=G4.reshape(k*k,k*k); s4=sv(M4)
    r=lambda s: int((s>s.max()*1e-9).sum())
    print(f"trial {trial}: (k={k},W={W},bottleneck m={m})")
    print(f"  G2 (6x6): rank {r(s2)}  sv={np.array2string(s2,precision=2)}")
    print(f"  G3 unfold (6x36): rank {r(s3)}  sv={np.array2string(s3,precision=2)}")
    print(f"  G4 unfold (36x36): rank {r(s4)}  sv[:10]={np.array2string(s4[:10],precision=2)}")
    print(f"     G4 sv gap? sv[3]/sv[4]={s4[3]/s4[4]:.1f}  sv[m-1]/sv[m]=sv[3]/sv[4], sv[W-1]/sv[W]=sv[7]/sv[8]={s4[7]/s4[8]:.1f}")
