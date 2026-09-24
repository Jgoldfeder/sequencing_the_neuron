"""STRUCTURED bias EXTRACTOR (tensor route), clean case W<k, deep tail.
Germ Hessian G_ij=W2^T S W2 has rank<=W (S folds ALL deeper layers). With W<k the
off-diagonals over-determine the rank-W structure => reconstruct the DIAGONAL G_ii by
rank-W completion. Then repeated-vs-distinct mismatch:
   F_ii = G_ii sdot_i^2 + G_i sddot_i,  F_i = G_i sdot_i
   => (F_ii - G_ii sdot_i^2)/F_i = sddot_i/sdot_i = 1-2 s_i   (dual coords, a=1)
=> s_i => b_i. NO downstream fit, NO optimization over the deep net; just linear
algebra on the measured jet + the bias guess to iterate s_i. Deep tail present."""
import torch, numpy as np
torch.set_default_dtype(torch.float64)

def build(dims, seed):
    g=torch.Generator().manual_seed(seed)
    Ws=[(torch.randn(dims[i+1],dims[i],generator=g))/np.sqrt(dims[i]) for i in range(len(dims)-1)]
    bs=[(2*torch.rand(dims[i+1],generator=g)-1)*0.6 for i in range(len(dims)-1)]
    d,k=dims[0],dims[1]; W1=Ws[0]; b1=bs[0]
    V=W1.t()@torch.linalg.inv(W1@W1.t())            # W1 @ V = I  (dual coords: dz_i/dt_i=1)
    x0=torch.zeros(d)
    def F(t):                                        # t: (k,) dual coords
        x=x0+V@t; h=torch.sigmoid(W1@x+b1)
        for l in range(1,len(Ws)):
            z=Ws[l]@h+bs[l]; h=torch.sigmoid(z) if l<len(Ws)-1 else z
        return h.squeeze()
    z0=(W1@x0+b1)                                    # z_i^0 ; b_i recovered as z_i^0 - W1_i x0 (=z0 here since x0=0)
    return F, b1, W1, x0, k

def grad_hess(F,k):
    t=torch.zeros(k,requires_grad=True)
    f=F(t); g=torch.autograd.grad(f,t,create_graph=True)[0]
    H=torch.zeros(k,k)
    for i in range(k):
        H[i]=torch.autograd.grad(g[i],t,retain_graph=True)[0]
    return g.detach().numpy(), H.detach().numpy()

def complete_diag(G_off, W, iters=400):
    """rank-W symmetric completion: off-diagonals fixed, recover diagonal."""
    k=G_off.shape[0]; G=G_off.copy();
    for _ in range(iters):
        wv,U=np.linalg.eigh(G)
        keep=np.argsort(-np.abs(wv))[:W]
        Glr=(U[:,keep]*wv[keep])@U[:,keep].T
        newdiag=np.diag(Glr).copy()
        G=G_off.copy(); np.fill_diagonal(G,newdiag)
    return np.diag(G).copy()

def recover(dims, seed, guess_noise=0.05):
    F,b1,W1,x0,k=build(dims,seed); W=dims[2]
    g,H=grad_hess(F,k)                                 # measured jet: F_i=g, F_ij=H
    z0=(W1@x0+b1).numpy()                              # true z_i^0
    b_true=b1.numpy()
    rng=np.random.default_rng(seed+123)
    z=z0+guess_noise*(2*rng.random(k)-1)              # GUESS of z_i^0 (= bias guess)
    for it in range(60):
        s=1/(1+np.exp(-z)); sd=s*(1-s); sdd=sd*(1-2*s)
        # germ Hessian off-diagonals from measured F_ij (need s via current guess)
        Goff=np.zeros((k,k))
        for i in range(k):
            for j in range(k):
                if i!=j: Goff[i,j]=H[i,j]/(sd[i]*sd[j])
        Gii=complete_diag(Goff, W)                     # reconstruct diagonal via rank-W
        mu=(np.diag(H) - Gii*sd**2)/g                  # = 1-2 s_i
        s_new=np.clip((1-mu)/2,1e-6,1-1e-6)
        z=np.log(s_new/(1-s_new))                      # update z_i^0
    b_rec=z                                            # since x0=0, b_i=z_i^0
    return np.abs(b_rec-b_true).max(), guess_noise

for dims in [[8,6,3,1],[8,6,3,3,1],[10,8,4,4,1]]:
    errs=[recover(dims,s)[0] for s in range(6)]
    errs=np.array(errs)
    print(f"{'->'.join(map(str,dims))} (k={dims[1]},W={dims[2]}): recovered bias err "
          f"median {np.median(errs):.2e} worst {errs.max():.2e} | guess was ~5e-2")
