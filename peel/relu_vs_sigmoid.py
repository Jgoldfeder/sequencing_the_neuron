"""Does the rank certificate distinguish the true layer-n weights?  sigmoid vs ReLU.
Small random funnel net.  Seal at layer-0 activations, take B = true rowspace of W1,
and measure the certificate's bottom-singular energy as A moves away from A_true.
If it RISES, the certificate is informative (weights identifiable). If it stays ~0,
the certificate is degenerate (weights NOT identifiable)."""
import torch
torch.set_default_dtype(torch.float64); dev="cuda"
torch.manual_seed(0)
dims=[24,16,10,6]; n=1                      # solve W1 (16->10); next 10->6, d_next=6<d_n=10
d_prev,d_n,d_next=dims[n],dims[n+1],dims[n+2]; d_out=dims[-1]; bottom=d_n-d_next

# random weights; large positive biases so ReLU probes sit in the all-active (linear) region
W=[torch.randn(dims[i+1],dims[i],device=dev)*0.5 for i in range(3)]
b=[torch.full((dims[i+1],),1.5,device=dev) for i in range(3)]
def logit(h): return torch.log(h/(1-h))
def sig1(z): s=torch.sigmoid(z); return s*(1-s)

def run(act):
    g   = (lambda z: torch.sigmoid(z)) if act=="sig" else (lambda z: torch.relu(z))
    gp  = (lambda z: sig1(z))          if act=="sig" else (lambda z: (z>0).double())   # activation derivative
    ginv= (lambda a: logit(a))         if act=="sig" else (lambda a: a)                 # invert on active region
    def BB(x):
        a=x
        for i in range(3): a=g(a@W[i].t()+b[i])
        return a
    # seal at layer-0 activations:  x_of_h inverts layer 0
    W0p=W[0].t()@torch.linalg.inv(W[0]@W[0].t())
    def BBh(h): return BB((ginv(h)-b[0])@W0p.t())
    # probes in the active region (h>0, and for sigmoid in (0,1))
    if act=="sig": H=torch.sigmoid(torch.randn(300,d_prev,device=dev)).clamp(2e-2,1-2e-2)
    else:          H=(torch.rand(300,d_prev,device=dev)*0.6+0.3)                          # in (0.3,0.9)>0
    # finite-diff sealed Jacobian (300,d_out,d_prev)
    fd=1e-5; E=torch.eye(d_prev,device=dev)*fd
    Hp=(H[:,None,:]+E[None]).reshape(-1,d_prev); Hm=(H[:,None,:]-E[None]).reshape(-1,d_prev)
    if act=="sig": Hp=Hp.clamp(1e-4,1-1e-4); Hm=Hm.clamp(1e-4,1-1e-4)
    J=((BBh(Hp).reshape(300,d_prev,d_out)-BBh(Hm).reshape(300,d_prev,d_out))/(2*fd)).permute(0,2,1)
    # true rowspace B of W1, and the certificate ingredients
    B=torch.linalg.svd(W[n],full_matrices=False)[2][:d_n]     # d_n x d_prev
    U=H@B.t(); Q=J@B.t()                                      # (300,d_n), (300,d_out,d_n)
    A_true=W[n]@B.t(); b_n=b[n]
    def botE(A):
        Dp=gp(U@A.t()+b_n).clamp_min(1e-9)                    # divide by activation derivative
        K=torch.einsum('nok,kj->noj',Q,torch.linalg.inv(A))/Dp[:,None,:]
        sv=torch.linalg.svdvals(K.reshape(300*d_out,d_n))
        return float((sv[d_next:]**2).sum()/(sv**2).sum())
    # move A along a random (small, activity-preserving) direction away from truth
    torch.manual_seed(1); Dir=torch.randn_like(A_true); Dir=Dir/Dir.norm()*A_true.norm()
    print(f"  [{act}] bottom-{bottom} energy vs distance from A_true:")
    for t in [0.0,0.02,0.05,0.10,0.20]:
        print(f"      t={t:.2f}: {botE(A_true+t*Dir):.3e}")

print("=== SIGMOID net ==="); run("sig")
print("=== ReLU net ===");    run("relu")
