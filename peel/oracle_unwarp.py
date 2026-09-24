"""ORACLE profile diagnostic (chat's unwarp idea). For candidate bias beta,
re-parametrize the black box by target activation h: t_beta(h)=(logit(h)-beta)/a.
If beta=b*, the true first-layer activation IS h, so the curves obey
Y_m(h)=R_theta(sigma(alpha h + beta_m)) exactly (affine in h). A wrong beta injects
Mobius warp T_delta(h)=sigma(logit(h)+delta), NOT representable by the known
downstream family with affine h. Profile loss L(beta)=min_nuisance ||Y - Rhat||.
Plot L vs |beta-b*|. Clean minimum resolving 1e-4 => bias is crackable."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1=teacher.layers[0].weight.detach(); b1=teacher.layers[0].bias.detach()
W2=teacher.layers[1].weight.detach(); b2=teacher.layers[1].bias.detach()   # 32x24, 32
W3=teacher.layers[2].weight.detach(); b3=teacher.layers[2].bias.detach()
W4=teacher.layers[3].weight.detach(); b4=teacher.layers[3].bias.detach()
N=W1/W1.norm(dim=1,keepdim=True); V=N.t()@torch.linalg.inv(N@N.t()); W1pinv=W1.t()@torch.linalg.inv(W1@W1.t())
g=torch.Generator(device=dev).manual_seed(3)

j=0; a=float(W1[j].norm()); b_star=float(b1[j]); vj=V[:,j]
M=16; H=160
hh=torch.linspace(0.02,0.98,H,device=dev,dtype=torch.float64); logit=torch.log(hh/(1-hh))
# contexts: other pre-activations = zc_k; need w_j . x_m = 0  (so h_j=h at beta=b*)
zc=(2*torch.rand(M,k,generator=g,device=dev,dtype=torch.float64)-1)*2.5
target=zc-b1; target[:,j]=0.0                           # W1 x_m = target, so w_j.x_m=0
Xm=target@W1pinv.t()
hmj=torch.sigmoid(zc)                                   # other activations sigma(z_k)  (M,k)
# beta_m* = W2[:,-j] @ h_-j + b2  (drop column j)
mask=torch.ones(k,dtype=torch.bool); mask[j]=False
beta_star=(hmj[:,mask])@(W2[:,mask].t())+b2             # (M,32)
alpha_star=W2[:,j].clone()                              # (32,)

def query(beta):                                        # Y_m,beta(h): (M,H,O)
    t=(logit[None,:]-beta)/a                            # (M? ,H) -> broadcast per context
    Y=[]
    for m in range(M):
        x=Xm[m].unsqueeze(0)+t[0].unsqueeze(1)*vj.unsqueeze(0)  # t same across m
        with torch.no_grad(): Y.append(teacher(x))
    return torch.stack(Y)                               # (M,H,O)
# t depends only on h (not m); precompute
t_of=lambda beta:(logit-beta)/a
def query_beta(beta):
    t=t_of(beta); pts=Xm.unsqueeze(1)+t[None,:,None]*vj[None,None,:]   # (M,H,d)
    with torch.no_grad(): return teacher(pts.reshape(-1,d)).reshape(M,H,-1)

def reduced_forward(alpha,beta_m,w3,b3_,w4,b4_):
    z2=alpha[None,None,:]*hh[None,:,None]+beta_m[:,None,:]   # (M,H,32)
    a2=torch.sigmoid(z2); a3=torch.sigmoid(a2@w3.t()+b3_)
    return a3@w4.t()+b4_                                     # (M,H,8)

def profile_loss(beta):
    Y=query_beta(beta)
    alpha=alpha_star.clone().requires_grad_(True); bm=beta_star.clone().requires_grad_(True)
    w3=W3.clone().requires_grad_(True); bb3=b3.clone().requires_grad_(True)
    w4=W4.clone().requires_grad_(True); bb4=b4.clone().requires_grad_(True)
    ps=[alpha,bm,w3,bb3,w4,bb4]
    opt=torch.optim.LBFGS(ps,lr=1.0,max_iter=120,history_size=60,line_search_fn='strong_wolfe')
    def cl():
        opt.zero_grad(); L=((reduced_forward(alpha,bm,w3,bb3,w4,bb4)-Y)**2).mean(); L.backward(); return L
    for _ in range(12): opt.step(cl)
    with torch.no_grad(): return float(((reduced_forward(alpha,bm,w3,bb3,w4,bb4)-Y)**2).mean())

print(f"neuron {j}: a={a:.3f} b*={b_star:.4f}\nprofile loss L(beta) vs delta=beta-b*:")
for delta in (0.0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2):
    L=profile_loss(b_star+delta)
    print(f"  delta={delta:+.0e}:  L = {L:.3e}")
