"""PRACTICAL unwarp bias recovery: scan candidate bias beta, fit the reduced
nuisance (h->32->16->8, alpha shared, beta_m free) from RANDOM init with
continuation (warm-start each beta from the previous), take argmin L(beta).
No oracle downstream. Does it recover the bias, and to what precision?"""
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
N=W1/W1.norm(dim=1,keepdim=True); V=N.t()@torch.linalg.inv(N@N.t()); W1pinv=W1.t()@torch.linalg.inv(W1@W1.t())
g=torch.Generator(device=dev).manual_seed(3)
j=0; a=float(W1[j].norm()); b_star=float(b1[j]); vj=V[:,j]
M=12; H=120
hh=torch.linspace(0.02,0.98,H,device=dev,dtype=torch.float64); logit=torch.log(hh/(1-hh))
zc=(2*torch.rand(M,k,generator=g,device=dev,dtype=torch.float64)-1)*2.5
target=zc-b1; target[:,j]=0.0; Xm=target@W1pinv.t()

def query_beta(beta):
    t=(logit-beta)/a; pts=Xm.unsqueeze(1)+t[None,:,None]*vj[None,None,:]
    with torch.no_grad(): return teacher(pts.reshape(-1,d)).reshape(M,H,-1)

# reduced-model params (h -> z2=alpha*h+beta_m -> sig -> W3 -> sig -> W4)
def init_params():
    return [torch.randn(32,generator=g,device=dev,dtype=torch.float64)*0.5,       # alpha
            torch.randn(M,32,generator=g,device=dev,dtype=torch.float64)*0.5,      # beta_m
            (torch.randn(16,32,generator=g,device=dev,dtype=torch.float64)*math.sqrt(2/48)),
            torch.zeros(16,device=dev,dtype=torch.float64),
            (torch.randn(8,16,generator=g,device=dev,dtype=torch.float64)*math.sqrt(2/24)),
            torch.zeros(8,device=dev,dtype=torch.float64)]
def fwd(p):
    z2=p[0][None,None,:]*hh[None,:,None]+p[1][:,None,:]; a2=torch.sigmoid(z2)
    a3=torch.sigmoid(a2@p[2].t()+p[3]); return a3@p[4].t()+p[5]
def fit(Y,p,n_adam,n_lbfgs):
    p=[q.detach().clone().requires_grad_(True) for q in p]
    opt=torch.optim.Adam(p,lr=3e-3)
    for _ in range(n_adam):
        L=((fwd(p)-Y)**2).mean(); opt.zero_grad(); L.backward(); opt.step()
    opt2=torch.optim.LBFGS(p,lr=1.0,max_iter=100,history_size=60,line_search_fn='strong_wolfe')
    def cl(): opt2.zero_grad(); L=((fwd(p)-Y)**2).mean(); L.backward(); return L
    for _ in range(n_lbfgs): opt2.step(cl)
    with torch.no_grad(): return float(((fwd(p)-Y)**2).mean()), [q.detach() for q in p]

betas=torch.linspace(b_star-0.05, b_star+0.05, 21).tolist()   # scan (centered wide around b*)
p=init_params(); Ls=[]
for i,beta in enumerate(betas):
    Y=query_beta(beta)
    L,p=fit(Y,p, 4000 if i==0 else 800, 40 if i==0 else 25)    # continuation warm-start
    Ls.append(L)
    print(f"  beta-b*={beta-b_star:+.4f}: L={L:.3e}")
Ls=torch.tensor(Ls); i=int(Ls.argmin())
# parabolic refine
if 0<i<len(Ls)-1:
    y0,y1,y2=float(Ls[i-1]),float(Ls[i]),float(Ls[i+1]); den=y0-2*y1+y2
    dd=max(-1,min(1,0.5*(y0-y2)/den)) if abs(den)>1e-30 else 0.0
    b_rec=betas[i]+dd*(betas[1]-betas[0])
else: b_rec=betas[i]
print(f"\nrecovered bias error |b_rec - b*| = {abs(b_rec-b_star):.3e}   (fit floor L_min={float(Ls.min()):.2e})")
