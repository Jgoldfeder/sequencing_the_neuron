"""Test chat's BIAS solution: freeze the (recovered) first-layer weights, then fit
the downstream G_theta + first-layer bias beta jointly on black-box queries sampled
in first-layer coordinates. Only k bias scalars are free in layer 1. Does beta
recover past the ~6e-2 wall? Test with W1 EXACT+frozen to isolate the bias question."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
W1pinv=W1t.t()@torch.linalg.inv(W1t@W1t.t())
g=torch.Generator(device=dev).manual_seed(1)

b_hat=b1t+0.08*(2*torch.rand(k,generator=g,device=dev,dtype=torch.float64)-1)   # bias guess, 8e-2 off
print(f"bias guess error (init): {float((b_hat-b1t).abs().max()):.3e}")

# --- queries in first-layer coords: x = W1^+ (r - b_hat), r ~ N(0, tau^2) ---
N=16384; tau=2.5
r=torch.randn(N,k,generator=g,device=dev,dtype=torch.float64)*tau
X=(r-b_hat)@W1pinv.t()
with torch.no_grad(): Y=teacher(X)

# --- surrogate: W1 frozen=true, beta trainable (init b_hat), downstream random ---
stu=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    stu.layers[0].weight.copy_(W1t); stu.layers[0].bias.copy_(b_hat)
stu.layers[0].weight.requires_grad_(False)          # freeze W1
beta=stu.layers[0].bias                             # trainable
down=[p for l in stu.layers[1:] for p in l.parameters()]
opt=torch.optim.Adam([{"params":down,"lr":3e-3},{"params":[beta],"lr":1e-3}])
Xg=X; Yg=Y
for step in range(8001):
    idx=torch.randint(0,N,(1024,),generator=g,device=dev)
    loss=((stu(Xg[idx])-Yg[idx])**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): beta.clamp_(b_hat-0.08, b_hat+0.08)   # stay in eps-ball
    if step in (0,2000,4000,6000,8000):
        be=float((stu.layers[0].bias.detach()-b1t).abs().max())
        print(f"  step {step:4d}: loss {float(loss):.2e}   bias max_err {be:.3e}")
