"""Whole first-layer guess + KNOWN ARCHITECTURE (widths+sigmoid) of the deeper
layers, weights unknown. Refine layer 1 as hard as we can: joint distillation --
build a student with the known architecture, warm-start W1 from the (projected)
guess, random downstream, and fit the whole thing to the teacher's I/O. Measure
layer-1 max_eps before/after. This breaks the circularity: the downstream is now
fittable, not derived from the guess."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch, torch.nn as nn
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[64,12,20,12,6]; d=dims[0]; k=dims[1]           # 3 hidden layers, sigmoid
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True)
g=torch.Generator(device=dev).manual_seed(1)

@torch.no_grad()
def J_at(x,fd=1e-3):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()

def layer1_maxeps(W):
    errs=[]
    W=W.detach()
    for kk in range(k):
        u=W[kk]/W[kk].norm()
        # best-matching teacher neuron (|cos|), sign-aligned
        c=(tn@u); j=int(c.abs().argmax()); s=1.0 if float(c[j])>0 else -1.0
        errs.append(float((s*u-tn[j]).abs().max()))
    return max(errs), sorted(errs)[k//2]

# --- guess: whole first layer 5deg off, then row-space projection (free) ---
Wg=torch.empty(k,d,device=dev,dtype=torch.float64)
for kk in range(k):
    u=tn[kk]; v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
    Wg[kk]=(math.cos(math.radians(5))*u+math.sin(math.radians(5))*v)*W1t[kk].norm()
rows=[J_at(torch.randn(d,generator=g,device=dev,dtype=torch.float64)*2) for _ in range(6)]
_,_,Vh=torch.linalg.svd(torch.cat(rows,0),full_matrices=False); B=Vh[:k]
Wg=(Wg@B.t())@B
print("guess 5deg -> row-projected:  max_eps %.3e  median %.3e"%layer1_maxeps(Wg))

# --- joint distillation with KNOWN architecture, W1 warm-started ---
student=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    student.layers[0].weight.copy_(Wg); student.layers[0].bias.copy_(b1t)   # warm start L1
opt=torch.optim.Adam(student.parameters(),lr=3e-3)
for step in range(6000):
    x=torch.randn(512,d,device=dev,dtype=torch.float64)*1.5
    with torch.no_grad(): y=teacher(x)
    loss=((student(x)-y)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (0,1000,3000,5999):
        mx,md=layer1_maxeps(student.layers[0].weight)
        print(f"  distill step {step:4d}: loss {float(loss):.2e}  layer-1 max_eps {mx:.3e} (med {md:.3e})")
