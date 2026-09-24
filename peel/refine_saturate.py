"""Test the 'saturate the others' method on a DEEP sigmoid net.
For neuron j: use the whole first-layer guess to CONSTRUCT x with guessed
z_j = 0 and guessed z_k = +-S (large) for k != j -- solve W1g x + b1g = t.
Then the teacher's Jacobian at x should be ~rank-1 = c_j w_j^T (others' sigma'
suppressed exponentially by saturation). SVD -> refined w_j. Sweep S."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]        # 3 hidden layers
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True)
g=torch.Generator(device=dev).manual_seed(1)

@torch.no_grad()
def J_at(x,fd=1e-4):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()  # (O,d)

# whole first-layer guess, 5deg off
Wg=torch.empty(k,d,device=dev,dtype=torch.float64); bg=b1t.clone()
for kk in range(k):
    u=tn[kk]; v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
    Wg[kk]=(math.cos(math.radians(5))*u+math.sin(math.radians(5))*v)*W1t[kk].norm()
Wg_pinv = Wg.t() @ torch.linalg.inv(Wg @ Wg.t())    # min-norm right inverse (d,k)

def me(u,kk): s=1.0 if float(u@tn[kk])>0 else -1.0; return float((s*u/u.norm()-tn[kk]).abs().max())
def ang(u,kk): return math.degrees(math.acos(min(1.0,abs(float((u/u.norm())@tn[kk])))))

signs=(torch.randint(0,2,(k,),generator=g,device=dev)*2-1).double()
for S in (4.0, 8.0, 12.0, 20.0):
    befe=[]; afte=[]; befa=[]; afta=[]
    for j in range(k):
        t=signs*S; t[j]=0.0
        x=Wg_pinv @ (t - bg)                        # guessed z = t
        J=J_at(x)
        U,Sv,Vh=torch.linalg.svd(J,full_matrices=False)
        w=Vh[0]; w = w if float(w@Wg[j])>0 else -w
        befe.append(me(Wg[j],j)); afte.append(me(w,j))
        befa.append(ang(Wg[j],j)); afta.append(ang(w,j))
    print(f"S={S:>4}: guess ang med {sorted(befa)[k//2]:.3f}deg (max_eps {max(befe):.2e})  ->  "
          f"refined ang med {sorted(afta)[k//2]:.4f}deg  max_eps med {sorted(afte)[k//2]:.2e} max {max(afte):.2e}")
