import sys, torch
from torch.func import jacrev, vmap
from scipy.optimize import linear_sum_assignment
torch.set_default_dtype(torch.float64); dev="cuda"
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
pk=torch.load("peel_committee.pt",map_location=dev,weights_only=False); dims=pk["dims"]
tt=MLP(dims,act="sigmoid").to(dev).double(); tt.load_state_dict({k:v.double() for k,v in pk["teacher_state"].items()}); tt.eval()
W2s=tt.layers[1].weight.detach(); b2=tt.layers[1].bias.detach()
W3=tt.layers[2].weight.detach(); b3=tt.layers[2].bias.detach(); W4=tt.layers[3].weight.detach(); b4=tt.layers[3].bias.detach()
W5=tt.layers[4].weight.detach(); b5=tt.layers[4].bias.detach()
B=torch.linalg.svd(W2s,full_matrices=False)[2][:80]; Astar=W2s@B.t()
def outh(h):
    s2=torch.sigmoid(h@W2s.t()+b2); s3=torch.sigmoid(s2@W3.t()+b3); s4=torch.sigmoid(s3@W4.t()+b4); return s4@W5.t()+b5
Jb=vmap(jacrev(outh))
def sig1(z): s=torch.sigmoid(z); return s*(1-s)
def werr(A):
    W2=A@B;Cp=torch.cdist(W2,W2s);Cm=torch.cdist(-W2,W2s);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);return float((torch.tensor([C[ri[i],ci[i]] for i in range(80)])/W2s.norm(dim=1)[ci].cpu()).mean())
cf=torch.load("consensus_full.pt",map_location=dev,weights_only=False); W2c=cf["consensus"]["layers.1.weight"].double()
Cp=torch.cdist(W2s,W2c); Cm=torch.cdist(W2s,-W2c); C=torch.minimum(Cp,Cm)
ri,ci=linear_sum_assignment(C.cpu().numpy()); ci=torch.tensor(ci,device=dev)
sgn=torch.where(Cm[torch.arange(80),ci]<Cp[torch.arange(80),ci],-1.0,1.0); A=((W2c[ci]*sgn[:,None])@B.t()).clone()
NB=400; gh=torch.Generator(device=dev).manual_seed(7); H=torch.sigmoid(torch.randn(NB,128,generator=gh,device=dev)*1.0).clamp(2e-2,1-2e-2)
U=H@B.t(); Q=(Jb(H)@B.t())
def Kof(A): Dp=sig1(U@A.t()+b2); Ainv=torch.linalg.inv(A); return torch.einsum('nok,kj->noj',Q,Ainv)/Dp[:,None,:]
def Lobj(A): sv=torch.linalg.svdvals(Kof(A).reshape(NB*10,80)); return (sv[40:]**2).sum()/(sv**2).sum()
print(f"start: L {float(Lobj(A)):.4e}  werr {werr(A)*100:.4f}%",flush=True)
eta=1e-2; traj=[]
for it in range(3001):
    Ar=A.clone().requires_grad_(True); L=Lobj(Ar); g,=torch.autograd.grad(L,Ar); g=g.detach(); L0=float(L)
    traj.append((L0,werr(A)*100))
    if it in (0,25,50,100,200,400,800,1200,1800,2400,3000):
        print(f"  it{it:4d}: L {L0:.4e}  werr {werr(A)*100:.5f}%  eta {eta:.1e}",flush=True); torch.save({"traj":traj,"A":A.cpu()},"rank40_gd_traj.pt")
    # backtracking line search: try growing then shrinking around current eta
    best=None
    for e in (eta*4,eta*2,eta,eta/2,eta/4,eta/8,eta/16,eta/64,eta/256):
        Ln=float(Lobj(A-e*g))
        if Ln<L0 and (best is None or Ln<best[1]): best=(e,Ln)
    if best is None: eta=eta/4;  # no decrease found; shrink and retry next iter
    else:
        A=A-best[0]*g; eta=best[0]
    if eta<1e-14: print("eta underflow stop",flush=True); break
print("DONE",flush=True)
