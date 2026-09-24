"""HONEST L2 direction refinement via geometric DEFLATION (no downstream, no saturation, in-cube).

At an in-cube point h, the black-box FD Jacobian is J(h)=D diag(sigma') W2 (D=unknown downstream).
Project J onto the complement of the GUESSED other rows {W2g[i]: i!=j}: the other neurons' input-side
contributions cancel, leaving a rank-1 map in the direction of (W2[j] deflated). Its top right singular
vector is a REFINED estimate of neuron j's dual direction v_j = colnorm(W2^+ e_j).

Reports per-neuron ANGLE to the TRUE dual direction for:
  guess  : dual dir from committee guess (no refinement)
  refine : deflation using GUESSED other rows  <- the honest method
  sanity : deflation using TRUE other rows (mechanism check; uses truth, not an honest result)
Idealization: h set exactly (L1 assumed solved); oracle = true subnetwork layers 2..5, forward-query only.
"""
import sys, time, torch, numpy as np
torch.set_default_dtype(torch.float32)
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
dev="cuda"; P="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
pk=torch.load(P+"peel_committee.pt",map_location=dev,weights_only=False); dims=[784,128,80,40,32,10]
_t=MLP(dims,act="sigmoid").to(dev); _t.load_state_dict(pk["teacher_state"]); _t.eval()
for p in _t.parameters(): p.requires_grad_(False)

class BB:
    def __init__(self,net):
        Wt=[net.layers[i].weight.detach().float() for i in range(5)]; bt=[net.layers[i].bias.detach().float() for i in range(5)]
        def run(h):
            q2=torch.sigmoid(h@Wt[1].t()+bt[1]);q3=torch.sigmoid(q2@Wt[2].t()+bt[2])
            q4=torch.sigmoid(q3@Wt[3].t()+bt[3]);return q4@Wt[4].t()+bt[4]
        object.__setattr__(self,"_run",run); object.__setattr__(self,"nq",0)
    def query(self,h):
        object.__setattr__(self,"nq",self.nq+h.shape[0])
        with torch.no_grad(): return self._run(h)
    def __getattr__(self,n): raise AttributeError(f"blocked: {n}")
bb=BB(_t)
W2t=_t.layers[1].weight.detach().float(); b2t=_t.layers[1].bias.detach().float()
Vt=torch.linalg.pinv(W2t); Vt=Vt/Vt.norm(dim=0,keepdim=True)

def match(A,B):
    Cp=torch.cdist(A,B);Cm=torch.cdist(-A,B);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r);c=torch.tensor(c);s=torch.where(Cm[r,c]<Cp[r,c],-1.,1.);return r,c,s
mem=[{k:v.to(dev).float() for k,v in sd.items()} for sd in pk["pop_states"]]
ref=mem[0]["layers.1.weight"]; A=[ref.clone()]
for sd in mem[1:]:
    Wm=sd["layers.1.weight"];r,c,s=match(Wm,ref);Wa=torch.zeros_like(Wm);Wa[c]=Wm[r]*s[:,None];A.append(Wa)
A=torch.stack(A); W2g=A.median(0).values.clone(); b2g=mem[0]["layers.1.bias"].clone()
Vg=torch.linalg.pinv(W2g); Vg=Vg/Vg.norm(dim=0,keepdim=True); W2gp=torch.linalg.pinv(W2g)

def angles(Vrec):
    Cabs=(Vrec.t()@Vt).abs(); r,c=linear_sum_assignment((1-Cabs).cpu().numpy())
    cos=Cabs[r,c].clamp(0,1); return torch.rad2deg(torch.arccos(cos))
def deflate(rows_for_proj, base_pts=8, fd=1e-3):
    Vrec=torch.zeros(128,80,device=dev); gg=torch.Generator(device="cpu").manual_seed(0)
    for j in range(80):
        Qb,_=torch.linalg.qr(rows_for_proj[j].t())
        G=torch.zeros(128,128,device=dev)
        for _ in range(base_pts):
            t=(torch.randn(80,generator=gg)*2.0).to(dev); t[j]=0.0
            h0=torch.clamp(W2gp@(t-b2g),1e-3,1-1e-3); E=torch.eye(128,device=dev)
            Hp=(h0+fd*E).clamp(1e-6,1-1e-6); Hm=(h0-fd*E).clamp(1e-6,1-1e-6)
            J=((bb.query(Hp)-bb.query(Hm))/(2*fd)).t()
            JP=J-(J@Qb)@Qb.t(); G+=JP.t()@JP
        w,V=torch.linalg.eigh(G); Vrec[:,j]=V[:,-1]
    return Vrec
guess_rows=[torch.cat([W2g[:j],W2g[j+1:]],0) for j in range(80)]
true_rows =[torch.cat([W2t[:j],W2t[j+1:]],0) for j in range(80)]
t0=time.time(); ag=angles(Vg)
print(f"guess (no refine)      dual-dir angle deg: mean {float(ag.mean()):.2f}  median {float(ag.median()):.2f}  max {float(ag.max()):.2f}")
Vr=deflate(guess_rows); ar=angles(Vr)
print(f"refine (deflate guess) dual-dir angle deg: mean {float(ar.mean()):.2f}  median {float(ar.median()):.2f}  max {float(ar.max()):.2f}  [improved {int((ar<ag).sum())}/80]")
Vs=deflate(true_rows);  aS=angles(Vs)
print(f"sanity (deflate TRUE)  dual-dir angle deg: mean {float(aS.mean()):.2f}  median {float(aS.median()):.2f}  max {float(aS.max()):.2f}   <- mechanism check")
print(f"[oracle queries: {bb.nq}]  [{time.time()-t0:.1f}s]")
