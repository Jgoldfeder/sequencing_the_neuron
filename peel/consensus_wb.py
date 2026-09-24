"""Consensus accuracy at layer 2, WEIGHT and BIAS reported SEPARATELY.
Usage: python3 consensus_wb.py [committee.pt]   (default: original pop)
For each eps: cluster members per neuron (teacher-free, within eps of the per-neuron mean),
call it consensus if >= quorum agree, then score the cluster-average vs true -- W and b apart."""
import sys, torch, numpy as np
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from scipy.optimize import linear_sum_assignment
path = sys.argv[1] if len(sys.argv) > 1 else "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt"
pk = torch.load(path, map_location="cpu", weights_only=False)
ts = pk["teacher_state"]; W1t=ts["layers.0.weight"]; W2t=ts["layers.1.weight"]; b2t=ts["layers.1.bias"]
nt=W2t.norm(dim=1); P=len(pk["pop_states"]); quorum=int(np.ceil(0.625*P))
def match(A,B):
    Cp=torch.cdist(A,B); Cm=torch.cdist(-A,B); C=torch.minimum(Cp,Cm); r,c=linear_sum_assignment(C.numpy())
    s=torch.where(Cm[r,c]<Cp[r,c],-1.,1.); return torch.tensor(r),torch.tensor(c),s
def align(sd):
    W1m=sd["layers.0.weight"]; W2m=sd["layers.1.weight"]; b2m=sd["layers.1.bias"]
    r1,c1,s1=match(W1m,W1t); W2c=torch.zeros_like(W2m); W2c[:,c1]=W2m[:,r1]*s1[None,:]
    r2,c2,s2=match(W2c,W2t); W2a=torch.zeros_like(W2c); b2a=torch.zeros_like(b2m)
    W2a[c2]=W2c[r2]*s2[:,None]; b2a[c2]=b2m[r2]*s2; return W2a,b2a
AW=[]; AB=[]
for sd in pk["pop_states"]:
    w,b=align(sd); AW.append(w); AB.append(b)
AW=torch.stack(AW); AB=torch.stack(AB)                                  # (P,80,128),(P,80)
print(f"committee P={P} (quorum {quorum}).  true: mean||w||={float(nt.mean()):.2f}  mean|b2|={float(b2t.abs().mean()):.3f}")
print(f"{'eps':>6} {'L2 cons':>8} | {'WEIGHT rel-err mean/max':>26} | {'BIAS abs-err mean/max':>24}")
for eps in [0.05,0.1,0.2,0.3]:
    We,Be=[],[]
    for j in range(80):
        rel=(AW[:,j]-W2t[j]).norm(dim=1)/nt[j]                          # each member's weight rel-err vs true
        med=AW[:,j].median(0).values; d=(AW[:,j]-med).norm(dim=1)/nt[j]  # distance to per-neuron median (teacher-free)
        clus=d<eps
        if int(clus.sum())>=quorum:
            wj=AW[clus,j].mean(0); bj=AB[clus,j].mean()
            We.append(float((wj-W2t[j]).norm()/nt[j])); Be.append(float((bj-b2t[j]).abs()))
    if We:
        We=np.array(We); Be=np.array(Be)
        print(f"{eps:>6} {len(We):>4}/80 | W mean {We.mean():.3e} max {We.max():.3e} | b mean {Be.mean():.3e} max {Be.max():.3e}")
    else:
        print(f"{eps:>6}    0/80 | -- | --")
