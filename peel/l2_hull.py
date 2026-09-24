"""POPULATION AFFINE-SPAN diagnostic (oracle-only). Does the committee already span a direction
toward true W2? For each true neuron j, the 8 aligned member rows w_j^(m) span a <=7-dim affine hull;
find the oracle-best point in it. Report per-representation the best achievable W2 row error.
Representations: (1) committee median, (2) ONE global affine combo of the 8 members (shared coeffs),
(3) per-row affine combo, (4) per-row affine + row-space(B) projection.  Truth allowed (diagnostic).
"""
import sys, torch
torch.set_default_dtype(torch.float64)
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
dev="cuda"; CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
pop=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False)
dims=pop["dims"]
t=MLP(dims,act="sigmoid").to(dev).double(); t.load_state_dict(pop["teacher_state"]); t.eval()
W2t=t.layers[1].weight.detach(); nt=W2t.norm(dim=1)
Uw,Sw,Vh=torch.linalg.svd(W2t,full_matrices=False); B=Vh[:80]; Pb=B.t()@B
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False)
mem=[{kk:vv.to(dev).double() for kk,vv in sd.items()} for sd in pk["pop_states"]]
def match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r);c=torch.tensor(c);return r,c,torch.where(Cm[r,c]<Cp[r,c],-1.,1.)
# align every member to TRUE order+sign -> per-true-neuron candidates
al=[]
for sd in mem:
    W2m=sd["layers.1.weight"]; r,c,s=match(W2m,W2t); Wa=torch.zeros_like(W2m); Wa[c]=W2m[r]*s[:,None]; al.append(Wa)
Wal=torch.stack(al)                                   # (8,80,128) aligned to truth
cand=Wal.permute(1,0,2)                               # (80,8,128): cand[j,m]=member m row for true neuron j
M=len(mem)
def report(tag,rec):
    rel=(rec-W2t).norm(dim=1)/nt
    print(f"{tag:34s} mean {float(rel.mean()):.3e} max {float(rel.max()):.3e} | <1%:{int((rel<0.01).sum()):>2} <3%:{int((rel<0.03).sum()):>2} <5%:{int((rel<0.05).sum()):>2} /80")
# (1) median
report("(1) committee median", Wal.median(0).values)
# (2) ONE global affine combo (shared alpha over 8 members, sum=1), oracle-fit to all rows
Mflat=Wal.reshape(M,-1).t()                           # (10240, 8)
ones=torch.ones(M,device=dev); a0=ones/M
_,_,Vh8=torch.linalg.svd(ones.reshape(1,M)); Z=Vh8[1:].t()   # (8,7) basis of sum=0
AZ=Mflat@Z; rhs=W2t.reshape(-1)-Mflat@a0
beta=torch.linalg.lstsq(AZ,rhs).solution; ag=a0+Z@beta
rec_glob=(Wal*ag[:,None,None]).sum(0)
report("(2) global affine (shared coeffs)", rec_glob)
# (3) per-row affine combo (each row its own alpha, sum=1)
A=cand.transpose(1,2)                                  # (80,128,8)
Aa0=A@a0                                               # (80,128)
AZj=A@Z                                                # (80,128,7)
rhsj=W2t-Aa0
betaj=torch.linalg.lstsq(AZj,rhsj).solution           # (80,7)
alphaj=a0+ (betaj@Z.t())                              # (80,8)
rec_row=torch.einsum('jdc,jc->jd',A,alphaj)
report("(3) per-row affine", rec_row)
# (4) per-row affine + row-space projection
report("(4) per-row affine + B-projection", rec_row@Pb)
# reference: what does the row space alone give the median?
report("    [ref] median + B-projection", Wal.median(0).values@Pb)
print("\ninterpretation: if (3)/(4) put most rows <5%, the population spans the bridge into the basin.")
