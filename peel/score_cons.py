import torch
from scipy.optimize import linear_sum_assignment
PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
d=torch.load(PEEL+"cons_out.pt",map_location="cpu",weights_only=False)
pk=torch.load(PEEL+"peel_committee.pt",map_location="cpu",weights_only=False)
W2t=pk["teacher_state"]["layers.1.weight"].double(); nt=W2t.norm(dim=1)
def err(W):
    Cp=torch.cdist(W.double(),W2t);Cm=torch.cdist(-W.double(),W2t);C=torch.minimum(Cp,Cm).numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);return float((e/nt[ci]).mean())
T=d["traj"]; errs=[err(W) for W in T]; n=len(T)
print(f"CONSENSUS start {errs[0]*100:.2f}% -> iter{n-1} {errs[-1]*100:.2f}% (best {min(errs)*100:.2f}%)")
for i in range(0,n,5): print(f"  it{i}: {errs[i]*100:.2f}%")
print("  last5:", " ".join(f"{errs[i]*100:.2f}" for i in range(max(0,n-5),n)))
