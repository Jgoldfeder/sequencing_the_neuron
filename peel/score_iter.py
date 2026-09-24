import torch
from scipy.optimize import linear_sum_assignment
PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
d=torch.load(PEEL+"iter_out.pt",map_location="cpu",weights_only=False)
pk=torch.load(PEEL+"peel_committee.pt",map_location="cpu",weights_only=False)
W2t=pk["teacher_state"]["layers.1.weight"].double(); nt=W2t.norm(dim=1)
def err(W):
    Cp=torch.cdist(W.double(),W2t);Cm=torch.cdist(-W.double(),W2t);C=torch.minimum(Cp,Cm).numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);return float((e/nt[ci]).mean())
print("[score] W2 error per iteration (lambda chosen truth-free):")
for i,W in enumerate(d["traj"]):
    lam=d["lams"][i-1] if i>0 else "-"
    print(f"  iter {i:2d}: {err(W)*100:.2f}%   (lambda={lam})")
