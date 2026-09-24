import torch
from scipy.optimize import linear_sum_assignment
PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
res=torch.load(PEEL+"W2_est.pt",map_location="cpu",weights_only=False)
pk=torch.load(PEEL+"peel_committee.pt",map_location="cpu",weights_only=False)
W2t=pk["teacher_state"]["layers.1.weight"].double(); nt=W2t.norm(dim=1)
def err(W):
    Cp=torch.cdist(W.double(),W2t);Cm=torch.cdist(-W.double(),W2t);C=torch.minimum(Cp,Cm).numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);return float((e/nt[ci]).mean())
print(f"[score] guess: {err(res['W2_guess']):.3e}")
for it in sorted(res.get("snaps",{})): print(f"[score] after iter {it:2d}: {err(res['snaps'][it]):.3e}")
