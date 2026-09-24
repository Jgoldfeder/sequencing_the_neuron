import torch
from scipy.optimize import linear_sum_assignment
PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
d=torch.load(PEEL+"adv_sealed_out.pt",map_location="cpu",weights_only=False)
pk=torch.load(PEEL+"peel_committee.pt",map_location="cpu",weights_only=False)
W2t=pk["teacher_state"]["layers.1.weight"].double(); nt=W2t.norm(dim=1)
def err(W):
    Cp=torch.cdist(W.double(),W2t);Cm=torch.cdist(-W.double(),W2t);C=torch.minimum(Cp,Cm).numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);return float((e/nt[ci]).mean())
print(f"[score] start (member0) {err(d['start'])*100:.2f}%")
for probes in ["ordinary","adversarial"]:
    ws=" ".join(f"l={l}:{err(d[probes][l])*100:.2f}%" for l in (0.03,0.1,0.3,1.0))
    print(f"[score] {probes:12s} {ws}")
