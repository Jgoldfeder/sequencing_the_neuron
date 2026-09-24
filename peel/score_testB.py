import torch
from scipy.optimize import linear_sum_assignment
PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
d=torch.load(PEEL+"testB_out.pt",map_location="cpu",weights_only=False)
pk=torch.load(PEEL+"peel_committee.pt",map_location="cpu",weights_only=False)
W2t=pk["teacher_state"]["layers.1.weight"].double(); nt=W2t.norm(dim=1)
def err(W):
    Cp=torch.cdist(W.double(),W2t);Cm=torch.cdist(-W.double(),W2t);C=torch.minimum(Cp,Cm).numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);return float((e/nt[ci]).mean())
print(f"{'member':>7} {'start (proj)':>13} {'after step (warp-nuis)':>23} {'truth-ward?':>12}")
for m in sorted(d["results"]):
    s,a=d["results"][m]; es,ea=err(s),err(a); print(f"{m:>7} {es:>13.3e} {ea:>23.3e} {'YES' if ea<es else 'no':>12}")
