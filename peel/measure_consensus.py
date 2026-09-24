"""Measure the REAL first-layer weight error of the extracted models vs the true teacher,
under alignment (permutation + sign/complement). No fabricated guess."""
import sys, torch, numpy as np
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from scipy.optimize import linear_sum_assignment
CD = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
pop = torch.load(CD + "_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt", map_location="cpu", weights_only=False)
merged = torch.load(CD + "mergedbest512_sigmoid__784x128x80x40x32x10__s0_final.pt", map_location="cpu", weights_only=False)
W1t = pop["teacher_state"]["layers.0.weight"].double()          # (128,784) true
b1t = pop["teacher_state"]["layers.0.bias"].double()
def row_err_aligned(W, b, tag):
    W = W.double()
    nt = W1t.norm(dim=1)
    # cost: min over sign of ||+-W[i]-W1t[j]|| (sign/complement); pick best perm
    Cp = torch.cdist(W, W1t); Cm = torch.cdist(-W, W1t)
    sign_is_minus = (Cm < Cp)
    C = torch.minimum(Cp, Cm).numpy()
    ri, ci = linear_sum_assignment(C)
    errs = torch.tensor([C[ri[t], ci[t]] for t in range(len(ri))])
    rel = errs / nt[ci]
    print(f"  {tag}: first-layer weight-row err (aligned)  abs mean {float(errs.mean()):.3e} max {float(errs.max()):.3e} | rel mean {float(rel.mean()):.3e} max {float(rel.max()):.3e}")
    return errs, rel
print(f"true ||w|| mean {float(W1t.norm(dim=1).mean()):.3f}")
print("=== merged 'mergedbest512_sigmoid' final reconstruction ===")
sd = merged["state_dict"]
row_err_aligned(sd["layers.0.weight"], sd["layers.0.bias"], "merged")
print(f"  (stored: final_mean_eps={merged.get('final_mean_eps')}, final_max_eps={merged.get('final_max_eps')}, final_mae={merged.get('final_mae')})")
print("=== population members (each state_dict) ===")
best = None
for i, s in enumerate(pop["pop_states"]):
    e, r = row_err_aligned(s["layers.0.weight"], s["layers.0.bias"], f"pop[{i}]")
    if best is None or r.mean() < best[0]: best = (float(r.mean()), i)
print(f"best pop member by mean rel: pop[{best[1]}] ({best[0]:.3e})")
