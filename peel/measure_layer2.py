"""Second-layer (80x128) weight error of each population member vs true teacher, under FULL
alignment: layer-1 perm+sign propagated into W2 columns, then layer-2 perm+sign on rows.
Layer 2 did NOT reach consensus -> report each member and pick the BEST."""
import torch, numpy as np
from scipy.optimize import linear_sum_assignment
CD = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
pop = torch.load(CD + "_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt", map_location="cpu", weights_only=False)
merged = torch.load(CD + "mergedbest512_sigmoid__784x128x80x40x32x10__s0_final.pt", map_location="cpu", weights_only=False)
W1t = pop["teacher_state"]["layers.0.weight"].double()   # (128,784)
W2t = pop["teacher_state"]["layers.1.weight"].double()   # (80,128)
def match_sign(A, B):                                     # match rows of A to rows of B, allow sign
    Cp = torch.cdist(A, B); Cm = torch.cdist(-A, B); C = torch.minimum(Cp, Cm)
    r, c = linear_sum_assignment(C.numpy())
    sgn = torch.where(Cm[r, c] < Cp[r, c], -1.0, 1.0)
    return r, c, sgn
def layer_errs(W1m, W2m):
    # 1) align layer-1 neurons (member -> true): perm c1, sign s1
    r1, c1, s1 = match_sign(W1m, W1t)                     # member l1-neuron i -> true c1[i], sign s1[i]
    l1_abs = (s1[:, None] * W1m[r1] - W1t[c1]).norm(dim=1); l1_rel = l1_abs / W1t[c1].norm(dim=1)
    # 2) reindex W2 columns into true-l1 order, apply column signs (complement of an l1 unit flips its incoming weight)
    W2re = torch.zeros_like(W2m); W2re[:, c1] = W2m[:, r1] * s1[None, :]
    # 3) align layer-2 neurons (rows)
    r2, c2, s2 = match_sign(W2re, W2t)
    l2_abs = (s2[:, None] * W2re[r2] - W2t[c2]).norm(dim=1); l2_rel = l2_abs / W2t[c2].norm(dim=1)
    return l1_rel, l2_rel
print(f"{'model':>10} | {'L1 rel mean':>11} {'L1 rel max':>10} | {'L2 rel mean':>11} {'L2 rel max':>10}")
results = []
for i, s in enumerate(pop["pop_states"]):
    l1, l2 = layer_errs(s["layers.0.weight"].double(), s["layers.1.weight"].double())
    results.append((float(l2.mean()), i, float(l1.mean()), float(l1.max()), float(l2.mean()), float(l2.max())))
    print(f"  pop[{i}]    | {float(l1.mean()):>11.3e} {float(l1.max()):>10.3e} | {float(l2.mean()):>11.3e} {float(l2.max()):>10.3e}")
l1m, l2m = layer_errs(merged["state_dict"]["layers.0.weight"].double(), merged["state_dict"]["layers.1.weight"].double())
print(f"  merged    | {float(l1m.mean()):>11.3e} {float(l1m.max()):>10.3e} | {float(l2m.mean()):>11.3e} {float(l2m.max()):>10.3e}")
best = min(results, key=lambda x: x[0])
print(f"\nBEST second layer: pop[{best[1]}]  L2 rel mean {best[4]:.3e}  max {best[5]:.3e}   (its L1 rel mean {best[2]:.3e})")
