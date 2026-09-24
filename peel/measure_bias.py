"""First-layer BIAS (b1) error of the consensus / population vs true teacher, under the same
alignment as the weights (perm + sign/complement, with the sign flip applied to the bias:
complement neuron has (w,b)->(-w,-b), so aligned bias = s*b_hat). abs |db| is the preactivation
error that drives the peel blur. Compare against the weight error."""
import torch, numpy as np
from scipy.optimize import linear_sum_assignment
CD = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
pop = torch.load(CD + "_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt", map_location="cpu", weights_only=False)
merged = torch.load(CD + "mergedbest512_sigmoid__784x128x80x40x32x10__s0_final.pt", map_location="cpu", weights_only=False)
W1t = pop["teacher_state"]["layers.0.weight"].double(); b1t = pop["teacher_state"]["layers.0.bias"].double()
nt = W1t.norm(dim=1)
print(f"true b1: mean|b*| {float(b1t.abs().mean()):.3f} max|b*| {float(b1t.abs().max()):.3f} | mean||w|| {float(nt.mean()):.3f}")
def errs(W, b, tag):
    W = W.double(); b = b.double()
    Cp = torch.cdist(W, W1t); Cm = torch.cdist(-W, W1t); minus = Cm < Cp; C = torch.minimum(Cp, Cm)
    ri, ci = linear_sum_assignment(C.numpy()); ri = torch.tensor(ri); ci = torch.tensor(ci)
    s = torch.where(minus[ri, ci], -1.0, 1.0)                    # sign/complement per matched pair
    werr = (s[:, None] * W[ri] - W1t[ci]).norm(dim=1)            # weight err (aligned)
    berr = (s * b[ri] - b1t[ci]).abs()                           # bias err (aligned, sign-flipped)
    woff = werr / nt[ci]                                         # rel weight
    boff = berr / nt[ci]                                         # bias err as input-space offset shift (|db|/||w||)
    print(f"  {tag}")
    print(f"    WEIGHT rel ||dw||/||w||: mean {float(woff.mean()):.3e} max {float(woff.max()):.3e}")
    print(f"    BIAS   abs |db|        : mean {float(berr.mean()):.3e} max {float(berr.max()):.3e}")
    print(f"    BIAS   |db|/||w|| (offset): mean {float(boff.mean()):.3e} max {float(boff.max()):.3e}")
    return float(berr.mean())
errs(merged["state_dict"]["layers.0.weight"], merged["state_dict"]["layers.0.bias"], "CONSENSUS (merged)")
print("  --- per population member (bias |db| mean / max) ---")
for i, s in enumerate(pop["pop_states"]):
    W = s["layers.0.weight"].double(); b = s["layers.0.bias"].double()
    Cp = torch.cdist(W, W1t); Cm = torch.cdist(-W, W1t); minus = Cm < Cp; C = torch.minimum(Cp, Cm)
    ri, ci = linear_sum_assignment(C.numpy()); ri = torch.tensor(ri); ci = torch.tensor(ci)
    sg = torch.where(minus[ri, ci], -1.0, 1.0)
    be = (sg * b[ri] - b1t[ci]).abs()
    print(f"    pop[{i}]: bias |db| mean {float(be.mean()):.3e} max {float(be.max()):.3e}")
