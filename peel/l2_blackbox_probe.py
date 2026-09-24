"""STEP 1 for honest L2: (a) build the committee W2 guess WITHOUT touching truth
(align members to member-0, not to true W2), (b) probe whether the layer-2 neurons
can be ISOLATED given the bounded subnetwork input h in (0,1)^128.

Layer 2 = teacher.layers[1] (80x128). Its input h is the layer-1 output, which -- because
we've solved L1 (full row rank) -- we can set to ANY h in (0,1)^128 via x=W1^+(logit(h)-b1).
So layers 2..4 form a black-box subnetwork N2(h) we can query freely. This script only
diagnoses feasibility; it does not yet solve.
"""
import sys, torch, numpy as np
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from scipy.optimize import linear_sum_assignment
from nets import MLP
torch.set_default_dtype(torch.float64)
dev = "cuda" if torch.cuda.is_available() else "cpu"
P = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"

pk = torch.load(P + "peel_committee.pt", map_location=dev, weights_only=False)
dims = [784,128,80,40,32,10]
teacher = MLP(dims, act="sigmoid").to(dev); teacher.load_state_dict(pk["teacher_state"])
W2t = teacher.layers[1].weight.detach(); b2t = teacher.layers[1].bias.detach(); nt = W2t.norm(dim=1)

# ---------- honest committee guess: align members to MEMBER 0 (no truth) ----------
def match(A, B):
    Cp = torch.cdist(A, B); Cm = torch.cdist(-A, B); C = torch.minimum(Cp, Cm)
    r, c = linear_sum_assignment(C.cpu().numpy()); r = torch.tensor(r); c = torch.tensor(c)
    s = torch.where(Cm[r, c] < Cp[r, c], -1.0, 1.0)
    return r, c, s
mem = [(sd["layers.1.weight"].to(dev).double(), sd["layers.1.bias"].to(dev).double()) for sd in pk["pop_states"]]
W0, b0 = mem[0]                                  # reference frame = member 0 (arbitrary but truth-free)
AW = [W0.clone()]; AB = [b0.clone()]
for Wm, bm in mem[1:]:
    r, c, s = match(Wm, W0)                       # align this member's rows to member 0
    Wa = torch.zeros_like(Wm); ba = torch.zeros_like(bm)
    Wa[c] = Wm[r] * s[:, None]; ba[c] = bm[r] * s
    AW.append(Wa); AB.append(ba)
AW = torch.stack(AW); AB = torch.stack(AB)
W2g = AW.median(0).values.clone(); b2g = AB.median(0).values.clone()   # truth-free guess (member-0 order)

# score the guess vs truth (SCORING ONLY: truth used here and nowhere in the guess construction)
def score(W, tag):
    Cp = torch.cdist(W, W2t); Cm = torch.cdist(-W, W2t); C = torch.minimum(Cp, Cm).cpu().numpy()
    ri, ci = linear_sum_assignment(C)
    e = torch.tensor([C[ri[t], ci[t]] for t in range(len(ri))]); rel = e / nt[ci].cpu()
    print(f"  {tag:28s} rel-row-err: mean {float(rel.mean()):.3e}  max {float(rel.max()):.3e}")
    return ri, ci
print(f"committee: {len(mem)} members, member-0 frame (truth-free)")
score(W2g, "GUESS (committee median)")

# ---------- isolation feasibility: can we drive z2 to isolate one neuron in (0,1)^128? ----------
# For neuron j: target preactivation t with t[j]=0, t[i!=j]=+/-G (push others to saturate).
# Solve min-norm h = W2g^+(t - b2g), then CLAMP into the cube. Measure how saturated others get.
W2gp = W2g.t() @ torch.linalg.inv(W2g @ W2g.t())
def sigp(z): s = torch.sigmoid(z); return s * (1 - s)          # sigma'
print("\nisolation feasibility (drive neuron j to transition, others to saturation), true z2 after clamp:")
print(f"{'G target':>9} {'|z_j| (want~0)':>16} {'median|z_i!=j|':>16} {'frac others |z|>4':>18} {'sigmaprime ratio j:others':>26}")
gsrc = torch.Generator(device="cpu").manual_seed(0)
for G in [4.0, 8.0, 15.0]:
    zj, zi_med, frac_sat, ratio = [], [], [], []
    for j in range(80):
        sign = (2*(torch.rand(80, generator=gsrc) > 0.5).double() - 1).to(dev)
        t = sign * G; t[j] = 0.0
        h = torch.clamp(W2gp @ (t - b2g), 1e-4, 1-1e-4)         # projected into cube
        z = W2t @ h + b2t                                       # TRUE preactivation at that h (scoring/diagnostic)
        zj.append(float(z[j].abs()))
        others = torch.cat([z[:j], z[j+1:]])
        zi_med.append(float(others.abs().median()))
        frac_sat.append(float((others.abs() > 4).double().mean()))
        ratio.append(float(sigp(z[j]) / (sigp(others).mean() + 1e-30)))
    print(f"{G:>9.1f} {np.mean(zj):>16.3f} {np.mean(zi_med):>16.3f} {np.mean(frac_sat):>18.3f} {np.mean(ratio):>26.2f}")
print("\n(sigmaprime ratio >> 1 means neuron j dominates the Jacobian -> isolation works; ~1 means swamped.)")
