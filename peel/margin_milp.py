"""Exact MILP for the isolation margin M_j at layer 2 (true weights). Free signs via binaries.
max gamma  s.t.  |w_j.h + b_j| <= tau,  h in [0,1]^d,  and for each k!=j: |w_k.h+b_k| >= gamma
(binary y_k picks the sign). Decisive structural test: if the exact optimum is still small,
no reachable cube point can isolate a layer-2 neuron."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import milp, LinearConstraint, Bounds
dims = [784, 128, 80, 40, 32, 10]
teacher = MLP(dims, act="sigmoid").double(); teacher.load_state_dict(torch.load(
    "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt", map_location="cpu", weights_only=False))
W = teacher.layers[1].weight.detach().numpy(); b = teacher.layers[1].bias.detach().numpy()   # layer 2: 80x128
H, d = W.shape; tau = 0.1
def solve(j, tl=18.0):
    others = [k for k in range(H) if k != j]; nb = len(others)
    n = d + 1 + nb                                   # x = [h(d), gamma, y(nb)]
    c = np.zeros(n); c[d] = -1.0                      # maximize gamma
    Mk = np.array([max(abs(np.maximum(W[k], 0).sum() + b[k]), abs(np.minimum(W[k], 0).sum() + b[k])) for k in others])
    A = []; ub = []
    A.append(np.r_[W[j], 0, np.zeros(nb)]); ub.append(tau - b[j])      # w_j h <= tau - b_j
    A.append(np.r_[-W[j], 0, np.zeros(nb)]); ub.append(tau + b[j])     # -w_j h <= tau + b_j
    for m, k in enumerate(others):
        e = np.zeros(nb); e[m] = Mk[m]
        A.append(np.r_[-W[k], 1.0, e]); ub.append(b[k] + Mk[m])        # -w_k h + g + Mk y <= b_k + Mk
        e2 = np.zeros(nb); e2[m] = -Mk[m]
        A.append(np.r_[W[k], 1.0, e2]); ub.append(-b[k])              # w_k h + g - Mk y <= -b_k
    A = np.array(A); ub = np.array(ub)
    lb = np.r_[np.zeros(d), 0.0, np.zeros(nb)]; hb = np.r_[np.ones(d), 50.0, np.ones(nb)]
    intg = np.r_[np.zeros(d + 1), np.ones(nb)]
    r = milp(c, constraints=[LinearConstraint(A, -np.inf, ub)], integrality=intg,
             bounds=Bounds(lb, hb), options={"time_limit": tl, "mip_rel_gap": 0.02})
    if r.x is None: return None, None
    return float(r.x[d]), r.x[:d]
from torch.func import jacrev
Wl = [teacher.layers[i].weight.detach() for i in range(5)]; bl = [teacher.layers[i].bias.detach() for i in range(5)]
def Gfrom2(h):
    x = h
    for L in range(1, 5):
        z = x @ Wl[L].t() + bl[L]; x = torch.sigmoid(z) if L < 4 else z
    return x
w2n = Wl[1] / Wl[1].norm(dim=1, keepdim=True)                # true layer-2 unit directions
import math
print("EXACT MILP isolation at layer 2 (true weights): margin -> rho -> recovered direction")
for j in [0, 30, 55, 79]:
    g, h = solve(j)
    if h is None: print(f"  neuron {j}: infeasible"); continue
    ht = torch.tensor(h)
    J = jacrev(Gfrom2)(ht); U, Sv, Vh = torch.linalg.svd(J, full_matrices=False)
    rho = float(Sv[1] / Sv[0]); de = math.degrees(math.acos(min(1.0, float((w2n @ Vh[0]).abs().max()))))
    sp = 1 / (1 + np.exp(-g)) * (1 - 1 / (1 + np.exp(-g)))
    print(f"  neuron {j:2d}: M_j={g:.2f} (sigma'={sp:.1e}) | rho={rho:.2e} | dir to nearest true = {de:.2f} deg", flush=True)
