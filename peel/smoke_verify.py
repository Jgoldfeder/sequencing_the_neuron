import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
from verify_layer1 import verify_layer1, probe_neuron, _double_teacher

torch.manual_seed(0)
dev = "cpu"
# deep-ish teacher like the run: input 20, several hidden layers, LeakyReLU
dims = [20, 12, 8, 6, 4]
teacher = MLP(dims).to(dev)
# spread biases so planes sit in a queryable region
with torch.no_grad():
    for l in teacher.layers[:-1]:
        l.bias.uniform_(-0.5, 0.5)

W1, b1 = teacher.layers[0].weight.detach(), teacher.layers[0].bias.detach()
X = torch.randn(2000, dims[0])  # stand-in query pool

# --- test 1: exact hypothesis == true neuron k -> residuals ~ 0 ---
print("=== exact hypotheses (should be ~0 offset, ~0 angle) ===")
neurons = [(W1[k].clone(), b1[k].clone()) for k in range(dims[1])]
res = verify_layer1(teacher, neurons, X, dev, n_base=4, want_normal=True)
for r in res["per"]:
    print(f"  n{r['neuron']}: hit={r['hit']} matched={r['matched']} "
          f"offset={r['offset_resid']:+.2e} angle={r['angle_deg']:.2e} deg "
          f"jump_ratio={r.get('jump_ratio', float('nan')):.1f}")
print("  summary:", res["summary"])

# --- test 2: perturbed hypothesis -> verifier should recover the perturbation ---
print("\n=== perturbed hypotheses (offset should track the injected shift) ===")
k = 0
# shift the plane along its normal by delta: b -> b - delta*||w||  moves plane by delta
w = W1[k].clone(); b = b1[k].clone()
nrm = w.norm()
for delta in (0.0, 0.05, -0.1):
    b_shift = b - delta * nrm          # plane {w.x + b_shift = 0} is delta from true along w_hat
    td = _double_teacher(teacher)
    out = None
    for j in range(6):
        base = X[j].double()
        o = probe_neuron(td, w.double(), b_shift.double(), base, want_normal=True)
        if o and (out is None or abs(o["offset_resid"]) < abs(out["offset_resid"])):
            out = o
    # verifier walks from the HYPOTHESIS plane; true kink is at -delta along w_hat
    print(f"  delta={delta:+.3f}: offset_resid={out['offset_resid']:+.4f} "
          f"(expect {-delta:+.3f}), angle={out['angle_deg']:.2e} deg")

# --- test 3: rotate the normal -> angle residual should track the rotation ---
print("\n=== rotated normal (angle should track injected rotation) ===")
w = W1[0].clone(); b = b1[0].clone()
u = torch.randn_like(w); u = u - (u @ w) / (w @ w) * w; u = u / u.norm()  # perp dir
for ang in (0.0, 1.0, 5.0):
    th = math.radians(ang)
    w_rot = math.cos(th) * (w / w.norm()) + math.sin(th) * u  # unit, rotated by ang
    td = _double_teacher(teacher)
    out = None
    for j in range(6):
        o = probe_neuron(td, w_rot.double(), (b / w.norm()).double(), X[j].double())
        if o and (out is None or abs(o["offset_resid"]) < abs(out["offset_resid"])):
            out = o
    print(f"  injected {ang:.1f} deg -> measured angle={out['angle_deg']:.3f} deg")
