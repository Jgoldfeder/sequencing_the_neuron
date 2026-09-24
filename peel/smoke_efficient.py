import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
from verify_layer1 import _double_teacher, _Oracle, verify_neuron, refine_neuron

torch.manual_seed(0)
dev = "cpu"
d = 200                      # input dim, to show verify cost is d-INDEPENDENT
dims = [d, 24, 16, 10, 6]
teacher = MLP(dims).to(dev)
with torch.no_grad():
    for l in teacher.layers[:-1]:
        l.bias.uniform_(-0.5, 0.5)
W1, b1 = teacher.layers[0].weight.detach(), teacher.layers[0].bias.detach()
X = torch.randn(2000, d)
td = _double_teacher(teacher)
gen = torch.Generator(device=dev).manual_seed(1)

def perp(w):
    u = torch.randn(d); u = u - (u @ w) / (w @ w) * w; return u / u.norm()

print("=== 1. VERIFY: cost is O(k), independent of d =========================")
k = 0
w = W1[k].clone(); b = b1[k].clone(); nrm = w.norm()
for label, ww, bb in [
    ("exact",                 w.clone(),                              b.clone()),
    ("offset off 0.05 (>eps)",w.clone(),                              b - 0.05 * nrm),
    ("angle off ~3deg",       math.cos(math.radians(3))*(w/nrm) + math.sin(math.radians(3))*perp(w), b / nrm),
    ("angle off ~0.2deg",     math.cos(math.radians(.2))*(w/nrm)+ math.sin(math.radians(.2))*perp(w), b / nrm),
]:
    orc = _Oracle(td)
    r = verify_neuron(orc, ww.double(), bb.double(), X[0].double(), gen=gen,
                      eps_offset=1e-2, eps_angle=1.0, k=16)
    off = f"{r['offset']:+.4f}" if r['offset'] is not None else "  none "
    ang = f"{r['angle_deg']:.3f}" if r['angle_deg'] is not None else " n/a "
    print(f"  {label:24s}: within={str(r['within']):5s} offset={off} "
          f"angle~{ang}deg  queries={r['queries']}  (d={d})")

print("\n=== 2. REFINE: offset->machine precision, normal at ~2d floor =========")
for label, ww, bb in [
    ("angle off ~2deg", math.cos(math.radians(2))*(w/nrm)+math.sin(math.radians(2))*perp(w), b/nrm),
    ("offset off 0.02", w/nrm, (b - 0.02*nrm)/nrm),
]:
    orc = _Oracle(td)
    rf = refine_neuron(orc, ww.double(), bb.double(), X[0].double())
    # ground-truth check (allowed here: we're validating the probe, not using it)
    true_n = (W1[k] / W1[k].norm()).double()
    ang_true = math.degrees(math.acos(float((rf['w_refined'].to(true_n) @ true_n).abs().clamp(max=1))))
    print(f"  {label:18s}: refined offset={rf['offset']:+.2e}  "
          f"angle(vs guess)={rf['angle_deg']:.3f}deg  "
          f"TRUE angle(refined vs teacher)={ang_true:.2e}deg  "
          f"queries={rf['queries']}  (2d={2*d})")

print("\n  -> verify ~O(k) queries regardless of d; refine offset cheap, "
      "normal ~2d as expected.")
