"""Decisive test: does REFINE work on the real 3072x256x100 run's consensus?

Loads the iter-35 dump (population + true teacher + queries), extracts the real
consensus layer-1 neurons, and for each compares:
  raw consensus plane vs TRUE teacher plane      (how good the guess is)
  refined plane        vs TRUE teacher plane      (how good refine made it)
Refine is black-box (queries only); the teacher weights are used ONLY to score.
If refine works, refined-vs-teacher << raw-vs-teacher.
"""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
from method import consensus_layer1_neurons
from verify_layer1 import _double_teacher, _Oracle, verify_neuron, refine_neuron

DUMP = "/tmp/claude-1001/-home-judah/480010f4-8e92-4a50-8319-7f2f544ea513/scratchpad/grabbed_3072x256x100.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(DUMP, map_location=dev, weights_only=False)
dims = ck["dims"]
teacher = MLP(dims).to(dev); teacher.load_state_dict(ck["teacher_state"])
pop = []
for s in ck["pop_states"]:
    m = MLP(dims).to(dev); m.load_state_dict(s); pop.append(m)
X = ck["X"].to(dev)
print(f"loaded dump: dims={dims}  pop={len(pop)}  queries={len(X)}")

# true teacher layer-1 planes (normalized): n_j.x + o_j = 0
Wt = teacher.layers[0].weight.detach().double()
bt = teacher.layers[0].bias.detach().double()
tn = Wt / Wt.norm(dim=1, keepdim=True)                     # (H, d) unit normals
to = bt / Wt.norm(dim=1)                                   # (H,) offsets

l1 = consensus_layer1_neurons(pop, dims, quorum_ratio=0.625)
print(f"consensus layer-1 neurons: {len(l1)}\n")

td = _double_teacher(teacher)
gen = torch.Generator(device=dev).manual_seed(0)

def score(nrm, off):
    """nearest teacher plane; return (angle_deg, offset_err) aligned by sign."""
    nrm = nrm.to(dev).double()
    cos = tn @ nrm                                         # (H,)
    j = int(cos.abs().argmax())
    s = 1.0 if cos[j] >= 0 else -1.0
    ang = math.degrees(math.acos(min(1.0, float(cos[j].abs()))))
    off_err = abs(float(off) - s * float(to[j]))
    return ang, off_err, j

N = min(40, len(l1))                                       # sample to keep it quick
raw_a, raw_o, ref_a, ref_o, q = [], [], [], [], 0
for (idx, w, b) in l1[:N]:
    wn = (w / w.norm()).double()
    ra, ro, j = score(wn, (b / w.norm()))
    orc = _Oracle(td)
    rf = refine_neuron(orc, w, b, X[idx % len(X)], gen=gen)
    q += orc.n
    if rf is None:
        continue
    fa, fo, _ = score(rf["w_refined"], rf["b_refined"])
    raw_a.append(ra); raw_o.append(ro); ref_a.append(fa); ref_o.append(fo)

def med(v): v = sorted(v); return v[len(v)//2]
print(f"probed {len(raw_a)} neurons, {q} queries ({q//max(len(raw_a),1)}/neuron)\n")
print("                     RAW consensus     ->   REFINED")
print(f"  angle vs teacher:  med {med(raw_a):.4f}°  max {max(raw_a):.4f}°"
      f"   ->  med {med(ref_a):.2e}°  max {max(ref_a):.2e}°")
print(f"  offset vs teacher: med {med(raw_o):.2e}  max {max(raw_o):.2e}"
      f"   ->  med {med(ref_o):.2e}  max {max(ref_o):.2e}")
print("\n  per-neuron sample (first 8):")
for i in range(min(8, len(raw_a))):
    print(f"    raw angle {raw_a[i]:6.3f}° off {raw_o[i]:.2e}  ->  "
          f"refined angle {ref_a[i]:.2e}° off {ref_o[i]:.2e}")
imp = med(raw_a) / max(med(ref_a), 1e-30)
print(f"\n  => refine shrinks median angle error by ~{imp:.0f}x")
