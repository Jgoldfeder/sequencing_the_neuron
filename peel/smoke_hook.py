import sys
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
from method import consensus_layer1_neurons
from verify_layer1 import verify_layer1, format_summary

torch.manual_seed(1)
dev = "cpu"
dims = [100, 16, 10, 6, 4]      # deep, 100-dim input
teacher = MLP(dims).to(dev)
with torch.no_grad():
    for l in teacher.layers[:-1]:
        l.bias.uniform_(-0.5, 0.5)

# 8 members agreeing on layer 1 (clone + tiny noise), diverging deeper
pop = []
for i in range(8):
    m = teacher.clone()
    with torch.no_grad():
        m.layers[0].weight.add_(torch.randn_like(m.layers[0].weight) * 2e-3)
        m.layers[0].bias.add_(torch.randn_like(m.layers[0].bias) * 2e-3)
        for l in m.layers[1:]:
            l.weight.add_(torch.randn_like(l.weight) * 0.3)
    pop.append(m)
X = torch.randn(3000, dims[0])

# ----- exactly the reconstruct hook path -----
l1 = consensus_layer1_neurons(pop, dims, quorum_ratio=0.625)
print(f"consensus_layer1_neurons -> {len(l1)}/{dims[1]} L1 neurons\n")

print("VERIFY only (cheap):")
res = verify_layer1(teacher, [(w, b) for _, w, b in l1], X, dev,
                    eps_offset=1e-2, eps_angle=1.0, k=16, refine=False, seed=3)
print("  [verify] L1", format_summary(res["summary"]))

print("\nVERIFY + REFINE:")
res2 = verify_layer1(teacher, [(w, b) for _, w, b in l1], X, dev,
                     eps_offset=1e-2, eps_angle=1.0, k=16, refine=True, seed=3)
print("  [verify] L1", format_summary(res2["summary"]))
print("\n  per-neuron (first 5), refined normal vs TRUE teacher plane:")
W1 = teacher.layers[0].weight.detach()
import math
for r in res2["per"][:5]:
    wr = r.get("w_refined")
    if wr is None:
        continue
    # find nearest true teacher neuron, report exact residual angle (validation only)
    best = min(range(dims[1]), key=lambda j:
               1 - abs(float(wr @ (W1[j]/W1[j].norm()).double())))
    tn = (W1[best]/W1[best].norm()).double()
    ang = math.degrees(math.acos(min(1.0, abs(float(wr @ tn)))))
    rf = r["refined"]
    print(f"    n{r['neuron']}: within={r['within']} verify_angle~{r['angle_deg']:.2f}deg "
          f"-> refined offset={rf['offset']:+.1e} TRUE-angle={ang:.1e}deg")
