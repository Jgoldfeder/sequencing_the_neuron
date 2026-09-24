import sys
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
from method import consensus_layer1_neurons
from verify_layer1 import verify_layer1, format_summary

torch.manual_seed(1)
dev = "cpu"
# deep teacher, matching the shape of the real run (small input for speed)
dims = [30, 16, 10, 6, 4]
teacher = MLP(dims).to(dev)
with torch.no_grad():
    for l in teacher.layers[:-1]:
        l.bias.uniform_(-0.5, 0.5)

# population of 8 members that AGREE on layer 1 (clones + tiny noise on layer 1,
# larger noise deeper) -> layer-1 consensus should form, deeper layers should not
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

# --- exactly what the reconstruct hook does ---
l1 = consensus_layer1_neurons(pop, dims, quorum_ratio=0.625)
print(f"consensus_layer1_neurons -> {len(l1)} / {dims[1]} layer-1 neurons")
assert len(l1) > 0, "expected some layer-1 consensus"

res = verify_layer1(teacher, [(w, b) for _, w, b in l1], X, dev,
                    want_normal=True, max_neurons=0, seed=7)
print("[verify] L1", format_summary(res["summary"]))
print("\nper-neuron:")
for r in res["per"]:
    print(f"  n{r['neuron']}: matched={r['matched']} hit={r['hit']} "
          f"offset={r['offset_resid']:+.2e} angle={r['angle_deg']:.3f} deg")

# sanity: since members are teacher-clones+tiny-noise, most should match the
# teacher's true planes with small residuals
nm = res["summary"]["n_matched"]
print(f"\nmatched {nm}/{len(l1)}  (expect most, small offsets/angles)")

# --- also exercise offset-only mode (want_normal=False) ---
res2 = verify_layer1(teacher, [(w, b) for _, w, b in l1], X, dev,
                     want_normal=False, seed=7)
print("[verify offset-only]", format_summary(res2["summary"]))
