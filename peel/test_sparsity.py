"""Is the consensus->truth correction SPARSE? If yes, refinement can beat O(d)
via compressed sensing. If dense, O(d) is unavoidable. Measured on real neurons."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
from method import consensus_layer1_neurons
from verify_layer1 import _double_teacher, _Oracle, refine_neuron

DUMP = "/tmp/claude-1001/-home-judah/480010f4-8e92-4a50-8319-7f2f544ea513/scratchpad/grabbed_3072x256x100.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(DUMP, map_location=dev, weights_only=False)
dims = ck["dims"]; d = dims[0]
teacher = MLP(dims).to(dev); teacher.load_state_dict(ck["teacher_state"])
pop = [MLP(dims).to(dev) for _ in ck["pop_states"]]
for m, s in zip(pop, ck["pop_states"]): m.load_state_dict(s)
X = ck["X"].to(dev)

l1 = consensus_layer1_neurons(pop, dims, quorum_ratio=0.625)
td = _double_teacher(teacher); gen = torch.Generator(device=dev).manual_seed(0)

def compressibility(delta):
    """how many components hold 90% / 99% of the correction's energy."""
    e = (delta.double() ** 2)
    tot = e.sum()
    se, _ = torch.sort(e, descending=True)
    cum = torch.cumsum(se, 0) / tot
    n90 = int((cum < 0.90).sum()) + 1
    n99 = int((cum < 0.99).sum()) + 1
    return n90, n99

print(f"input dim d = {d}\n")
print(f"neuron |  ||delta|| | comps for 90pct | comps for 99pct | (d/2 ref = {d//2})")
n90s, n99s = [], []
for (idx, w, b) in l1[:12]:
    wc = (w / w.norm()).double()                      # consensus normal (unit)
    orc = _Oracle(td)
    rf = refine_neuron(orc, w, b, X[idx % len(X)], gen=gen)
    wr = rf["w_refined"].to(dev).double()
    if float(wr @ wc) < 0: wr = -wr
    delta = wr - wc                                    # correction vector (in input space)
    delta = delta - (delta @ wc) * wc                  # perpendicular part (the actual tilt fix)
    n90, n99 = compressibility(delta)
    n90s.append(n90); n99s.append(n99)
    print(f"  n{idx:3d}  |  {delta.norm():.2e} |     {n90:5d}     |     {n99:5d}     |")

print(f"\nmedian comps for 90pct energy: {sorted(n90s)[len(n90s)//2]}  (of d={d})")
print(f"median comps for 99pct energy: {sorted(n99s)[len(n99s)//2]}  (of d={d})")
print("\nIf these are SMALL (<< d) -> correction is sparse -> compressed sensing "
      "can refine in ~O(k log d), user is RIGHT.")
print("If these are ~d/2 -> correction is dense -> O(d) is unavoidable.")
