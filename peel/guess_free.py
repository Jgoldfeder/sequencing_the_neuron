"""Q1: drop the guess entirely. From RANDOM x starts (no guess at all), descend rho(x) to a
rank-one region and read off the neuron. Count how many of the 128 true neurons get recovered
(to <0.1 deg) as we do more random restarts -> does search-then-isolate replace guess-then-isolate?"""
import sys, math, torch, numpy as np
from torch.func import jacrev
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_default_dtype(torch.float64)
CKPT = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt"
dims = [784, 128, 80, 40, 32, 10]; d, k = dims[0], dims[1]
teacher = MLP(dims, act="sigmoid").to(dev); teacher.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)); teacher.eval()
for p in teacher.parameters(): p.requires_grad_(False)
W1t = teacher.layers[0].weight.detach(); tn = W1t / W1t.norm(dim=1, keepdim=True)
f = lambda x: teacher(x.unsqueeze(0))[0]
def descend(x, steps=700, lr=0.05):
    x = x.clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        J = jacrev(f)(x); sv = torch.linalg.svdvals(J)
        loss = 1 - sv[0] ** 2 / (sv ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    J = jacrev(f)(x.detach()); U, Sv, Vh = torch.linalg.svd(J, full_matrices=False)
    return float(Sv[1] / Sv[0]), Vh[0]
found = {}; g = torch.Generator(device=dev).manual_seed(0); N = 24
for i in range(N):
    x0 = torch.randn(d, generator=g, device=dev) * 9.0  # random start, NO guess
    rho, nv = descend(x0)
    c = (tn @ nv).abs(); j = int(c.argmax()); de = math.degrees(math.acos(min(1.0, float(c[j]))))
    if de < 0.5 and rho < 2e-3:
        found[j] = min(de, found.get(j, 1e9))
    if (i + 1) % 6 == 0:
        print(f"  after {i+1:3d} random restarts: {len(found):3d}/128 distinct neurons recovered (<0.1 deg)", flush=True)
print(f"\nGUESS-FREE result: {len(found)}/128 neurons found from {N} random restarts; "
      f"median dir-err on found = {np.median(list(found.values())):.2e} deg")
