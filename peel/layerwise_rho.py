"""Layerwise isolation-quality rho_{l,j} for the fully-contractive net 784->128->80->40->32->10.
For each layer l, construct neuron j's isolation probe (drive z_j~0, others to +-S) in that layer's
INPUT space, then measure rho = sigma2/sigma1 of dG_l/d(input). Layer 1 input = x (UNBOUNDED);
layers >=2 input = h in (0,1)^d (BOUNDED sigmoid activations -> probe must be clamped to be reachable).
Uses TRUE weights for the probe (best-case: isolates the structural bounded-input limit, no guess error)."""
import sys, torch, numpy as np
from torch.func import jacrev
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_default_dtype(torch.float64)
CKPT = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt"
dims = [784, 128, 80, 40, 32, 10]
teacher = MLP(dims, act="sigmoid").to(dev); teacher.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)); teacher.eval()
Wl = [teacher.layers[i].weight.detach() for i in range(5)]; bl = [teacher.layers[i].bias.detach() for i in range(5)]
def Gfrom(h, ell):                       # forward from layer ell (1-indexed); input h feeds weight ell-1
    x = h
    for L in range(ell - 1, 5):
        z = x @ Wl[L].t() + bl[L]; x = torch.sigmoid(z) if L < 4 else z
    return x
S = 20.0
print("layer  d_in   #neurons | rho over sampled neurons (median / MAX) | others' saturation: max sigma'(z_k)")
for ell in range(1, 6):
    W = Wl[ell - 1]; b = bl[ell - 1]; H, din = W.shape
    Wpinv = W.t() @ torch.linalg.inv(W @ W.t())
    rhos = []; sats = []
    js = torch.linspace(0, H - 1, min(H, 16)).long().tolist()
    for j in js:
        t = torch.full((H,), S, device=dev); t[j] = 0.0
        hraw = Wpinv @ (t - b)
        h = hraw if ell == 1 else hraw.clamp(1e-3, 1 - 1e-3)         # layer>=2 input bounded to (0,1)
        z = W @ h + b; sp = (torch.sigmoid(z) * (1 - torch.sigmoid(z)))
        mask = torch.ones(H, dtype=torch.bool, device=dev); mask[j] = False
        sats.append(float(sp[mask].max()))                          # want ~0 (others saturated)
        J = jacrev(lambda hh: Gfrom(hh, ell))(h)
        sv = torch.linalg.svdvals(J); rhos.append(float(sv[1] / sv[0]))
    rhos = np.array(rhos)
    print(f"  {ell}   {din:4d}   {H:4d}     | rho med {np.median(rhos):.2e}  MAX {rhos.max():.2e}    | max sigma'(z_k!=j) med {np.median(sats):.2e}", flush=True)
