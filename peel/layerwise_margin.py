"""Optimized in-cube isolation margin M_j (the real structural test, not pinv+clamp).
For each neuron j of layer ell (true weights), find h in (0,1)^d that MAXIMIZES the worst-case
saturation of the OTHER neurons, min_{k!=j}|w_k h + b_k|, while keeping |w_j h + b_j| <= tau.
Free signs (some others +, some -). Then read rho and recovered direction at that optimum.
If even this oracle-optimal cube point can't saturate the others (M_j small), the wall is real."""
import sys, math, torch, numpy as np
from torch.func import jacrev
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_default_dtype(torch.float64)
CKPT = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt"
dims = [784, 128, 80, 40, 32, 10]
teacher = MLP(dims, act="sigmoid").to(dev); teacher.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)); teacher.eval()
Wl = [teacher.layers[i].weight.detach() for i in range(5)]; bl = [teacher.layers[i].bias.detach() for i in range(5)]
def Gfrom(h, ell):
    x = h
    for L in range(ell - 1, 5):
        z = x @ Wl[L].t() + bl[L]; x = torch.sigmoid(z) if L < 4 else z
    return x
tau = 0.1; T = 0.5
def optimize_margin(W, b, j, iters=500):
    H, d = W.shape; best = None
    for seed in range(3):
        u = (torch.randn(d, generator=torch.Generator(device=dev).manual_seed(seed), device=dev) * 1.5).requires_grad_(True)
        opt = torch.optim.Adam([u], lr=0.08)
        for _ in range(iters):
            h = torch.sigmoid(u); z = W @ h + b
            oth = torch.cat([z[:j], z[j + 1:]]).abs()
            softmin = -T * torch.logsumexp(-oth / T, 0)                     # ~ min_{k!=j}|z_k|
            loss = -softmin + 20.0 * torch.relu(z[j].abs() - tau)          # maximize margin, keep j active
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            h = torch.sigmoid(u); z = W @ h + b
            if z[j].abs() <= tau + 1e-3:
                m = float(torch.cat([z[:j], z[j + 1:]]).abs().min())
                if best is None or m > best[0]: best = (m, h.detach())
    return best
print(f"optimized in-cube isolation (true weights, tau={tau}):  M_j = max achievable min_(k!=j)|z_k|")
for ell in [2, 3, 4]:                                # skip layer1 (unbounded, works) and layer5 (linear)
    W, b = Wl[ell - 1], bl[ell - 1]; H, d = W.shape
    Ms, rhos, des = [], [], []
    wn = W / W.norm(dim=1, keepdim=True)
    for j in torch.linspace(0, H - 1, min(H, 10)).long().tolist():
        r = optimize_margin(W, b, j)
        if r is None: continue
        M, h = r; Ms.append(M)
        J = jacrev(lambda hh: Gfrom(hh, ell))(h); U, Sv, Vh = torch.linalg.svd(J, full_matrices=False)
        rhos.append(float(Sv[1] / Sv[0]))
        c = (wn @ Vh[0]).abs().max(); des.append(math.degrees(math.acos(min(1.0, float(c)))))
    Ms = np.array(Ms)
    print(f"  layer {ell} (d_in={d}, {H} neurons): M_j med {np.median(Ms):.2f} min {Ms.min():.2f} max {Ms.max():.2f}  "
          f"-> sigma'(M) med {float(torch.sigmoid(torch.tensor(np.median(Ms)))*(1-torch.sigmoid(torch.tensor(np.median(Ms))))):.1e}  "
          f"| rho med {np.median(rhos):.2e}  | dir-err med {np.median(des):.2e} deg", flush=True)
