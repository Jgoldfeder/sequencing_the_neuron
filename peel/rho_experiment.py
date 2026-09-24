"""Q2: is 'good guess' better defined by rho = sigma2/sigma1 of the isolation Jacobian than by
parameter error? Real 784x128 sigmoid first layer. For many corrupted guesses, construct each
neuron's isolation probe, measure rho (observable), the neuron's param error, and the recovered
direction error. Then P(recover | rho) vs P(recover | param)."""
import sys, math, torch, numpy as np
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_default_dtype(torch.float64)
CKPT = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt"
dims = [784, 128, 80, 40, 32, 10]; d, k = dims[0], dims[1]
teacher = MLP(dims, act="sigmoid").to(dev); teacher.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)); teacher.eval()
for p in teacher.parameters(): p.requires_grad_(False)
W1t = teacher.layers[0].weight.detach(); b1t = teacher.layers[0].bias.detach(); tn = W1t / W1t.norm(dim=1, keepdim=True)
@torch.no_grad()
def J_at(x, fd=5e-5):
    E = torch.eye(d, device=dev); return ((teacher(x.unsqueeze(0) + fd * E) - teacher(x.unsqueeze(0) - fd * E)) / (2 * fd)).t()
S = 20.0; g = torch.Generator(device=dev).manual_seed(0)
rows = []   # (rho, dir_err_deg, param_rel_err, theta_dir, eps_mag, eps_b)
combos = [(th, em, eb) for th in (1, 3, 5, 10, 20, 30, 45) for em in (0.05, 0.3) for eb in (0.05, 0.3, 1.0)]
for (theta, emag, eb) in combos:
    Wg = torch.empty(k, d, device=dev); bg = torch.empty(k, device=dev)
    for j in range(k):
        u = tn[j]; v = torch.randn(d, generator=g, device=dev); v = v - (v @ u) * u; v = v / v.norm()
        Wg[j] = (math.cos(math.radians(theta)) * u + math.sin(math.radians(theta)) * v) * W1t[j].norm() * (1 + emag * (2 * torch.rand(1, generator=g, device=dev) - 1))
        bg[j] = b1t[j] + eb * (2 * torch.rand(1, generator=g, device=dev).item() - 1)
    Wgpinv = Wg.t() @ torch.linalg.inv(Wg @ Wg.t())
    for j in torch.randint(0, k, (12,), generator=torch.Generator().manual_seed(theta * 100 + int(eb * 10))).tolist():
        t = torch.full((k,), S, device=dev); t[j] = 0.0
        x0 = Wgpinv @ (t - bg)
        U, Sv, Vh = torch.linalg.svd(J_at(x0), full_matrices=False)
        rho = float(Sv[1] / Sv[0]); nv = Vh[0]; nv = nv if float(nv @ Wg[j]) > 0 else -nv
        de = math.degrees(math.acos(min(1.0, abs(float(nv @ tn[j])))))
        pe = float((Wg[j] - W1t[j]).norm() / W1t[j].norm())
        rows.append((rho, de, pe))
R = np.array(rows)
succ = R[:, 1] < 0.1     # recovered direction to <0.1 deg
print(f"{len(R)} (guess,neuron) samples on real 784x128 first layer.  success = dir err < 0.1 deg\n")
print("P(recover | rho = sigma2/sigma1 of isolation Jacobian):")
for lo, hi in [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 5e-2), (5e-2, 2e-1), (2e-1, 1e9)]:
    m = (R[:, 0] >= lo) & (R[:, 0] < hi)
    if m.sum(): print(f"  rho in [{lo:.0e},{hi:.0e}): n={int(m.sum()):3d}  P(recover)={succ[m].mean():.2f}  med dir-err={np.median(R[m,1]):.1e} deg")
print("\nP(recover | parameter error) -- same samples, binned by relative weight error:")
for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1e9)]:
    m = (R[:, 2] >= lo) & (R[:, 2] < hi)
    if m.sum(): print(f"  param in [{lo:.2f},{hi:.2f}): n={int(m.sum()):3d}  P(recover)={succ[m].mean():.2f}  rho range=[{R[m,0].min():.1e},{R[m,0].max():.1e}]")
