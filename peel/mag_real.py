"""Magnitude (and direction) recovery on the REAL trained sigmoid teacher
teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt  (first layer W1 = 128x784).
Guess = 5deg direction, +-8% magnitude, +-0.08 bias.  Report before -> after."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
dev = "cuda"
CKPT = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt"
dims = [784, 128, 80, 40, 32, 10]; d = dims[0]; k = dims[1]
teacher = MLP(dims, act="sigmoid").to(dev).double()
teacher.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)); teacher.eval()
W1t = teacher.layers[0].weight.detach(); b1t = teacher.layers[0].bias.detach()
tn = W1t / W1t.norm(dim=1, keepdim=True); g = torch.Generator(device=dev).manual_seed(1)

@torch.no_grad()
def J_at(x, fd=1e-4):
    E = torch.eye(d, device=dev, dtype=torch.float64)
    return ((teacher(x.unsqueeze(0) + fd * E) - teacher(x.unsqueeze(0) - fd * E)) / (2 * fd)).t()

# ---- guess: 5deg dir, +-8% mag, +-0.08 bias ----
Wg = torch.empty(k, d, device=dev, dtype=torch.float64); bg = torch.empty(k, device=dev, dtype=torch.float64)
for j in range(k):
    u = tn[j]; v = torch.randn(d, generator=g, device=dev, dtype=torch.float64); v = v - (v @ u) * u; v = v / v.norm()
    Wg[j] = (math.cos(math.radians(5)) * u + math.sin(math.radians(5)) * v) * W1t[j].norm() * (1 + 0.08 * (2 * torch.rand(1, generator=g, device=dev, dtype=torch.float64) - 1))
    bg[j] = b1t[j] + 0.08 * (2 * torch.rand(1, generator=g, device=dev, dtype=torch.float64).item() - 1)
Wg_pinv = Wg.t() @ torch.linalg.inv(Wg @ Wg.t())

mag_true = W1t.norm(dim=1)                                   # (128,)
before_abs = (Wg.norm(dim=1) - mag_true).abs()
before_rel = before_abs / mag_true
before_dir = torch.tensor([math.degrees(math.acos(min(1.0, abs(float(Wg[j] @ tn[j] / Wg[j].norm()))))) for j in range(k)])

@torch.no_grad()
def tail_mag(x0, n, wg, dt, sign):
    base = 3.5 / wg
    ts = sign * (base + torch.arange(0, 9, device=dev, dtype=torch.float64) * dt)
    F = teacher(x0.unsqueeze(0) + ts.unsqueeze(1) * n.unsqueeze(0))
    D = (F[:-1] - F[1:]).norm(dim=1)
    r = (D[:-1] / D[1:].clamp_min(1e-300)).log() / abs(dt)
    return float(r.median())

Sval = 20.0
after_abs = torch.zeros(k); after_dir = torch.zeros(k)
for j in range(k):
    t = torch.full((k,), Sval, device=dev, dtype=torch.float64); t[j] = 0.0
    x0 = Wg_pinv @ (t - bg)
    U, Sv, Vh = torch.linalg.svd(J_at(x0), full_matrices=False)
    n = Vh[0]; n = n if float(n @ Wg[j]) > 0 else -n
    dt = 0.4 / float(Wg[j].norm())
    a = 0.5 * (tail_mag(x0, n, float(Wg[j].norm()), dt, +1) + tail_mag(x0, n, float(Wg[j].norm()), dt, -1))
    after_abs[j] = abs(a - float(mag_true[j]))
    after_dir[j] = math.degrees(math.acos(min(1.0, abs(float(n @ tn[j])))))
after_rel = after_abs / mag_true.cpu()

print(f"REAL sigmoid teacher 784x128x80x40x32x10, first layer W1=128x784, {k} neurons")
print(f"  true row-norm ||w||: min {float(mag_true.min()):.3f} mean {float(mag_true.mean()):.3f} max {float(mag_true.max()):.3f}")
print("MAGNITUDE (row-norm) error   |  BEFORE (guess)        ->  AFTER (recovered, exp-tail)")
print(f"  absolute:  median {float(before_abs.median()):.3e} max {float(before_abs.max()):.3e}  ->  median {float(after_abs.median()):.3e} max {float(after_abs.max()):.3e}")
print(f"  relative:  median {float(before_rel.median()):.3e} max {float(before_rel.max()):.3e}  ->  median {float(after_rel.median()):.3e} max {float(after_rel.max()):.3e}")
print("DIRECTION (deg)              |  BEFORE 5.000  ->  AFTER: median {:.2e} max {:.2e}".format(float(after_dir.median()), float(after_dir.max())))
