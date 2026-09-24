"""Q1 bridge: rho(x)=sigma2/sigma1 is observable, so OPTIMIZE the probe x to minimize it.
Start from a BAD-guess isolation probe (high rho), descend the distance-to-rank-1, and see if
x drifts into a rank-one region -> recovering SOME true neuron. If yes, the guess only needs to
land in the basin of a rank-one region (much weaker), and it bridges toward guess-free search."""
import sys, math, torch, numpy as np
from torch.func import jacrev
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_default_dtype(torch.float64)
CKPT = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt"
dims = [784, 128, 80, 40, 32, 10]; d, k = dims[0], dims[1]
teacher = MLP(dims, act="sigmoid").to(dev); teacher.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)); teacher.eval()
for p in teacher.parameters(): p.requires_grad_(False)
W1t = teacher.layers[0].weight.detach(); b1t = teacher.layers[0].bias.detach(); tn = W1t / W1t.norm(dim=1, keepdim=True)
f = lambda x: teacher(x.unsqueeze(0))[0]
def Jf(x): return jacrev(f)(x)                                   # (10,784)
def rho_and_dir(x):
    J = Jf(x); U, Sv, Vh = torch.linalg.svd(J, full_matrices=False)
    return float(Sv[1] / Sv[0]), Vh[0]
def bestmatch(nv):                                              # angle to the closest TRUE neuron
    c = (tn @ nv).abs(); j = int(c.argmax()); return j, math.degrees(math.acos(min(1.0, float(c[j]))))
# build a BAD guess (20 deg dir, big bias err) and its isolation probes
g = torch.Generator(device=dev).manual_seed(3); theta = 20.0; eb = 0.6
Wg = torch.empty(k, d, device=dev); bg = torch.empty(k, device=dev)
for j in range(k):
    u = tn[j]; v = torch.randn(d, generator=g, device=dev); v = v - (v @ u) * u; v = v / v.norm()
    Wg[j] = (math.cos(math.radians(theta)) * u + math.sin(math.radians(theta)) * v) * W1t[j].norm()
    bg[j] = b1t[j] + eb * (2 * torch.rand(1, generator=g, device=dev).item() - 1)
Wgpinv = Wg.t() @ torch.linalg.inv(Wg @ Wg.t())
print("start from BAD-guess probes (theta=20deg, bias err 0.6), then minimize rho(x) over x:")
print(f"{'neuron':>7} {'rho0':>9} {'dir0(deg)':>10} -> {'rho*':>9} {'dir*(deg)':>10} {'matched':>8}")
for j in [5, 17, 33, 60, 90, 120]:
    t = torch.full((k,), 20.0, device=dev); t[j] = 0.0
    x = (Wgpinv @ (t - bg)).clone()
    r0, n0 = rho_and_dir(x); _, de0 = bestmatch(n0)
    x = x.requires_grad_(True); opt = torch.optim.Adam([x], lr=0.05)
    for step in range(400):
        J = Jf(x); sv = torch.linalg.svdvals(J)
        loss = 1 - sv[0] ** 2 / (sv ** 2).sum()                # distance-to-rank-1 (smooth)
        opt.zero_grad(); loss.backward(); opt.step()
    r1, n1 = rho_and_dir(x.detach()); mj, de1 = bestmatch(n1)
    print(f"{j:>7} {r0:>9.2e} {de0:>10.2f} -> {r1:>9.2e} {de1:>10.2e} {mj:>8}", flush=True)
