"""Is the basin a PER-NEURON (max) or COLLECTIVE (mean) criterion?
x0 = Wg^+(t-bg) uses the WHOLE guess matrix, so bad rows could corrupt good rows' isolation.
Test: half the neurons at tiny error, half at large error -> do the good ones still recover?"""
import sys, math, torch
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
dev = "cuda"
CKPT = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt"
dims = [784, 128, 80, 40, 32, 10]; d = dims[0]; k = dims[1]
teacher = MLP(dims, act="sigmoid").to(dev).double(); teacher.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)); teacher.eval()
W1t = teacher.layers[0].weight.detach(); b1t = teacher.layers[0].bias.detach(); tn = W1t / W1t.norm(dim=1, keepdim=True)
@torch.no_grad()
def J_at(x, fd=5e-5):
    E = torch.eye(d, device=dev, dtype=torch.float64)
    return ((teacher(x.unsqueeze(0) + fd * E) - teacher(x.unsqueeze(0) - fd * E)) / (2 * fd)).t()
def recover(Wg, bg):
    Wgpinv = Wg.t() @ torch.linalg.inv(Wg @ Wg.t()); errs = torch.zeros(k)
    for j in range(k):
        t = torch.full((k,), 20.0, device=dev, dtype=torch.float64); t[j] = 0.0
        x0 = Wgpinv @ (t - bg); U, Sv, Vh = torch.linalg.svd(J_at(x0), full_matrices=False)
        nv = Vh[0]; nv = nv if float(nv @ Wg[j]) > 0 else -nv
        errs[j] = math.degrees(math.acos(min(1.0, abs(float(nv @ tn[j])))))
    return errs
g = torch.Generator(device=dev).manual_seed(7)
R = torch.randn(k, d, generator=g, device=dev, dtype=torch.float64); R = R - (R * tn).sum(1, keepdim=True) * tn; R = R / R.norm(dim=1, keepdim=True)
eps = torch.zeros(k, device=dev, dtype=torch.float64)
GOOD = torch.arange(0, k, 2); BAD = torch.arange(1, k, 2)          # interleave 64 good / 64 bad
eps[GOOD] = 0.02
for lvl in [0.20, 0.40, 0.80]:
    eps[BAD] = lvl
    Wg = W1t + eps[:, None] * W1t.norm(dim=1, keepdim=True) * R
    bg = b1t + eps * b1t.abs() * (2 * torch.rand(k, generator=g, device=dev, dtype=torch.float64) - 1)
    r = recover(Wg, bg)
    print(f"BAD group at eps={lvl:.2f} (GOOD group fixed at 0.02):")
    print(f"   GOOD (eps 0.02): recov dir err med {float(r[GOOD].median()):.2e} max {float(r[GOOD].max()):.2e} deg")
    print(f"   BAD  (eps {lvl:.2f}): recov dir err med {float(r[BAD].median()):.2e} max {float(r[BAD].max()):.2e} deg")
