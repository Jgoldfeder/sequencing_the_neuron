"""How good must the first-layer GUESS be for the isolation/refinement to work?
Perturb the true (w,b) by relative eps, run direction recovery (saturate+SVD), measure
the recovered direction error. Sweeps eps to find where isolation breaks."""
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
def recover_dirs(Wg, bg):
    Wgpinv = Wg.t() @ torch.linalg.inv(Wg @ Wg.t()); errs = torch.zeros(k)
    for j in range(k):
        t = torch.full((k,), 20.0, device=dev, dtype=torch.float64); t[j] = 0.0
        x0 = Wgpinv @ (t - bg)
        U, Sv, Vh = torch.linalg.svd(J_at(x0), full_matrices=False)
        nv = Vh[0]; nv = nv if float(nv @ Wg[j]) > 0 else -nv
        errs[j] = math.degrees(math.acos(min(1.0, abs(float(nv @ tn[j])))))
    return errs
print("guess quality (relative w-error) -> recovered DIRECTION error (deg), and the guess's own dir error:")
print(f"{'eps':>7} {'guess_dir_med':>14} {'guess_dir_max':>14} | {'recov_dir_med':>14} {'recov_dir_max':>14}")
for eps in [0.006, 0.03, 0.10, 0.20, 0.40, 0.80]:
    g = torch.Generator(device=dev).manual_seed(7)
    R = torch.randn(k, d, generator=g, device=dev, dtype=torch.float64)
    R = R - (R * tn).sum(1, keepdim=True) * tn; R = R / R.norm(dim=1, keepdim=True)   # perpendicular
    Wg = W1t + eps * W1t.norm(dim=1, keepdim=True) * R                                 # rel weight err = eps
    bg = b1t + eps * b1t.abs() * (2 * torch.rand(k, generator=g, device=dev, dtype=torch.float64) - 1)
    gdir = torch.tensor([math.degrees(math.acos(min(1.0, abs(float(Wg[j] @ tn[j] / Wg[j].norm()))))) for j in range(k)])
    rdir = recover_dirs(Wg, bg)
    print(f"{eps:>7.3f} {float(gdir.median()):>14.3f} {float(gdir.max()):>14.3f} | {float(rdir.median()):>14.2e} {float(rdir.max()):>14.2e}")
