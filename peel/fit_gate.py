"""Clean fit-capacity gate for Expand-and-Cluster: does WIDTH let a student fit
the teacher, decoupled from the pipeline's confounds (adversarial moving target,
per-iter budget, factorization)? Fixed teacher-labeled query set, train each
width to convergence, report held-out MAE (the pipeline's loss metric) + MSE.

If baseline width already fits low -> the pipeline underfitting is the adversarial
arms race, not capacity (expansion irrelevant). If baseline can't fit but wide
can -> expansion is the lever. If nothing fits -> deep-sigmoid regression is the
wall.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import make_teacher            # noqa: E402
from nets import MLP                     # noqa: E402

dev = "cuda"
dims = [3072, 1024, 512, 100]
teacher = make_teacher(dims, epochs=25, seed=0, device=dev, act="sigmoid")
teacher.eval()

g = torch.Generator(device=dev).manual_seed(0)
NTR, NTE = 120000, 20000
# uniform[-1,1]: the pipeline's non-adversarial query regime (qg_range=1)
Xtr = (torch.rand(NTR, dims[0], generator=g, device=dev) * 2 - 1)
Xte = (torch.rand(NTE, dims[0], generator=g, device=dev) * 2 - 1)
with torch.no_grad():
    Ytr = teacher(Xtr)
    Yte = teacher(Xte)
print(f"[teacher] {dims}  |Y| std={Yte.std():.3f} mean|Y|={Yte.abs().mean():.3f}  "
      f"(baseline pipeline loss ~0.43-0.55 MAE for reference)\n", flush=True)


@torch.no_grad()
def evalloss(net, X, Y, bs=8192):
    mae = mse = n = 0.0
    for i in range(0, len(X), bs):
        p = net(X[i:i + bs])
        d = p - Y[i:i + bs]
        mae += d.abs().sum().item(); mse += (d ** 2).sum().item(); n += d.numel()
    return mae / n, mse / n


def train_width(mult, epochs=160, lr=1e-3, bs=512):
    sd = [dims[0]] + [dims[i] * mult for i in range(1, len(dims) - 1)] + [dims[-1]]
    torch.manual_seed(1)
    net = MLP(sd, act="sigmoid").to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=epochs // 3, gamma=0.3)
    t0 = time.time()
    print(f"=== width x{mult}  {sd}  ({sum(p.numel() for p in net.parameters())/1e6:.1f}M params) ===",
          flush=True)
    for ep in range(epochs):
        perm = torch.randperm(NTR, device=dev)
        for i in range(0, NTR, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            ((net(Xtr[idx]) - Ytr[idx]) ** 2).mean().backward()
            opt.step()
        sched.step()
        if ep % 20 == 0 or ep == epochs - 1:
            trm, trs = evalloss(net, Xtr, Ytr)
            tem, tes = evalloss(net, Xte, Yte)
            print(f"  x{mult} ep{ep:3d} | train MAE {trm:.4f} MSE {trs:.2e} | "
                  f"test MAE {tem:.4f} MSE {tes:.2e} | {round(time.time()-t0)}s",
                  flush=True)


if __name__ == "__main__":
    mults = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "2", "4"])]
    for m in mults:
        train_width(m)
        print(flush=True)
