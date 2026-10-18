"""Why does C5 learn under DIRECT 400-dim queries but not through the prefix? Same tail MLP (LeNet's C5->F6->head),
same optimizer, only the query distribution differs:
  gauss     : N(0, 0.5^2)^400                       (direct access; 83/120 in basin at 30k in diag_deep_relu_gd)
  phi       : pooled conv2 activations of random images N(0,0.5^2)^784 (what the prefix lets through)
  phi_gauss : Gaussian with phi's mean and covariance (same second moments, no cone constraint)
  absgauss  : |N(0,0.5^2)| (nonnegative cone, isotropic)
  phi_x2    : phi of images with std 1.0 (wider images -> wider cone)
Metric: aligned per-row max error of the tail's first layer (C5), # rows < 1e-2, one-sided count on the query dist.
"""
import argparse, json, time, torch
from nets import MLP
from data import make_teacher, make_teacher_cnn
from diag_deep_relu_gd import score, hidden_preacts
ap = argparse.ArgumentParser(); ap.add_argument("--arms", default="gauss,phi,phi_gauss,absgauss,phi_x2"); ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="results/diag/tail_inputdist.json"); a = ap.parse_args(); dev = a.device
dims = [400, 120, 84, 10]
teacher = make_teacher(dims, 25, 0, device=dev, verbose=False).to(dev)
lenet = make_teacher_cnn((1, 28, 28), [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)], (84,), 10, epochs=25, seed=0, device=dev, act="leaky_relu")
for p in list(teacher.parameters()) + list(lenet.parameters()): p.requires_grad_(False)
@torch.no_grad()
def phi(n, std=0.5):
    x = (torch.randn(n, 784, device=dev) * std).view(n, 1, 28, 28)
    for i in range(2): x = torch.nn.functional.avg_pool2d(lenet.act(lenet.layers[i](x)), 2)
    return x.flatten(1)
with torch.no_grad():
    P = phi(20000); mu = P.mean(0); C = torch.cov(P.T) + 1e-6 * torch.eye(400, device=dev); Lc = torch.linalg.cholesky(C)
draws = {"gauss": lambda n: torch.randn(n, 400, device=dev) * 0.5, "phi": lambda n: phi(n),
         "phi_gauss": lambda n: mu + torch.randn(n, 400, device=dev) @ Lc.T, "absgauss": lambda n: (torch.randn(n, 400, device=dev) * 0.5).abs(),
         "phi_x2": lambda n: phi(n, 1.0)}
res = {}
for arm in a.arms.split(","):
    draw = draws[arm]; torch.manual_seed(0); net = MLP(dims).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    with torch.no_grad():
        xq = draw(8192); z = teacher.layers[0](xq); pos = (z > 0).float().mean(0); onesided = int((torch.minimum(pos, 1 - pos) < 0.01).sum())
    xeval = torch.randn(4096, 400, device=dev) * 0.5; t0 = time.time(); rec = []
    for s in range(1, a.steps + 1):
        x = draw(512)
        with torch.no_grad(): y = teacher(x)
        opt.zero_grad(); (net(x) - y).abs().mean().backward(); opt.step()
        if s % 5000 == 0:
            layers, head = score(net, teacher, xeval); d = layers[0]
            rec.append(dict(step=s, med=d["med"], n_1e2=d["n_1e2"], flips=d["flips"]))
            print(f"  [{arm}] step {s} | C5 median {d['med']:.1e}, rows<1e-2 {d['n_1e2']}/120, flips {d['flips']} | one-sided C5 channels on this query dist: {onesided} | {time.time()-t0:.0f}s", flush=True)
    res[arm] = dict(onesided=onesided, rec=rec); json.dump(res, open(a.out, "w"), indent=1)
print("done")
