"""Can the prefix route be rescued by choosing IMAGES whose prefix activations look like direct Gaussian queries?
Same tail MLP (LeNet C5->F6->head, exact prefix conv1+conv2 used only to map images), single student, Adam 1e-3.
  phi      : random images N(0,0.5^2)^784 -> pooled conv2 activations (baseline: 0/120 rows in basin)
  inverse  : per batch, start from random images and run `--inv-steps` Adam steps on the pixels so that
             phi(x) matches a Gaussian target g ~ N(0,0.5^2)^400 (least squares); train on the achieved phi(x)
  straddle : per batch, pixels optimised so that the STUDENT's own C5 pre-activations have zero mean per unit over
             the batch and unit variance (attacker-legal: uses the student, not the teacher)
Reports C5 rows in the refiner basin and the one-sided count of TEACHER C5 channels on the achieved query distribution.
"""
import argparse, json, time, torch
from nets import MLP
from data import make_teacher, make_teacher_cnn
from diag_deep_relu_gd import score
ap = argparse.ArgumentParser(); ap.add_argument("--arms", default="phi,inverse,straddle"); ap.add_argument("--steps", type=int, default=15000)
ap.add_argument("--inv-steps", type=int, default=30); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="results/diag/tail_inverse_sampling.json")
a = ap.parse_args(); dev = a.device; dims = [400, 120, 84, 10]
teacher = make_teacher(dims, 25, 0, device=dev, verbose=False).to(dev)
lenet = make_teacher_cnn((1, 28, 28), [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)], (84,), 10, epochs=25, seed=0, device=dev, act="leaky_relu")
for p in list(teacher.parameters()) + list(lenet.parameters()): p.requires_grad_(False)
def phi(x):
    x = x.view(-1, 1, 28, 28)
    for i in range(2): x = torch.nn.functional.avg_pool2d(lenet.act(lenet.layers[i](x)), 2)
    return x.flatten(1)
def draw_phi(n): 
    with torch.no_grad(): return phi(torch.randn(n, 784, device=dev) * 0.5)
def draw_inverse(n):
    g = torch.randn(n, 400, device=dev) * 0.5
    x = (torch.randn(n, 784, device=dev) * 0.5).requires_grad_(True); opt = torch.optim.Adam([x], lr=0.05)
    for _ in range(a.inv_steps):
        opt.zero_grad(); ((phi(x) - g) ** 2).mean().backward(); opt.step()
    with torch.no_grad(): return phi(x)
def draw_extreme(n, steps=None, lr=0.1, box=3.0, mix=0.5):
    # attacker-legal, unit-agnostic: push images so that phi(x) is EXTREME along random directions of the 400-space
    # (maximise sign * u.phi(x)); half the batch random images for coverage. Visits the reachable set's boundary.
    steps = steps or a.inv_steps
    m = int(n * mix)
    u = torch.randn(n - m, 400, device=dev); u = u / u.norm(dim=1, keepdim=True)
    sgn = torch.where(torch.rand(n - m, 1, device=dev) < 0.5, -1.0, 1.0)
    x = (torch.randn(n - m, 784, device=dev) * 0.5).requires_grad_(True); opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        opt.zero_grad(); (-(sgn * u * phi(x)).sum(1)).mean().backward(); opt.step()
        with torch.no_grad(): x.clamp_(-box, box)
    with torch.no_grad():
        return torch.cat([phi(x), phi(torch.randn(m, 784, device=dev) * 0.5)]) if m else phi(x)
def draw_straddle(n, net):
    x = (torch.randn(n, 784, device=dev) * 0.5).requires_grad_(True); opt = torch.optim.Adam([x], lr=0.05)
    for _ in range(a.inv_steps):
        opt.zero_grad(); z = net.layers[0](phi(x)); loss = (z.mean(0) ** 2).mean() + ((z.std(0) - 1) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad(): return phi(x)
res = {}
for arm in a.arms.split(","):
    torch.manual_seed(0); net = MLP(dims).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    draw = {"phi": lambda n: draw_phi(n), "inverse": lambda n: draw_inverse(n), "straddle": lambda n: draw_straddle(n, net),
            "extreme": lambda n: draw_extreme(n), "extreme_only": lambda n: draw_extreme(n, mix=0.0),
            "extreme_box10": lambda n: draw_extreme(n, box=10.0, lr=0.3)}[arm]
    xeval = torch.randn(4096, 400, device=dev) * 0.5; t0 = time.time(); rec = []
    for s in range(1, a.steps + 1):
        x = draw(512)
        with torch.no_grad(): y = teacher(x)
        opt.zero_grad(); (net(x) - y).abs().mean().backward(); opt.step()
        if s % 2500 == 0:
            xq = torch.cat([draw(512) for _ in range(4)])          # (the draw itself may need autograd)
            with torch.no_grad():
                z = teacher.layers[0](xq); pos = (z > 0).float().mean(0); onesided = int((torch.minimum(pos, 1 - pos) < 0.01).sum())
                print(f"    [{arm}] achieved query dist: mean {float(xq.mean()):.3f} std {float(xq.std()):.3f} frac<0 {float((xq < 0).float().mean()):.2f}", flush=True)
            layers, head = score(net, teacher, xeval); d = layers[0]
            rec.append(dict(step=s, med=d["med"], n_1e2=d["n_1e2"], onesided=onesided))
            print(f"  [{arm}] step {s} | C5 median {d['med']:.1e}, rows<1e-2 {d['n_1e2']}/120 | teacher C5 one-sided on this dist: {onesided} | {time.time()-t0:.0f}s", flush=True)
    res[arm] = rec; json.dump(res, open(a.out, "w"), indent=1)
print("done")
