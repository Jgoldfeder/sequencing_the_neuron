"""lean_chain.py -- exact recovery of a deep (leaky) ReLU MLP from a committee guess with the lean kink refiner.

    python lean_chain.py <pop snapshot .pt> [--member best|IDX] [--h 1e-5] [--eps-side 1e-4] [--tries 6] [--out out.pt]

Layer by layer: refine every unit of layer l with lean_kink (prefix = the already refined layers), install the refined
row at the guess's magnitude (keeps the next layer's guess in its own gauge), then the next layer.  Units the refiner
rejects keep their guess and are retried once with fresh base points.  Output layer: closed-form least squares on
fresh Gaussian queries.  Black box = fp64 teacher forward only; the teacher's parameters are read only for scoring.
"""
import argparse, time, torch
from nets import MLP
from method import param_errors
import lean_kink as lk

ap = argparse.ArgumentParser()
ap.add_argument("snapshot"); ap.add_argument("--member", default="best")
ap.add_argument("--h", type=float, default=1e-5); ap.add_argument("--eps-side", type=float, default=1e-4)
ap.add_argument("--tries", type=int, default=6); ap.add_argument("--head-points", type=int, default=20000)
ap.add_argument("--out", default=""); ap.add_argument("--device", default="cuda")
a = ap.parse_args(); dev = a.device
d = torch.load(a.snapshot, map_location=dev, weights_only=False)
dims, act = d["dims"], d["act"]
t64 = MLP(dims, act=act).to(dev); t64.load_state_dict(d["teacher_state"]); t64 = t64.double()
for p in t64.parameters():
    p.requires_grad_(False)
Q = {"n": 0}
def bb(X):
    Q["n"] += X.shape[0]
    with torch.no_grad():
        return t64(X)
X = torch.randn(4096, dims[0], device=dev, dtype=torch.float64) * 0.5
mems = []
for s in d["pop_states"]:
    m = MLP(dims, act=act).to(dev); m.load_state_dict(s); mems.append(m.double())
if a.member == "best":
    errs = [float((m(X) - bb(X)).abs().mean()) for m in mems]; bi = min(range(len(mems)), key=lambda i: errs[i])
else:
    bi = int(a.member)
net = mems[bi]
for p in net.parameters():
    p.requires_grad_(False)
q0 = Q["n"]; t0 = time.time()
print(f"[lean-chain] guess = member {bi} of {len(mems)} | dims {dims}", flush=True)
e = param_errors(net, t64); print(f"[lean-chain] start: max_eps {e['max_eps']:.3e}", flush=True)
Lh = len(dims) - 2
for l in range(Lh):
    tl = time.time()
    oracle = lk.Oracle(bb)
    W = net.layers[l].weight; B = net.layers[l].bias
    n = W.shape[0]; ok = torch.zeros(n, dtype=torch.bool, device=dev)
    for rnd, seed in enumerate((l, 1000 + l)):
        gen = torch.Generator(device=dev).manual_seed(seed)
        for j in [j for j in range(n) if not ok[j]]:
            out = lk.refine_unit(oracle, net, l, W[j].double(), B[j].double(), gen, h=a.h, eps_side=a.eps_side, tries=a.tries)
            if out is not None:
                r, rb, _ = out
                mag = float(torch.cat([W[j], B[j:j + 1]]).norm()) / float(torch.cat([r, torch.tensor([rb], device=dev, dtype=r.dtype)]).norm())
                W[j] = (mag * r).to(W.dtype); B[j] = mag * rb; ok[j] = True
    ep = param_errors(net, t64)["max_eps_per_matrix"]
    print(f"[lean-chain] L{l + 1}: {int(ok.sum())}/{n} refined | {oracle.n:,} queries ({oracle.n / n:.0f}/unit) | "
          f"{time.time() - tl:.0f}s | L{l + 1} max eps W {ep[2 * l]:.2e} b {ep[2 * l + 1]:.2e}", flush=True)
with torch.no_grad():
    Xh = torch.randn(a.head_points, dims[0], device=dev, dtype=torch.float64)
    Y = bb(Xh); H = Xh
    for lay in net.layers[:-1]:
        H = net.act(lay(H))
    A = torch.cat([H, torch.ones(len(H), 1, device=dev, dtype=H.dtype)], 1)
    sol = torch.linalg.lstsq(A, Y).solution
    net.layers[-1].weight.copy_(sol[:-1].T); net.layers[-1].bias.copy_(sol[-1])
e = param_errors(net, t64)
print(f"[lean-chain] head: closed-form LSQ on {a.head_points:,} queries", flush=True)
print(f"[lean-chain] FINAL max_eps {e['max_eps']:.3e} mean {sum(e['mean_eps_per_matrix']) / len(e['mean_eps_per_matrix']):.3e} | "
      f"per-matrix max {' '.join(f'{v:.1e}' for v in e['max_eps_per_matrix'])}", flush=True)
print(f"[lean-chain] refiner + head queries {Q['n'] - q0:,} | {time.time() - t0:.0f}s", flush=True)
if a.out:
    torch.save({"dims": dims, "act": act, "state_dict": net.state_dict(), "queries": Q["n"] - q0}, a.out)
    print("[lean-chain] saved", a.out, flush=True)
