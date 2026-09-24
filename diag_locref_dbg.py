"""Debug: replicate _collect_neuron with counters + solve diagnostics."""
import torch

import loc_refine as LR
from verify_layer1 import _Oracle
from nets import MLP
from diag_locref import perturb_rows

dev = "cuda"
dims = [3072, 200, 200, 200, 100]
frontier = 1
torch.manual_seed(0)
teacher = MLP(dims).double().to(dev).eval()
cons = teacher.clone()
gen = torch.Generator(device=dev).manual_seed(1)
perturb_rows(cons.layers[frontier], 1e-2, gen)

pre = cons
orc = _Oracle(teacher)
Wl = pre.layers[frontier].weight.detach()
bl = pre.layers[frontier].bias.detach()
c = 0
wg, bg = Wl[c], bl[c].reshape(())
Din = Wl.shape[1]
idim = dims[0]
wbn = float(torch.cat([wg, bg.reshape(1)]).norm())
g2 = torch.Generator(device=dev).manual_seed(99)
xcap = 4.0 * idim ** 0.5
LAD = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)

want = 2 * Din
B = 4 * want
sc = 0.3 + 3.5 * torch.rand(B, 1, generator=g2, device=dev, dtype=torch.float64)
seeds = torch.randn(B, idim, generator=g2, device=dev, dtype=torch.float64) * sc
X0, Uh, Nu, ok = LR._project_batch(pre, seeds, wg, bg, frontier)
keep = ok & (X0.norm(dim=1) <= xcap)
X0, Uh, Nu = X0[keep], Uh[keep], Nu[keep]
H0 = LR._phi(pre, X0, frontier)
print(f"pool: {len(X0)}")

hs = []
cnt = {"iso": 0, "dirty": 0, "exhaust": 0, "ok": 0}
ok_rung = {}
iso_rung = {}
q0 = orc.n
for idx in LR._greedy_span(H0, want):
    if len(hs) >= want:
        break
    x0, uh, nu = X0[idx], Uh[idx], float(Nu[idx])
    hn = float(torch.cat([H0[idx], torch.ones(1, dtype=torch.float64, device=dev)]).norm())
    done = False
    for ri, eps in enumerate(LAD):
        delta = 3.0 * eps * wbn * hn / max(nu, 1e-30)
        if not LR._isolation_ok(pre, x0, uh, delta, frontier, Wl, bl, c):
            cnt["iso"] += 1; iso_rung[ri] = iso_rung.get(ri, 0) + 1
            done = True
            break
        st, t = LR._locate(orc, x0, uh, delta)
        if st == "ok":
            hs.append(LR._phi(pre, (x0 + t * uh).unsqueeze(0), frontier).squeeze(0))
            cnt["ok"] += 1; ok_rung[ri] = ok_rung.get(ri, 0) + 1
            done = True
            break
        if st == "dirty":
            cnt["dirty"] += 1
            done = True
            break
    if not done:
        cnt["exhaust"] += 1
print(f"outcomes: {cnt}   queries: {orc.n - q0}")
print(f"ok by rung:  {dict(sorted(ok_rung.items()))}")
print(f"iso by rung: {dict(sorted(iso_rung.items()))}")

if len(hs) >= Din + 3:
    v = LR._solve(hs, Din)
    M = torch.cat([torch.stack(hs),
                   torch.ones(len(hs), 1, dtype=torch.float64, device=dev)], 1)
    _, S, Vh = torch.linalg.svd(M, full_matrices=False)
    print(f"solve: gap S[-2]/S[-1] = {float(S[-2]/S[-1]):.3e}  "
          f"(gate 1e5) -> {'None' if v is None else 'ok'}")
    vt = torch.cat([teacher.layers[frontier].weight.data[c],
                    teacher.layers[frontier].bias.data[c].reshape(1)])
    vt = vt / vt.norm()
    vv = Vh[-1] if v is None else v
    if (vv @ vt) < 0:
        vv = -vv
    print(f"eps vs true (untrimmed null vec): {float((vv - vt).abs().max()):.3e}")
else:
    print(f"only {len(hs)} points -- not enough to solve")
