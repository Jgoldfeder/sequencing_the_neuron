"""Orphan layer-1 unit recovery (black-box) for a deep tanh net: 199 rows (direction + scale) known,
one unit without a usable committee guess. The row-space complement c (orthogonal to all known rows)
changes ONLY the orphan's pre-activation. Saturate the 199 known units with an offset t v (v chosen so
min_j |w_j . v| is large, t so every |z_j| > 8), scan along c for the orphan's derivative peak (its
pre-activation zero), and read its row off the rank-one Jacobian there; then hand that direction to the
standard isolation rounds as the guess. Usage: python peel/diag/orphan_l1.py <guess.pt> <iso_merged.pt> <scales.pt> <unit> [gpu]"""
import sys, os, subprocess, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from nets import MLP
from tanh_solve import Oracle, measure_jets
guess_path, iso_path, sc_path, unit = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); dev = f"cuda:{sys.argv[5] if len(sys.argv) > 5 else 0}"
ck = torch.load(guess_path, map_location=dev, weights_only=False); dims = ck["dims"]
teacher = MLP(dims, act="tanh").to(dev); teacher.load_state_dict(ck["teacher_state"]); teacher = teacher.double()
bb = Oracle(lambda x: teacher(x.double())); d = dims[0]
iso = torch.load(iso_path, map_location="cpu", weights_only=False); res = iso["results"]
sc = torch.load(sc_path, map_location="cpu", weights_only=False)["results"]
known = [i for i in sorted(res) if i != unit and sc.get(i, {}).get("scale")]
U = torch.stack([res[i]["direction"].double() for i in known]).to(dev)
W = U * torch.tensor([sc[i]["scale"] for i in known], dtype=torch.float64, device=dev)[:, None]   # scaled rows
torch.manual_seed(0)
X = torch.randn(40, d, dtype=torch.float64, device=dev); _, J = measure_jets(bb, X, None, h=1e-3)
S, Vh = torch.linalg.svd(J.reshape(-1, d), full_matrices=False)[1:]; Vsub = Vh[:int((S > 1e-8 * S[0]).sum())]
Q, _ = torch.linalg.qr(U.T)
c = torch.randn(d, dtype=torch.float64, device=dev)
for _ in range(50):
    c = Vsub.T @ (Vsub @ c); c = c - Q @ (Q.T @ c)
c = c / c.norm()
print(f"known rows {len(U)} | complement: max |u_j.c| {float((U @ c).abs().max()):.1e}", flush=True)
# saturating offset direction: best of 256 random row-space directions by min_j |w_j . v|
best = None
for _ in range(256):
    v = Vsub.T @ (Vsub @ torch.randn(d, dtype=torch.float64, device=dev)); v = v - (v @ c) * c; v = v / v.norm()
    m = float((W @ v).abs().min())
    if best is None or m > best[0]: best = (m, v)
mmin, v = best; t = 8.0 / mmin
print(f"offset v: min_j |w_j.v| {mmin:.3e} -> t {t:.0f} (all known |z_j| >= 8; others' leakage sech^2(8) ~ 1e-6)", flush=True)
Wt = teacher.layers[0].weight.detach().double(); Wtn = Wt / Wt.norm(dim=1, keepdim=True)
j_true = int((U @ Wtn.T).abs().max(0).values.argmin())
def scan(x0, S_win, n):
    s = torch.linspace(-S_win, S_win, n, dtype=torch.float64, device=dev)
    P = x0[None] + s[:, None] * c[None]; h = 1e-3
    g = ((bb(P + h * c) - bb(P - h * c)) / (2 * h)).norm(dim=1)
    return s, g
x0 = t * v
s, g = scan(x0, 20000.0, 40001)                                   # coarse: step 0.5 along c
k = int(g.argmax()); s1 = float(s[k]); print(f"coarse scan: peak |dF/ds| {float(g[k]):.3e} at s {s1:.1f} (median {float(g.median()):.1e})", flush=True)
s, g = scan(x0 + s1 * c, 2.0, 4001); s2 = s1 + float(s[g.argmax()])
x_star = x0 + s2 * c
_, Jx = measure_jets(bb, x_star[None], None, h=1e-3); sv, vh = torch.linalg.svd(Jx[0], full_matrices=False)[1:]
w = vh[0]; err = float(min((w - Wtn[j_true]).norm(), (w + Wtn[j_true]).norm()))
print(f"at the centre: Jacobian sv ratio s2/s1 {float(sv[1]/sv[0]):.1e} | orphan direction err vs teacher unit {j_true} (scoring only) {err:.1e} | {bb.n} queries", flush=True)
# hand the direction to the standard isolation rounds as the guess for this unit
g2 = torch.load(guess_path, map_location="cpu", weights_only=False)
g2["state_dict"]["layers.0.weight"][unit] = (w.cpu() * 4.0).float()
gp = guess_path.replace(".pt", f"_orphan{unit}.pt"); torch.save(g2, gp)
out = iso_path.replace(".pt", f"_orphan{unit}.pt")
subprocess.run(f"CUDA_VISIBLE_DEVICES={dev.split(':')[1]} /opt/miniconda3/bin/python peel/isolate_l1_deep.py {guess_path} {gp} {unit} {out} 0 > results/logs/iso_l1_orphan{unit}.log 2>&1", shell=True)
r = torch.load(out, map_location="cpu", weights_only=False)["results"][unit]
print(f"isolation rounds from the orphan guess: err {r['err_hist'][0]:.1e} -> {r['err_hist'][-1]:.1e}", flush=True)
res[unit] = dict(r, orphan=True); torch.save({"results": res, "dims": dims}, iso_path); print("merged file updated", flush=True)
