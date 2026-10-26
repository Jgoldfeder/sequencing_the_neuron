"""Absolute per-entry weight error (max/mean) in the teacher's coordinates for every member of a ckpt
(residual gauge A + RMSNorm scale undone; MLP fc1 rows matched by pre-activation, sign-flip aware)."""
import sys, torch
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from layer_report import stages, rms, zcorr, build, dev, canon_, rope_cos_sin, qk_circuit, ov_circuit
torch.set_grad_enabled(False)
ck = torch.load(sys.argv[1], map_location=dev, weights_only=False); cfg = ck['cfg']
X = torch.randn(256, 32, cfg['d_model'], generator=torch.Generator(device=dev).manual_seed(5), device=dev) * 0.5
cos, sin = rope_cos_sin(32, 16, cfg['theta'], dev)
t = canon_(build(cfg, ck['teacher_sd'], dev), True); St, _ = stages(t, X, cos, sin)
mm = lambda d: f"{d.abs().max():.1e}/{d.abs().mean():.1e}"
tq = [qk_circuit(b, 0).double() for b in t.blocks]; to = [ov_circuit(b, 0).double() for b in t.blocks]
print(f"{sys.argv[1].split('/')[-1]}  iter {ck['iter']}   (max/mean abs error per entry)")
print(f"  {'member':12s} {'B1 qk':>15s} {'B1 ov':>15s} {'B1 fc1':>15s} {'B2 qk':>15s} {'B2 ov':>15s} {'B2 fc1':>15s} {'E':>15s}")
for mi in sorted(range(len(ck['student_sds'])), key=lambda i: ck['fit'][i]):
    s = canon_(build(ck.get('student_cfg', cfg), ck['student_sds'][mi], dev), True); Ss, _ = stages(s, X, cos, sin)
    A = {k: torch.linalg.lstsq(St[k], Ss[k]).solution.t() for k in St}
    r = {k: (rms(St[k]) / rms(St[k] @ A[k].t())).median().item() for k in St}
    cells = []
    for i, (ka, km) in enumerate([('in', 'B1.attn'), ('B1.mlp', 'B2.attn')]):
        tb, sb = t.blocks[i], s.blocks[i]; Aa, ra, Aw = A[ka], r[ka], A[f'B{i+1}.attn']
        qk = (ra ** 2) * Aa.t() @ qk_circuit(sb, 0).double() @ Aa
        ov = ra * torch.linalg.solve(Aw, ov_circuit(sb, 0).double()) @ Aa
        W1t = tb.mlp.fc1.weight.double(); W1s = sb.mlp.fc1.weight.double() @ A[km] * r[km]
        hn = St[km] / rms(St[km]); C = zcorr((hn @ W1t.t()).float(), (hn @ W1s.t()).float())
        live = (W1t.norm(dim=1) > 0.01 * W1t.norm(dim=1).median()).nonzero().flatten()
        d = torch.stack([torch.sign(C[u, C[u].abs().argmax()]).double() * W1s[C[u].abs().argmax()] - W1t[u] for u in live.tolist()])
        cells += [mm(qk - tq[i]), mm(ov - to[i]), mm(d)]
    Es = s.embed.weight.double() @ A['B2.mlp'] * r['B2.mlp']
    cells.append(mm(Es - t.embed.weight.double()))
    print(f"  m{mi} fit {ck['fit'][mi]:.3f} " + " ".join(f"{c:>15s}" for c in cells))
print(f"  {'teacher rms':12s} " + " ".join(f"{v:>15.2f}" for v in [tq[0].pow(2).mean().sqrt(), to[0].pow(2).mean().sqrt(), t.blocks[0].mlp.fc1.weight.pow(2).mean().sqrt(), tq[1].pow(2).mean().sqrt(), to[1].pow(2).mean().sqrt(), t.blocks[1].mlp.fc1.weight.pow(2).mean().sqrt(), t.embed.weight.pow(2).mean().sqrt()]))
