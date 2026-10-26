"""Per-frequency QK recovery of the WIDE members of archived checkpoints (diagnostic; uses the teacher to fit the
residual transform, like layer_report.py -- not a teacher-free score). usage: python freq_check.py ckpt1 [ckpt2 ...]"""
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, '..')
from layer_report import stages, rms, build, dev, canon_, rope_cos_sin
torch.set_grad_enabled(False)
def pairs(Wq, Wk):
    hh = Wq.shape[0] // 2
    return torch.stack([torch.outer(torch.complex(Wq[p], Wq[p + hh]), torch.complex(Wk[p], Wk[p + hh]).conj()) for p in range(hh)])
for f in sys.argv[1:]:
    ck = torch.load(f, map_location=dev, weights_only=False); cfg = ck['cfg']
    X = torch.randn(64, 256, 16, generator=torch.Generator(device=dev).manual_seed(5), device=dev) * 0.5; cos, sin = rope_cos_sin(256, 16, cfg['theta'], dev)
    t = canon_(build(cfg, ck['teacher_sd'], dev), True); St, _ = stages(t, X, cos, sin); Yt = t(inputs_embeds=X); res = []
    for mi in range(len(ck['student_sds'])):
        s = canon_(build(ck['student_cfg'], ck['student_sds'][mi], dev), True); Ss, _ = stages(s, X, cos, sin)
        L = (s(inputs_embeds=X) - Yt).abs().mean().item(); row = []
        for b, k in [(0, 'in'), (1, 'B1.mlp')]:
            A = torch.linalg.lstsq(St[k], Ss[k]).solution.t(); r = (rms(St[k]) / rms(St[k] @ A.t())).median().item()
            Pt = pairs(t.blocks[b].attn.q_proj.weight.double(), t.blocks[b].attn.k_proj.weight.double())
            Ps = pairs(s.blocks[b].attn.q_proj.weight.double() @ A * r, s.blocks[b].attn.k_proj.weight.double() @ A * r)
            row.append([((Ps[p] - Pt[p]).abs().pow(2).sum() / Pt[p].abs().pow(2).sum()).sqrt().item() for p in range(8)])
        res.append((L, mi, row))
    res.sort()
    good = lambda b, th: sum(all(x < th for x in r[2][b]) for r in res)  # noqa: E731
    print(f"\n{f.split('/')[-1]} (iter {ck['iter']}): members with ALL 8 pairs < 5%: B1 {good(0, .05)}/8, B2 {good(1, .05)}/8")
    for L, mi, row in res[:3]:
        print(f"  m{mi} T=256 L1 {L:.3f} | B1 " + " ".join(f"{x:5.1%}" for x in row[0]) + " | B2 " + " ".join(f"{x:5.1%}" for x in row[1]))
