"""B2 MLP input-weight (fc1) error over iterations, from the archived checkpoints in this folder.
Per checkpoint: absolute max/mean per entry for the best member (teacher coords: residual gauge + RMSNorm
scale undone, neurons matched by pre-activation, sign-flip aware) + median of the mean over all members.
usage: python track_b2.py [wide2 wide4 ...]"""
import glob, os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from layer_report import stages, rms, zcorr, build, dev, canon_, rope_cos_sin
torch.set_grad_enabled(False)


def b2_err(ck, t, St, X, cos, sin, mi):
    s = canon_(build(ck.get('student_cfg', ck['cfg']), ck['student_sds'][mi], dev), True); Ss, _ = stages(s, X, cos, sin)
    k = 'B2.attn'; A = torch.linalg.lstsq(St[k], Ss[k]).solution.t(); r = (rms(St[k]) / rms(St[k] @ A.t())).median().item()
    W1t = t.blocks[1].mlp.fc1.weight.double(); W1s = s.blocks[1].mlp.fc1.weight.double() @ A * r
    hn = St[k] / rms(St[k]); C = zcorr((hn @ W1t.t()).float(), (hn @ W1s.t()).float())
    live = (W1t.norm(dim=1) > 0.01 * W1t.norm(dim=1).median()).nonzero().flatten().tolist()
    d = torch.stack([torch.sign(C[u, C[u].abs().argmax()]).double() * W1s[C[u].abs().argmax()] - W1t[u] for u in live])
    rel = d.norm(dim=1) / W1t[live].norm(dim=1)
    return d.abs().max().item(), d.abs().mean().item(), rel.median().item()


for run in (sys.argv[1:] or ['wide2', 'wide4']):
    files = sorted(glob.glob(f"{HERE}/10K_{run}_it*.pt"))
    print(f"\n{run}:  it | best fit | B2 fc1 best member: max abs / mean abs / median row rel | mean abs, median over 8 members")
    for f in files:
        ck = torch.load(f, map_location=dev, weights_only=False); cfg = ck['cfg']
        X = torch.randn(256, 32, cfg['d_model'], generator=torch.Generator(device=dev).manual_seed(5), device=dev) * 0.5
        cos, sin = rope_cos_sin(32, 16, cfg['theta'], dev)
        t = canon_(build(cfg, ck['teacher_sd'], dev), True); St, _ = stages(t, X, cos, sin)
        errs = [b2_err(ck, t, St, X, cos, sin, i) for i in range(len(ck['fit']))]
        bi = min(range(len(ck['fit'])), key=lambda i: ck['fit'][i])
        mx, mn, rl = errs[bi]
        allmean = torch.tensor([e[1] for e in errs]).median().item()
        print(f"   {ck['iter']:3d} | {ck['fit'][bi]:.3f} | {mx:.2e} / {mn:.2e} / {rl:.3f} | {allmean:.2e}")
