import sys, glob, torch
sys.path.insert(0, '.'); sys.path.insert(0, '..')
from track_b2 import b2_err
from layer_report import stages, build, dev, canon_, rope_cos_sin, report
torch.set_grad_enabled(False)
for f in sys.argv[1:]:
    ck = torch.load(f, map_location=dev, weights_only=False); cfg = ck['cfg']
    X = torch.randn(32, 256, 16, generator=torch.Generator(device=dev).manual_seed(5), device=dev) * 0.5; cos, sin = rope_cos_sin(256, 16, cfg['theta'], dev)
    t = canon_(build(cfg, ck['teacher_sd'], dev), True); St, _ = stages(t, X, cos, sin)
    bi = min(range(8), key=lambda i: ck['fit'][i]); e = b2_err(ck, t, St, X, cos, sin, bi); rep = report(ck, bi, X)
    print(f"it {ck['iter']:2d} best m{bi} (fit {ck['fit'][bi]:.3f}): B2 MLP inputs max {e[0]:.2f} mean {e[1]:.1e} median row {e[2]:.1%}", flush=True)
    for k in ('B2.attn', 'B2.mlp', 'unembed'):
        print(f"      {k:8s} func err {rep[k]['func']:.3f} | {rep[k]['w']}", flush=True)
