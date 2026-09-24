"""Diagnostic only: true weights measure point error; never used for recovery.
Run from code/: CUDA_VISIBLE_DEVICES=0 python peel/diagnose_kink_conditioning.py
"""
import math
import sys
import torch
sys.path.insert(0, '.')
from nets import MLP
from align import scale_normalize_
from verify_layer1 import _Oracle
import kink_solve as K

torch.set_num_threads(1)  # Small CPU SVDs otherwise oversubscribe cores.
torch.manual_seed(0)
T = MLP([200]*8+[100]).double()
with torch.no_grad():
    for layer in T.layers:
        layer.bias.normal_(0, .1)
scale_normalize_(T)
T = T.cuda().eval()

@torch.no_grad()
def report(Hs, l, guess, label):
    rows = []
    for c, H in Hs.items():
        wt = T.layers[l].weight[c].cpu(); bt = T.layers[l].bias[c].cpu()
        w, b, kept, gap = K.solve_from_points(H, guess.layers[l].weight[c])
        h = H.cpu()
        residual = (h @ wt + bt).abs()
        a = torch.cat([h, torch.ones(len(h), 1, dtype=h.dtype)], 1)
        s = torch.linalg.svdvals(a)
        # Remove localization error in feature space to test the numerical fit alone.
        hp = h - ((h @ wt + bt) / wt.square().sum())[:, None] * wt
        wp, bp, _, _ = K.solve_from_points(hp, wt)
        rows.append([max((w.cpu()-wt).abs().max().item(), abs(b.cpu()-bt).item()),
                     residual.median().item(), residual.max().item(),
                     (s[0]/s[-2]).item(), s[-2].item(), gap,
                     max((wp-wt).abs().max().item(), abs(bp-bt).item())])
    if not rows:
        print(label, "no channels reached the required point count", flush=True)
        return
    v = torch.tensor(rows, dtype=torch.float64)
    print(label, 'channels', len(rows), 'columns: weight_err, true_res_med, true_res_max, identifiable_cond, sigma_next, gap, projected_fit_err', flush=True)
    print('median', ['%.3e'%x for x in v.median(0).values], flush=True)
    print('max   ', ['%.3e'%x for x in v.max(0).values], flush=True)
    for c, row in zip(Hs, rows):
        if row[0] > 1e-8:
            print('bad', c, ['%.3e'%x for x in row], flush=True)

for l in [1, 3, 5, 6]:
    G = T.clone(); gen = torch.Generator(device='cuda').manual_seed(1)
    with torch.no_grad():
        G.layers[l].weight.add_(.01*torch.randn((200,200), device='cuda', dtype=torch.float64, generator=gen)/math.sqrt(200))
        G.layers[l].bias.add_(.005*torch.randn(200, device='cuda', dtype=torch.float64, generator=gen))
    orc = _Oracle(T)
    anchors = K.polish_layer(orc, G, l, range(12), gen, full=True, eps0=.02,
                             need=48, max_rounds=3, return_points=True)
    for step in [.5, 5.]:
        hs = K.track_layer(orc, G, l, anchors, gen, step=step)
        report(hs, l, G, f'layer={l} track step={step}')
    independent = K.polish_layer(orc, G, l, range(12), gen, full=True, eps0=.02,
                                 need=280, max_rounds=20)
    report({c:v[2] for c,v in independent.items()}, l, G, f'layer={l} independent')
