"""Conv kink solver test: random LeNet-shaped ConvNet (fp64, leaky_relu), EXACT prefix,
frontier filters perturbed by 1e-2 relative noise, garbage downstream.
usage: python peel/test_conv_kink.py 0 1 2 3   (frontiers: 0-2 conv, 3 = fc84)"""
import torch, sys, time, math
sys.path.insert(0, '.')
from nets import ConvNet
import kink_solve
dev = 'cuda'
torch.manual_seed(0)
conv_cfgs = [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)]
T = ConvNet((1, 28, 28), conv_cfgs, fc_dims=(84,), out_dim=10, act='leaky_relu').double()
with torch.no_grad():
    for L in T.layers:
        L.bias.normal_(0, 0.05)
T = T.to(dev).eval()
for fr in [int(a) for a in sys.argv[1:]] or [0]:
    G = T.clone().double().eval(); gen = torch.Generator(device=dev).manual_seed(1 + fr)
    with torch.no_grad():
        Wt = T.layers[fr].weight; bt = T.layers[fr].bias
        rn = Wt.reshape(Wt.shape[0], -1).norm(dim=1).view(-1, *([1] * (Wt.dim() - 1)))
        G.layers[fr].weight.copy_(Wt + 1e-2 * rn * torch.randn(Wt.shape, device=dev, dtype=Wt.dtype, generator=gen) / math.sqrt(Wt[0].numel()))
        G.layers[fr].bias.copy_(bt + 5e-3 * rn.flatten() * torch.randn(bt.shape, device=dev, dtype=bt.dtype, generator=gen))
        for k in range(fr + 1, len(G.layers)):
            G.layers[k].weight.add_(0.3 * torch.randn_like(G.layers[k].weight)); G.layers[k].bias.add_(0.1 * torch.randn_like(G.layers[k].bias))
    t0 = time.time()
    W, b, mask, nq = kink_solve.recover_layer(T, G, fr, dev, verbose=2, sampling="track")
    Wf = W.reshape(W.shape[0], -1).double(); Wtf = Wt.reshape(Wt.shape[0], -1)
    errs = []
    for c in range(W.shape[0]):
        v = torch.cat([Wf[c], b[c].double().reshape(1)]); v = v / v.norm()
        g = torch.cat([Wtf[c], bt[c].reshape(1)]); g = g / g.norm()
        errs.append((v - g).abs().max().item())
    e = torch.tensor(errs)
    print(f"FRONTIER {fr} ({'conv' if fr < T.n_conv else 'fc'}, Din={Wtf.shape[1]}): {int(mask.sum())}/{len(e)} refined  max err {e.max():.1e} median {e.median():.1e}  {nq} q  {time.time()-t0:.1f}s", flush=True)
