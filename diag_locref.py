"""Validate guided loc_refine: guess@~1e-2 + near-exact prefix -> per-neuron
recovery eps (vs TRUE teacher, unit-[w|b] frame), queries, wall time.

  python diag_locref.py --dims 3072,200,200,200,100 --frontier 2 \
      --prefix-eps 1e-9 --guess-eps 1e-2 --chans 8 --device cuda
"""
import argparse
import time

import torch

import loc_refine
from nets import MLP


def perturb_rows(layer, eps, gen):
    """Add a random direction of relative size eps to each row's [w|b]."""
    W, b = layer.weight.data, layer.bias.data
    wb = torch.cat([W, b.unsqueeze(1)], 1)
    g = torch.randn(wb.shape, generator=gen, device=wb.device, dtype=wb.dtype)
    g = g / g.norm(dim=1, keepdim=True)
    wb = wb + eps * wb.norm(dim=1, keepdim=True) * g
    W.copy_(wb[:, :-1]); b.copy_(wb[:, -1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="3072,200,200,200,100")
    ap.add_argument("--frontier", type=int, default=1)
    ap.add_argument("--prefix-eps", type=float, default=1e-9)
    ap.add_argument("--guess-eps", type=float, default=1e-2)
    ap.add_argument("--chans", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    dims = [int(x) for x in args.dims.split(",")]
    torch.manual_seed(args.seed)
    teacher = MLP(dims).double().to(dev).eval()
    cons = teacher.clone()
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1)
    for i in range(args.frontier):
        perturb_rows(cons.layers[i], args.prefix_eps, gen)
    perturb_rows(cons.layers[args.frontier], args.guess_eps, gen)

    chans = list(range(args.chans))
    g2 = torch.Generator(device=dev).manual_seed(1234 + args.frontier)
    t0 = time.time()
    W_ref, b_ref, mask, nq = loc_refine.recover_layer(
        teacher, cons, args.frontier, dev, only_channels=chans, gen=g2,
        verbose=True)
    wall = time.time() - t0

    Wt = teacher.layers[args.frontier].weight.data
    bt = teacher.layers[args.frontier].bias.data
    eps_list = []
    for c in chans:
        vt = torch.cat([Wt[c], bt[c].reshape(1)]); vt = vt / vt.norm()
        vr = torch.cat([W_ref[c].double(), b_ref[c].double().reshape(1)])
        vr = vr / vr.norm()
        if (vr @ vt) < 0:
            vr = -vr
        e = float((vr - vt).abs().max())
        eps_list.append((c, bool(mask[c]), e))
        print(f"  c={c:3d} refined={bool(mask[c])!s:5s} eps={e:.3e}")
    solved = [e for _, m, e in eps_list if m]
    print(f"\nfrontier={args.frontier} prefix_eps={args.prefix_eps:g} "
          f"guess_eps={args.guess_eps:g}")
    print(f"solved {len(solved)}/{len(chans)} | "
          f"eps mean {sum(solved)/max(len(solved),1):.3e} "
          f"max {max(solved, default=float('nan')):.3e} | "
          f"{nq} queries ({nq // max(len(chans),1)}/neuron) | {wall:.1f}s "
          f"({wall / max(len(chans),1):.2f}s/neuron)")


if __name__ == "__main__":
    main()
