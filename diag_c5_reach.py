"""Are the one-sided C5 / F6 channels of the LeNet teacher identifiable AT ALL on the query box?
A channel whose pre-activation never changes sign on the input domain is affine there: no kink,
no gradient-descent signal for its row direction, nothing for the kink refiner to find.
For every hidden channel, maximise and minimise its pre-activation over x in [-B, B]^784
(white-box projected Adam ascent from several random starts; diagnostic only) and report whether
both signs are reachable, with what margin, and how that compares with the random-query crossing
rate. Repeats for a few box sizes B and for the Gaussian-ball radius the sampler uses.

Usage: python diag_c5_reach.py --device cuda:1 [--boxes 1,2,4] [--starts 8] [--steps 300]
"""
import argparse, json
import torch
from data import make_teacher_cnn
import cnn_balance as cb

CFGS = [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)]
FC = (84,)


def preact(net, x, l):
    """pre-activation of hidden layer l, WITH grad (cb.hidden_preacts is no_grad)."""
    x = x.view(x.shape[0], *net.input_shape)
    for i in range(net.n_conv):
        z = net.layers[i](x)
        if i == l:
            return z.reshape(z.shape[0], z.shape[1], -1)
        x = net.act(z)
        if net.pools[i] > 0:
            x = torch.nn.functional.avg_pool2d(x, net.pools[i])
    x = torch.flatten(x, 1)
    for j in range(net.n_conv, len(net.layers) - 1):
        z = net.layers[j](x)
        if j == l:
            return z.reshape(z.shape[0], z.shape[1], -1)
        x = net.act(z)
    raise IndexError(l)


def extreme(net, l, c, B, starts, steps, dev, sign):
    """max over x in box of sign * z_{l,c}(x) (max over positions). Returns the best value."""
    x = (torch.rand(starts, 784, device=dev) * 2 - 1) * B
    x.requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.05 * B)
    best = -float("inf")
    for _ in range(steps):
        opt.zero_grad()
        z = preact(net, x, l)[:, c, :].max(1).values * sign        # (starts,)
        best = max(best, float(z.max()))
        (-z.sum()).backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(-B, B)
    with torch.no_grad():
        z = preact(net, x, l)[:, c, :].max(1).values * sign
    return max(best, float(z.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boxes", default="1,2,4")
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--layers", default="2,3")
    ap.add_argument("--out", default="results/diag/c5_reach.json")
    a = ap.parse_args()
    dev = a.device
    t = make_teacher_cnn((1, 28, 28), CFGS, FC, 10, epochs=25, seed=0, device=dev, act="leaky_relu").float()
    for p in t.parameters():
        p.requires_grad_(False)
    res = {}
    for l in [int(v) for v in a.layers.split(",")]:
        C = t.layers[l].weight.shape[0]
        # random-query crossing balance on the unit cube and on N(0, 0.5^2)
        with torch.no_grad():
            xc = torch.rand(8192, 784, device=dev) * 2 - 1
            xg = torch.randn(8192, 784, device=dev) * 0.5
            pc = (preact(t, xc, l) > 0).float().mean((0, 2)); pg = (preact(t, xg, l) > 0).float().mean((0, 2))
            balc = torch.minimum(pc, 1 - pc); balg = torch.minimum(pg, 1 - pg)
        res[l] = {"bal_cube": balc.tolist(), "bal_gauss": balg.tolist(), "boxes": {}}
        print(f"=== layer {l + 1} ({C} ch): one-sided (<1% crossing) on cube {int((balc < 0.01).sum())}, gauss {int((balg < 0.01).sum())}; "
              f"never-cross on cube {int((balc == 0).sum())}, gauss {int((balg == 0).sum())}")
        for B in [float(v) for v in a.boxes.split(",")]:
            hi = torch.zeros(C); lo = torch.zeros(C)
            for c in range(C):
                hi[c] = extreme(t, l, c, B, a.starts, a.steps, dev, +1.0)
                lo[c] = -extreme(t, l, c, B, a.starts, a.steps, dev, -1.0)
            both = (hi > 0) & (lo < 0)
            margin = torch.minimum(hi.clamp_min(0), (-lo).clamp_min(0))
            unreach = (~both).nonzero().flatten().tolist()
            print(f"  box [-{B:g},{B:g}]: kink reachable for {int(both.sum())}/{C}; unreachable channels: {unreach[:40]}"
                  f"{'...' if len(unreach) > 40 else ''}")
            os_c = (balc < 0.01).nonzero().flatten()
            if len(os_c):
                print(f"     among the {len(os_c)} cube-one-sided channels: reachable {int(both[os_c.cpu()].sum())}, "
                      f"median crossing margin {float(margin[os_c.cpu()].median()):.3f} (teacher pre-act RMS on cube "
                      f"{float(preact(t, xc, l).pow(2).mean().sqrt()):.2f})")
            res[l]["boxes"][B] = {"hi": hi.tolist(), "lo": lo.tolist(), "reachable": int(both.sum()),
                                  "unreachable": unreach}
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
