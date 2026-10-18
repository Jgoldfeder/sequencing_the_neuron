"""Deep leaky-ReLU MLP, peel-stage oracle: the first K layers are EXACT and frozen (what a peel run
has after pinning them); a single student trains the remaining layers on fresh random cube
queries. Measures how fast layer K+1 enters the refiner's basin (<1e-2 per-row error) as a
function of K, of the prefix's gauge scale, and of the optimizer -- i.e. whether peeling
actually makes the next layer learnable, how the time-to-basin grows with depth, and whether
renormalising the pinned prefix (Judah's proposal) or dirlr changes it.

Arms (tail init identical): plain | tiny (prefix pre-act RMS 0.06) | big (RMS 10) | rownorm | dirlr
Usage: python diag_mlp_tail_gd.py --dims 3072,200,200,200,200,200,100 --prefix 2 --arms plain,tiny,rownorm,dirlr
"""
import argparse, json, math, time
import torch
from nets import MLP
from data import make_teacher
from align import align_clone_to
from diag_deep_relu_gd import hidden_preacts, row_norms, scale_rows_, project_, score


@torch.no_grad()
def set_prefix_rms(net, x, K, target):
    for l in range(K):
        z = hidden_preacts(net, x)[l]
        c = target / float(z.pow(2).mean().sqrt())
        net.layers[l].weight.mul_(c); net.layers[l].bias.mul_(c)


@torch.no_grad()
def reinit_tail(net, K, seed):
    g = torch.Generator(device=net.layers[0].weight.device).manual_seed(seed)
    for l in net.layers[K:]:
        std = math.sqrt(2.0 / (l.weight.shape[1] + l.weight.shape[0]))
        l.weight.copy_(torch.randn(l.weight.shape, generator=g, device=l.weight.device) * std)
        l.bias.zero_()


@torch.no_grad()
def canonicalize_tail_(net, x, K, opt):
    tg = []
    for l in range(K, len(net.layers) - 1):
        z = hidden_preacts(net, x)[l]
        c = 1.0 / float(z.pow(2).mean().sqrt().clamp_min(1e-12))
        scale_rows_(net, l, torch.full((z.shape[1],), c, device=x.device), opt)
        n = row_norms(net.layers[l]).clamp_min(1e-12)
        scale_rows_(net, l, n.median() / n, opt)
        tg.append(row_norms(net.layers[l]).median().item())
    return tg


@torch.no_grad()
def project_tail_(net, K, tg):
    for l, v in zip(range(K, len(net.layers) - 1), tg):
        lay = net.layers[l]
        f = v / row_norms(lay).clamp_min(1e-12)
        lay.weight.mul_(f.view(-1, 1)); lay.bias.mul_(f)


def run_arm(arm, dims, K, teacher, dev, steps, log, batch, seed, xeval):
    net = teacher.clone()
    x0 = torch.rand(2048, dims[0], device=dev) * 2 - 1
    rms = {"plain": 1.0, "rownorm": 1.0, "dirlr": 1.0, "tiny": 0.06, "big": 10.0, "teacher": None,
           "rms0.3": 0.3, "rms3": 3.0, "rms30": 30.0, "frontlr10": 1.0, "alllr10": 1.0, "frontlr3": 1.0,
           "frontlr10_restlow": 1.0}.get(arm, 1.0)
    if rms is not None:
        set_prefix_rms(net, x0, K, rms)
    reinit_tail(net, K, seed)
    for l in range(K):
        for p in net.layers[l].parameters():
            p.requires_grad_(False)
    tail = list(net.layers[K:])
    if arm == "dirlr":
        opt = torch.optim.Adam([{"params": list(l.parameters()), "lr": 1e-3} for l in tail], lr=1e-3)
    elif arm.startswith("frontlr") or arm == "alllr10":
        # is the prefix-scale effect just "the frontier layer wants a bigger lr"? frontier lr x10 (x3), rest 1e-3
        # (frontlr10_restlow: rest 1e-4); alllr10: every tail layer at 1e-2
        fl = {"frontlr10": 1e-2, "frontlr3": 3e-3, "alllr10": 1e-2, "frontlr10_restlow": 1e-2}[arm]
        rl = {"frontlr10": 1e-3, "frontlr3": 1e-3, "alllr10": 1e-2, "frontlr10_restlow": 1e-4}[arm]
        opt = torch.optim.Adam([{"params": list(l.parameters()), "lr": fl if i == 0 else rl} for i, l in enumerate(tail)], lr=rl)
    else:
        opt = torch.optim.Adam([p for l in tail for p in l.parameters()], lr=1e-3)
    tg = None; rec = []; t0 = time.time()
    with torch.no_grad():
        z = hidden_preacts(net, x0)
        print(f"    [{arm}] prefix pre-act RMS {[round(float(v.pow(2).mean().sqrt()), 3) for v in z[:K]]}")
    for s in range(1, steps + 1):
        x = torch.rand(batch, dims[0], device=dev) * 2 - 1
        with torch.no_grad():
            y = teacher(x)
        opt.zero_grad()
        loss = (net(x) - y).abs().mean()
        loss.backward()
        if arm == "dirlr" and tg is not None:
            eta = 0.03 * (0.85 ** (s // 1000))
            for l, g in zip(tail[:-1], opt.param_groups[:-1]):
                g["lr"] = eta * float(row_norms(l).median()) / math.sqrt(l.weight.shape[1])
        opt.step()
        if tg is not None:
            project_tail_(net, K, tg)
        if arm in ("rownorm", "dirlr") and s == 500:
            tg = canonicalize_tail_(net, x, K, opt)
            print(f"    [{arm}] tail canonicalised at step 500: norms {[round(v, 2) for v in tg]}")
        if s % log == 0 or s == steps:
            with torch.no_grad():
                mae = float((net(xeval) - teacher(xeval)).abs().mean())
            layers, head = score(net, teacher, xeval)
            rec.append(dict(step=s, mae=mae, layers=layers, **head, wall=time.time() - t0))
            print(f"  [{arm}] step {s:6d} mae {mae:.4f} | " + " ".join(
                f"L{l + 1}:{d['med']:.1e}/{d['n_1e2']}/{d['n_1e3']}/f{d['flips']}/d{d['dead']}/n{d['row_norm']:.1f}"
                for l, d in enumerate(layers) if l >= K) + f" | head {head['head_med']:.1e} n{head['head_norm']:.1f}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="3072,200,200,200,200,200,100")
    ap.add_argument("--prefix", type=int, default=2)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--log", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--arms", default="plain,tiny,rownorm,dirlr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dims = [int(v) for v in a.dims.split(",")]
    dev = a.device
    teacher = make_teacher(dims, 25, 0, device=dev, verbose=False).to(dev)
    for p in teacher.parameters():
        p.requires_grad_(False)
    torch.manual_seed(1234)
    xeval = torch.rand(4096, dims[0], device=dev) * 2 - 1
    res = {"dims": dims, "prefix": a.prefix, "arms": {}}
    for arm in a.arms.split(","):
        print(f"=== prefix {a.prefix} arm {arm}")
        res["arms"][arm] = run_arm(arm, dims, a.prefix, teacher, dev, a.steps, a.log, a.batch, a.seed, xeval)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
