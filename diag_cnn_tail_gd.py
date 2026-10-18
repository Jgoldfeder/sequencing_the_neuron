"""LeNet tail oracle: with conv1+conv2 EXACT and frozen (the state every peel run reaches), can a
single student trained on fresh random queries drive C5 (120 ch) / F6 (84) into the refiner's
basin (<1e-2 per-channel error)? And does the SCALE of the exact prefix matter -- the Sep-30
snapshot had the pinned prefix at pre-activation RMS 0.06 (guess magnitude) and C5 rows inflated
to norm 18.6 (teacher 1.85). Arms (tail init identical, only prefix gauge / optimizer differ):

  rms1        : prefix canonicalised to pre-act RMS 1 per layer, Adam 1e-3 on the tail
  teacher     : prefix at the teacher's own scale (RMS ~1.06)
  tiny        : prefix at RMS 0.06 (what the Sep-30 run trained under)
  big         : prefix at RMS 10
  rownorm     : rms1 + per-step projection of C5/F6 rows to fixed norms (canonicalised at step 500)
  dirlr       : rownorm + per-layer lr = eta*norm/sqrt(fan_in), eta 0.03*0.85^(step//1000)

Metrics every --log steps: cube MAE; per tail layer aligned per-channel max error (median,
#<1e-2, #<1e-3), sign flips, duplicates, dead/always-on, median row norm. Teacher used only
for the exact prefix (as a peel run would have it) and for scoring.

Usage: python diag_cnn_tail_gd.py --arms rms1,teacher,tiny,rownorm,dirlr --steps 30000 --device cuda:1
"""
import argparse, json, math, time
import torch
from nets import ConvNet
from data import make_teacher_cnn
from align import cnn_canonicalize_, cnn_align_to_
import cnn_balance as cb

CFGS = [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)]
FC = (84,)
PREFIX = 2          # layers 0,1 exact; frontier = layer 2 (C5)


def rows(layer):
    W = layer.weight; k = W.shape[0]
    return torch.cat([W.reshape(k, -1), layer.bias.view(-1, 1)], 1)


@torch.no_grad()
def set_prefix_rms(net, x, target):
    """gauge the exact prefix so each prefix layer has pre-activation RMS == target (tail is
    re-initialised afterwards, so no compensation is wanted)."""
    for l in range(PREFIX):
        z = cb.hidden_preacts(net, x)[l]
        c = target / float(z.pow(2).mean().sqrt())
        net.layers[l].weight.mul_(c); net.layers[l].bias.mul_(c)


@torch.no_grad()
def reinit_tail(net, seed):
    g = torch.Generator(device=net.layers[0].weight.device).manual_seed(seed)
    for l in net.layers[PREFIX:]:
        fan_in = l.weight.shape[1] if l.weight.dim() == 2 else l.weight[0].numel()
        std = math.sqrt(2.0 / (fan_in + l.weight.shape[0]))
        l.weight.copy_(torch.randn(l.weight.shape, generator=g, device=l.weight.device, dtype=l.weight.dtype) * std)
        l.bias.zero_()


@torch.no_grad()
def score(student, teacher, x):
    t = teacher.clone().double(); r = student.clone().double()
    cnn_canonicalize_(t); cnn_canonicalize_(r); cnn_align_to_(r, t)
    out = []
    zs = cb.hidden_preacts(student, x)
    for l in range(PREFIX, len(t.layers) - 1):
        A, B = rows(t.layers[l]), rows(r.layers[l])
        e = (A - B).abs().max(1).values
        cos = torch.nn.functional.cosine_similarity(A, B, dim=1)
        An = A / A.norm(dim=1, keepdim=True); Bn = B / B.norm(dim=1, keepdim=True)
        best = (Bn @ An.T).abs().argmax(1); dup = int(len(best) - len(torch.unique(best)))
        zc = zs[l].transpose(0, 1).reshape(zs[l].shape[1], -1); pos = (zc > 0).float().mean(1)
        out.append(dict(med=float(e.median()), n_1e2=int((e < 1e-2).sum()), n_1e3=int((e < 1e-3).sum()),
                        flips=int((cos < 0).sum()), dups=dup, dead=int((pos < 1e-3).sum()),
                        always_on=int((pos > 1 - 1e-3).sum()),
                        row_norm=float(cb.row_norms(student.layers[l]).median())))
    return out


def sampler(kind, dev):
    """query distribution: cube U[-1,1]^784 | gauss N(0,0.5^2) (the pipeline's qg init) | mnist (normalised
    training images, the teacher's own data) | mix (half gauss, half mnist)."""
    if kind in ("mnist", "mix"):
        from data import load_mnist
        (xtr, _), _ = load_mnist(dev)
    def draw(n):
        if kind == "cube":
            return torch.rand(n, 784, device=dev) * 2 - 1
        if kind == "gauss":
            return torch.randn(n, 784, device=dev) * 0.5
        if kind == "mnist":
            return xtr[torch.randint(0, len(xtr), (n,), device=dev)]
        h = n // 2
        return torch.cat([torch.randn(h, 784, device=dev) * 0.5, xtr[torch.randint(0, len(xtr), (n - h,), device=dev)]])
    return draw


def run_arm(arm, teacher, dev, steps, log, batch, seed, xeval, draw=None, init_near=0.0):
    draw = draw or (lambda n: torch.rand(n, 784, device=dev) * 2 - 1)
    net = teacher.clone().float()
    x0 = draw(2048)
    rms = {"rms1": 1.0, "rownorm": 1.0, "dirlr": 1.0, "tiny": 0.06, "big": 10.0}.get(arm)
    if rms is not None:
        set_prefix_rms(net, x0, rms)
    if init_near > 0:
        # basin test: tail = teacher tail + relative Gaussian noise of this size (is the truth an attractor?)
        g = torch.Generator(device=dev).manual_seed(seed)
        with torch.no_grad():
            for l in net.layers[PREFIX:]:
                l.weight.add_(torch.randn(l.weight.shape, generator=g, device=dev) * init_near * l.weight.abs().mean())
                l.bias.add_(torch.randn(l.bias.shape, generator=g, device=dev) * init_near * (l.bias.abs().mean() + 1e-3))
    else:
        reinit_tail(net, seed)
    for l in range(PREFIX):
        for p in net.layers[l].parameters():
            p.requires_grad_(False)
    tail = list(net.layers[PREFIX:])
    if arm == "dirlr":
        opt = torch.optim.Adam([{"params": list(l.parameters()), "lr": 1e-3} for l in tail], lr=1e-3)
    else:
        opt = torch.optim.Adam([p for l in tail for p in l.parameters()], lr=1e-3)
    targets = None
    rec = []; t0 = time.time()
    with torch.no_grad():
        z = cb.hidden_preacts(net, x0)
        print(f"    [{arm}] prefix pre-act RMS {float(z[0].pow(2).mean().sqrt()):.3f} {float(z[1].pow(2).mean().sqrt()):.3f}; "
              f"C5 input (pooled act) RMS {float(torch.nn.functional.avg_pool2d(net.act(z[1]), 2).pow(2).mean().sqrt()):.3f}")
    for s in range(1, steps + 1):
        x = draw(batch)
        with torch.no_grad():
            y = teacher(x)
        opt.zero_grad()
        loss = (net(x) - y).abs().mean()
        loss.backward()
        if arm == "dirlr" and targets is not None:
            eta = 0.03 * (0.85 ** (s // 1000))
            for l, g in zip(tail[:-1], opt.param_groups[:-1]):
                g["lr"] = eta * float(cb.row_norms(l).median()) / math.sqrt(cb.fan_in(l))
        opt.step()
        if targets is not None:
            cb.project_rows_(net, targets)
        if arm in ("rownorm", "dirlr") and s == 500:
            full = cb.canonicalize_(net, x, 1.0, 0.0, opt, layers=list(range(PREFIX, len(net.layers) - 1)))
            targets = full
            print(f"    [{arm}] tail canonicalised at step 500: norms {[round(float(v), 2) for v in full if v is not None]}")
        if s % log == 0 or s == steps:
            with torch.no_grad():
                mae = float((net(xeval) - teacher(xeval)).abs().mean())
            layers = score(net, teacher, xeval)
            rec.append(dict(step=s, mae=mae, layers=layers, wall=time.time() - t0,
                            head_norm=float(cb.row_norms(net.layers[-1]).median())))
            print(f"  [{arm}] step {s:6d} mae {mae:.4f} | " + " ".join(
                f"{nm}:{d['med']:.1e}/{d['n_1e2']}/{d['n_1e3']}/f{d['flips']}/dup{d['dups']}/d{d['dead']}/a{d['always_on']}/n{d['row_norm']:.2f}"
                for nm, d in zip(("C5", "F6"), layers)) + f" | head n{rec[-1]['head_norm']:.2f}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="rms1,teacher,tiny,big,rownorm,dirlr")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--log", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--queries", default="cube", choices=["cube", "gauss", "mnist", "mix"])
    ap.add_argument("--init-near", type=float, default=0.0,
                    help="basin test: start the tail at teacher + this relative noise instead of random init")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = a.device
    teacher = make_teacher_cnn((1, 28, 28), CFGS, FC, 10, epochs=25, seed=0, device=dev, act="leaky_relu").float()
    for p in teacher.parameters():
        p.requires_grad_(False)
    torch.manual_seed(1234)
    draw = sampler(a.queries, dev)
    xeval = draw(4096)
    with torch.no_grad():
        st = cb.layer_stats(teacher, xeval)
        print(f"teacher on {a.queries} queries: RMS", [round(r["rms"], 2) for r in st[:-1]], "row norm",
              [round(r["row_norm"], 2) for r in st], "dead", [r["dead"] for r in st[:-1]], "always-on",
              [r["always_on"] for r in st[:-1]])
    res = {"arms": {}, "queries": a.queries, "init_near": a.init_near}
    for arm in a.arms.split(","):
        print(f"=== arm {arm} (queries={a.queries}, init_near={a.init_near})")
        res["arms"][arm] = run_arm(arm, teacher, dev, a.steps, a.log, a.batch, a.seed, xeval, draw, a.init_near)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
