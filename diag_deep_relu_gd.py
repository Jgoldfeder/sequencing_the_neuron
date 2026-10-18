"""Oracle isolation for the deep-ReLU stall: can ONE student, trained on fresh random queries
(infinite data, no committee, no sampler), drive every layer of a deep leaky-ReLU teacher into
the kink refiner's basin (~1e-2 row error)? If not, the committee cannot either, and the
optimizer -- not the sampling -- is the bottleneck. Arms differ only in the optimizer / gauge:

  plain   : Adam 1e-3 (what the committee runs)
  lowlr   : Adam 2e-4
  adamw   : AdamW 1e-3, weight decay 1e-2 (restoring force on the gauge-invariant scale)
  rownorm : Adam 1e-3 + one-time canonicalisation (pre-act RMS 1, equal row norms) at step 500,
            then every hidden (row,bias) projected back to its fixed norm after every step
  dirlr   : rownorm + per-layer lr = eta * norm / sqrt(fan_in), eta = 0.03 * 0.85^(step//1000)
  sgd     : SGD momentum 0.9, lr 5e-3

Metrics every --log steps: cube MAE, per hidden layer the aligned per-row max error
(median, # rows < 1e-2, # rows < 1e-3), sign flips, dead / always-on census, median row norm.
Teacher-side knowledge is used ONLY for scoring.

Usage: python diag_deep_relu_gd.py --dims 3072,200,200,200,200,200,100 --steps 60000 \
         --arms plain,rownorm,dirlr,adamw --device cuda:1 --out results/diag_deep_relu_gd.json
"""
import argparse, json, math, time
import torch
from nets import MLP
from data import make_teacher
from align import align_clone_to


def hidden_preacts(net, x):
    zs = []
    for layer in net.layers[:-1]:
        z = layer(x); zs.append(z); x = net.act(z)
    return zs


def row_norms(layer):
    return torch.sqrt(layer.weight.pow(2).sum(1) + layer.bias.pow(2))


@torch.no_grad()
def scale_rows_(net, l, f, opt=None):
    lay, nxt = net.layers[l], net.layers[l + 1]
    lay.weight.mul_(f.view(-1, 1)); lay.bias.mul_(f); nxt.weight.div_(f.view(1, -1))
    if opt is not None:
        for p, g in ((lay.weight, 1 / f.view(-1, 1)), (lay.bias, 1 / f), (nxt.weight, f.view(1, -1))):
            st = opt.state.get(p)
            if st and "exp_avg" in st:
                st["exp_avg"].mul_(g); st["exp_avg_sq"].mul_(g * g)


@torch.no_grad()
def canonicalize_(net, x, opt=None, target=1.0):
    for l in range(len(net.layers) - 1):
        z = hidden_preacts(net, x)[l]
        c = target / float(z.pow(2).mean().sqrt().clamp_min(1e-12))
        scale_rows_(net, l, torch.full((z.shape[1],), c, device=x.device), opt)
        n = row_norms(net.layers[l]).clamp_min(1e-12)
        scale_rows_(net, l, n.median() / n, opt)
    return [row_norms(net.layers[l]).median().item() for l in range(len(net.layers) - 1)]


@torch.no_grad()
def project_(net, targets):
    for l, tg in enumerate(targets):
        lay = net.layers[l]
        f = tg / row_norms(lay).clamp_min(1e-12)
        lay.weight.mul_(f.view(-1, 1)); lay.bias.mul_(f)


@torch.no_grad()
def score(student, teacher, x):
    t, r = align_clone_to(student, teacher)
    out = []
    zs_s = hidden_preacts(student, x)
    for l in range(len(t.layers) - 1):
        dW = (t.layers[l].weight - r.layers[l].weight).abs().max(1).values
        db = (t.layers[l].bias - r.layers[l].bias).abs()
        e = torch.maximum(dW, db)
        cos = torch.nn.functional.cosine_similarity(
            torch.cat([t.layers[l].weight, t.layers[l].bias[:, None]], 1),
            torch.cat([r.layers[l].weight, r.layers[l].bias[:, None]], 1), dim=1)
        pos = (zs_s[l] > 0).float().mean(0)
        out.append(dict(med=float(e.median()), n_1e2=int((e < 1e-2).sum()), n_1e3=int((e < 1e-3).sum()),
                        flips=int((cos < 0).sum()), dead=int((pos < 1e-3).sum()),
                        always_on=int((pos > 1 - 1e-3).sum()),
                        row_norm=float(row_norms(student.layers[l]).median())))
    dH = (t.layers[-1].weight - r.layers[-1].weight).abs()
    return out, dict(head_med=float(dH.max(1).values.median()),
                     head_norm=float(row_norms(student.layers[-1]).median()))


def run_arm(arm, dims, teacher, dev, steps, log, batch, seed, xeval):
    torch.manual_seed(seed)
    net = MLP(dims).to(dev)
    base_lr = {"plain": 1e-3, "rownorm": 1e-3, "dirlr": 1e-3, "adamw": 1e-3, "lowlr": 2e-4, "sgd": 5e-3}[arm]
    if arm == "adamw":
        opt = torch.optim.AdamW(net.parameters(), lr=base_lr, weight_decay=1e-2)
    elif arm == "sgd":
        opt = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=0.9)
    elif arm == "dirlr":
        opt = torch.optim.Adam([{"params": list(l.parameters()), "lr": base_lr} for l in net.layers], lr=base_lr)
    else:
        opt = torch.optim.Adam(net.parameters(), lr=base_lr)
    targets = None
    rec = []
    t0 = time.time()
    for s in range(1, steps + 1):
        x = torch.rand(batch, dims[0], device=dev) * 2 - 1
        with torch.no_grad():
            y = teacher(x)
        opt.zero_grad()
        loss = (net(x) - y).abs().mean()
        loss.backward()
        if arm == "dirlr" and targets is not None:
            eta = 0.03 * (0.85 ** (s // 1000))
            for l, g in enumerate(opt.param_groups[:-1]):
                g["lr"] = eta * targets[l] / math.sqrt(dims[l])
        opt.step()
        if targets is not None:
            project_(net, targets)
        if arm in ("rownorm", "dirlr") and s == 500:
            targets = canonicalize_(net, x, opt)
            print(f"    [{arm}] canonicalised at step 500: row norms {['%.2f' % v for v in targets]}")
        if s % log == 0 or s == steps:
            with torch.no_grad():
                mae = float((net(xeval) - teacher(xeval)).abs().mean())
            layers, head = score(net, teacher, xeval)
            rec.append(dict(step=s, mae=mae, layers=layers, **head, wall=time.time() - t0))
            print(f"  [{arm}] step {s:6d} mae {mae:.4f} | " + " ".join(
                f"L{l + 1}:{d['med']:.1e}/{d['n_1e2']}/{d['n_1e3']}/f{d['flips']}/d{d['dead']}/a{d['always_on']}/n{d['row_norm']:.1f}"
                for l, d in enumerate(layers)) + f" | head {head['head_med']:.1e} n{head['head_norm']:.1f}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="3072,200,200,200,200,200,100")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--log", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--arms", default="plain,rownorm,dirlr,adamw")
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
    with torch.no_grad():
        zs = hidden_preacts(teacher, xeval)
        print("teacher: per layer dead/always-on/median row norm:",
              [(int(((z > 0).float().mean(0) < 1e-3).sum()), int(((z > 0).float().mean(0) > 1 - 1e-3).sum()),
                round(float(row_norms(teacher.layers[l]).median()), 2)) for l, z in enumerate(zs)],
              "head norm", round(float(row_norms(teacher.layers[-1]).median()), 2))
        print("teacher pre-act RMS per layer:", [round(float(z.pow(2).mean().sqrt()), 2) for z in zs])
    res = {"dims": dims, "steps": a.steps, "batch": a.batch, "arms": {}}
    for arm in a.arms.split(","):
        print(f"=== arm {arm}")
        res["arms"][arm] = run_arm(arm, dims, teacher, dev, a.steps, a.log, a.batch, a.seed, xeval)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
