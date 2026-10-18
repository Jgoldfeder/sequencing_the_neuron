"""Does DISAGREEMENT MINING (the committee's sampler) supply the next-layer signal that random
queries lack? Peel-stage oracle with an exact frozen prefix and P students (same prefix, different
tail inits). Each step: half the batch is random (gauss), the other half starts random and is
pushed by `--mine-steps` Adam steps on the inputs to maximise the mean pairwise |f_i(x) - f_j(x)|
(students only -- attacker-legal), then every student trains on the black box's labels.
Compare with the plain single-student arms of diag_cnn_tail_gd.py / diag_mlp_tail_gd.py.

Usage: python diag_tail_disagree.py --model lenet --students 2 --steps 30000
       python diag_tail_disagree.py --model mlp --dims 3072,200,200,200,200,200,100 --prefix 2 --students 2
"""
import argparse, json, time
import torch


def mine(students, x0, steps, lr):
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        fs = [s(x) for s in students]
        d = sum((fs[i] - fs[j]).abs().mean() for i in range(len(fs)) for j in range(i + 1, len(fs)))
        (-d).backward()
        opt.step()
    return x.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lenet", "mlp"], default="lenet")
    ap.add_argument("--dims", default="3072,200,200,200,200,200,100")
    ap.add_argument("--prefix", type=int, default=2)
    ap.add_argument("--students", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--log", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--mine-steps", type=int, default=20)
    ap.add_argument("--mine-lr", type=float, default=0.05)
    ap.add_argument("--mine-frac", type=float, default=0.5)
    ap.add_argument("--prefix-rms", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(1234)
    if a.model == "lenet":
        import diag_cnn_tail_gd as T
        from data import make_teacher_cnn
        teacher = make_teacher_cnn((1, 28, 28), T.CFGS, T.FC, 10, epochs=25, seed=0, device=dev, act="leaky_relu").float()
        draw = T.sampler("gauss", dev)
        K = T.PREFIX
        def build(seed):
            net = teacher.clone().float()
            T.set_prefix_rms(net, draw(2048), a.prefix_rms)
            T.reinit_tail(net, seed)
            return net
        names = ("C5", "F6")
        score = lambda net, x: T.score(net, teacher, x)
    else:
        import diag_mlp_tail_gd as T
        from data import make_teacher
        dims = [int(v) for v in a.dims.split(",")]
        teacher = make_teacher(dims, 25, 0, device=dev, verbose=False).to(dev)
        draw = lambda n: torch.rand(n, dims[0], device=dev) * 2 - 1
        K = a.prefix
        def build(seed):
            net = teacher.clone()
            T.set_prefix_rms(net, draw(2048), K, a.prefix_rms)
            T.reinit_tail(net, K, seed)
            return net
        names = tuple(f"L{l + 1}" for l in range(K, len(dims) - 2))
        score = lambda net, x: T.score(net, teacher, x)[0][K:]
    for p in teacher.parameters():
        p.requires_grad_(False)
    nets = [build(s) for s in range(a.students)]
    for net in nets:
        for l in range(K):
            for p in net.layers[l].parameters():
                p.requires_grad_(False)
    opts = [torch.optim.Adam([p for p in n.parameters() if p.requires_grad], lr=1e-3) for n in nets]
    xeval = draw(4096)
    nm = int(a.batch * a.mine_frac) if a.students > 1 else 0
    rec = []; t0 = time.time()
    print(f"=== {a.model} prefix {K} students {a.students} mine {nm}/{a.batch} x {a.mine_steps} steps lr {a.mine_lr}")
    for s in range(1, a.steps + 1):
        xr = draw(a.batch - nm)
        x = torch.cat([xr, mine(nets, draw(nm), a.mine_steps, a.mine_lr)]) if nm else xr
        with torch.no_grad():
            y = teacher(x)
        for net, opt in zip(nets, opts):
            opt.zero_grad()
            (net(x) - y).abs().mean().backward()
            opt.step()
        if s % a.log == 0 or s == a.steps:
            # how often does a mined query cross a TEACHER frontier kink? (diagnostic only)
            xm = mine(nets, draw(256), a.mine_steps, a.mine_lr) if nm else draw(256)
            with torch.no_grad():
                mae = [float((n(xeval) - teacher(xeval)).abs().mean()) for n in nets]
                import cnn_balance as cb
                z = cb.hidden_preacts(teacher, xm)[K]
                zc = z.transpose(0, 1).reshape(z.shape[1], -1); pos = (zc > 0).float().mean(1)
                bal = torch.minimum(pos, 1 - pos)
                zr = cb.hidden_preacts(teacher, draw(256))[K]
                zrc = zr.transpose(0, 1).reshape(zr.shape[1], -1); posr = (zrc > 0).float().mean(1)
                balr = torch.minimum(posr, 1 - posr)
            sc = [score(n, xeval) for n in nets]
            rec.append(dict(step=s, mae=mae, scores=sc, wall=time.time() - t0,
                            frontier_onesided_mined=int((bal < 0.01).sum()), frontier_onesided_random=int((balr < 0.01).sum())))
            print(f"  step {s:6d} mae {' '.join(f'{m:.4f}' for m in mae)} | " + " || ".join(
                f"s{i} " + " ".join(f"{nm_}:{d['med']:.1e}/{d['n_1e2']}/{d['n_1e3']}/f{d['flips']}" for nm_, d in zip(names, sci))
                for i, sci in enumerate(sc))
                + f" | teacher-frontier one-sided ch: mined {int((bal < 0.01).sum())} random {int((balr < 0.01).sum())}", flush=True)
            if a.out:
                json.dump(dict(args=vars(a), rec=rec), open(a.out, "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
