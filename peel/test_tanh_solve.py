"""Basin test for tanh_solve.refine: cached smooth teacher + synthetic guess
(every tensor perturbed by relative Frobenius noise `--noise`, default 1e-2)
-> refine with black-box jets only -> alignment eps before/after.

From code/:
  CUDA_VISIBLE_DEVICES=2 python peel/test_tanh_solve.py --arch 784,16,10 --act tanh --epochs 3
  CUDA_VISIBLE_DEVICES=2 python peel/test_tanh_solve.py --arch 784,128,80,40,32,10 --act sigmoid
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from data import make_teacher
from align import param_errors
import tanh_solve as ts

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="784,16,10")
ap.add_argument("--act", default="tanh", choices=["tanh", "sigmoid"])
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--teacher-seed", type=int, default=0)
ap.add_argument("--noise", type=float, default=1e-2, help="relative Frobenius noise per tensor")
ap.add_argument("--points", type=int, default=512)
ap.add_argument("--dirs", type=int, default=0, help="0 = full Jacobian")
ap.add_argument("--iters", type=int, default=40)
ap.add_argument("--cg-iters", type=int, default=200)
ap.add_argument("--scales", default="0.5,1,2")
ap.add_argument("--pool", default="none", choices=["none", "data"],
                help="'data': draw half the jet points from the teacher's test set "
                     "(mirrors run.py, which pools the run's collected queries)")
ap.add_argument("--precond", default="off", choices=["auto", "on", "off"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--output", default=None)
args = ap.parse_args()

dims = [int(x) for x in args.arch.split(",")]
dev = args.device
teacher = make_teacher(dims, epochs=args.epochs, seed=args.teacher_seed,
                       device=dev, verbose=False, act=args.act)
teacher64 = teacher.clone().double().eval()
bb = ts.Oracle(lambda x: teacher64(x))          # the solver sees ONLY this

torch.manual_seed(args.seed)
guess = teacher.clone().double()
with torch.no_grad():
    for p in guess.parameters():
        p.add_(args.noise * p.norm() / p.numel() ** 0.5 * torch.randn_like(p))

score = lambda n: param_errors(n, teacher)
e0 = score(guess)
print(f"guess ({args.noise:g} rel noise): max_eps {e0['max_eps']:.3e} "
      f"mean {sum(e0['mean_eps_per_matrix']) / len(e0['mean_eps_per_matrix']):.3e}")

pool = None
if args.pool == "data":
    from data import load_data
    (_, _), (xte, _) = load_data(dims, dev)
    pool = xte.double()
X = ts.sample_points(args.points, dims[0],
                     scales=tuple(float(s) for s in args.scales.split(",")),
                     pool=pool, seed=args.seed, device=dev)
t0 = time.time()
net, info = ts.refine(bb, guess, X, args.act, dirs=args.dirs, iters=args.iters,
                      cg_iters=args.cg_iters, seed=args.seed, score=score,
                      precond={"auto": "auto", "on": True, "off": False}[args.precond])
e1 = score(net)
with torch.no_grad():
    xt = torch.randn(4000, dims[0], dtype=torch.float64, device=dev)
    fdiff = (net(xt) - teacher64(xt)).abs().max().item()
res = dict(arch=dims, act=args.act, noise=args.noise, points=args.points,
           dirs=info["dirs"], queries=info["queries"], iters=info["iters"],
           wall_s=round(time.time() - t0, 1),
           before=dict(max_eps=e0["max_eps"], per=e0["max_eps_per_matrix"]),
           after=dict(max_eps=e1["max_eps"], per=e1["max_eps_per_matrix"],
                      mean_per=e1["mean_eps_per_matrix"]),
           loss_init=info["loss_init"], loss_final=info["loss_final"],
           fresh_gauss_max_fdiff=fdiff)
print(json.dumps(res, indent=1))
if args.output:
    Path(args.output).write_text(json.dumps(res, indent=1))
