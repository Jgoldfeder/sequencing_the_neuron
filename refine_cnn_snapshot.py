"""Refine a tanh-CNN committee snapshot to high precision with the jet solver (tanh_solve_cnn.refine_cnn).

    python refine_cnn_snapshot.py recon/_pop__mergedbest_lenet_tanh_q20000_cnn__1x28x28__..._s0.pt \
        [--act tanh] [--points 2000] [--dirs 8] [--iters 30] [--out recon/_refined_lenet_tanh.pt]

The snapshot's best member is the guess; the black box is the teacher in fp64, used ONLY through its forward pass
(Oracle); the teacher's parameters are read only by the diagnostic score (per-layer eps after alignment).
"""
import argparse
import torch

from data import make_teacher_cnn
from nets import ConvNet
from tanh_solve_cnn import refine_cnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--act", default="tanh")
    ap.add_argument("--points", type=int, default=2000)
    ap.add_argument("--dirs", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--cg-iters", type=int, default=150)
    ap.add_argument("--cg-tol", type=float, default=1e-4)
    ap.add_argument("--precond", default="kfac", help="kfac (default) | none")
    ap.add_argument("--x-std", type=float, default=0.5)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    d = torch.load(a.snapshot, map_location="cpu", weights_only=False)
    shape, cfgs, fc, out = tuple(d["input_shape"]), d["conv_cfgs"], tuple(d["fc_dims"]), d["out_dim"]
    act = d.get("act", a.act) or a.act
    teacher = make_teacher_cnn(shape, cfgs, fc, out, epochs=25, seed=0, device=dev, act=act, verbose=False)
    t64 = teacher.clone().double()
    for p in t64.parameters():
        p.requires_grad_(False)
    guess = ConvNet(shape, cfgs, fc, out, act)
    guess.load_state_dict({k: v.float() for k, v in d["state_dict"].items()})
    import copy
    from align import cnn_param_errors
    score = lambda n: cnn_param_errors(copy.deepcopy(n).double(), t64)      # copies: Module.float()/.double() are in-place
    din = int(torch.tensor(shape).prod())
    g = torch.Generator(device=dev).manual_seed(0)
    X = torch.randn(a.points, din, device=dev, dtype=torch.float64, generator=g) * a.x_std
    net, info = refine_cnn(lambda x: t64(x), guess, X, dirs=a.dirs, iters=a.iters, cg_iters=a.cg_iters, cg_tol=a.cg_tol, score=score,
                           precond=(None if a.precond == 'none' else a.precond))
    print(f"[refine] {info['iters']} accepted iterations, loss {info['loss_init']:.3e} -> {info['loss_final']:.3e}, "
          f"{info['queries']:,} queries, {info['wall_s']}s", flush=True)
    e64 = cnn_param_errors(copy.deepcopy(net).double(), t64)
    print("[refine] fp64 per-matrix max eps vs teacher (W, b per layer):", " ".join(f"{v:.2e}" for v in e64["max_eps_per_matrix"]), flush=True)
    if a.out:
        torch.save({"state_dict": net.state_dict(), "info": info, "input_shape": shape, "conv_cfgs": cfgs,
                    "fc_dims": fc, "out_dim": out, "act": act}, a.out)
        print("[refine] saved", a.out)


if __name__ == "__main__":
    main()
