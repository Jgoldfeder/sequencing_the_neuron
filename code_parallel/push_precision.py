"""How low can max_eps go? Take a saved reconstruction checkpoint (teacher +
solution + queries) and run a full-data float64 endgame, with targets recomputed
from the teacher in float64 (removes the float32 target-noise floor). Reports
max_eps vs the stored-float32-target fit for contrast."""
import argparse
import time
import torch
from nets import MLP
from align import param_errors


def load64(dims, state, dev):
    m = MLP(dims); m.load_state_dict(state)
    return m.double().to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--loss", choices=["mse", "mae", "staged"], default="staged",
                    help="staged = MSE into the basin, then MAE for the flat "
                         "directions MSE's vanishing gradient stalls on")
    ap.add_argument("--mse-steps", type=int, default=40)
    ap.add_argument("--mae-steps", type=int, default=300)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--report", type=int, default=20)
    ap.add_argument("--target", choices=["f64", "f32"], default="f64",
                    help="fit teacher outputs recomputed in float64, or the "
                         "stored float32 Y")
    args = ap.parse_args()
    dev = args.device

    print(f"[load] {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    dims = ck["dims"]
    teacher = load64(dims, ck["teacher_state"], dev)
    net = load64(dims, ck["best_state"], dev)
    X = ck["X"].double().to(dev)
    if args.target == "f64":
        with torch.no_grad():
            Y = teacher(X)                      # float64 targets (re-query)
    else:
        Y = ck["Y"].double().to(dev)            # stored float32 targets, cast
    print(f"[setup] dims={dims} queries={len(X)} target={args.target} "
          f"device={dev}", flush=True)

    def me():
        return param_errors(net, teacher)["max_eps"]

    @torch.no_grad()
    def losses():
        r = net(X) - Y
        return (r * r).mean().item(), r.abs().mean().item()

    m0, a0 = losses()
    print(f"[start] max_eps={me():.3e}  mse={m0:.3e}  mae={a0:.3e}", flush=True)

    t0 = time.time()

    def run_phase(kind, steps):
        opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=20,
                                history_size=50, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-30, tolerance_change=1e-30)

        def closure():
            opt.zero_grad()
            r = net(X) - Y
            loss_ = (r * r).mean() if kind == "mse" else r.abs().mean()
            loss_.backward()
            return loss_
        for s in range(steps):
            opt.step(closure)
            if (s + 1) % args.report == 0:
                mse, mae = losses()
                print(f"  [{kind}] step {s+1:4d}: mse={mse:.3e} mae={mae:.3e} "
                      f"max_eps={me():.3e}  ({time.time()-t0:.0f}s)", flush=True)

    if args.loss == "staged":
        run_phase("mse", args.mse_steps)
        run_phase("mae", args.mae_steps)
    else:
        run_phase(args.loss, args.steps)
    final = me()
    print(f"[done] final max_eps={final:.3e}", flush=True)
    import os
    out = os.path.join(os.path.dirname(args.ckpt),
                       "precision_solved_3072x128x100.pt")
    torch.save({"dims": dims, "loss": args.loss,
                "state": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "teacher_state": {k: v.detach().cpu()
                                  for k, v in teacher.state_dict().items()},
                "max_eps": final}, out)
    print(f"[save] -> {out}", flush=True)


if __name__ == "__main__":
    main()
