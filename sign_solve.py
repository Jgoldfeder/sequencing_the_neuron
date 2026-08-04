"""Sign-fixed exact linear endgame (chase 1e-9 past the optimizer floor).

LeakyReLU's only nonlinearity is the sign of each pre-activation: sigma(z)=g*z
with gain g in {1, alpha}. Freeze the per-sample gains and the network becomes
linear per layer, so each layer is an EXACT float64 least-squares solve rather
than a creeping optimizer. Alternate, recompute signs, iterate (active-set):

  round:
    a = gains from current W1                       # sign pattern
    solve W2,b2 exactly    (last layer LSQ, direct)
    solve W1,b1 exactly    (input layer, CGLS matrix-free) given W2 & gains a
    recompute signs; stop when they stop flipping

All float64. Warm-started from a good solution so the sign pattern is already
nearly correct.
"""
import argparse
import time
import torch
from nets import MLP
from align import param_errors

ALPHA = 0.01


def load64(dims, state, dev):
    m = MLP(dims); m.load_state_dict(state)
    return m.double().to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xckpt", help="checkpoint with X + teacher_state (+best_state)")
    ap.add_argument("--warm", default="", help="checkpoint whose 'state' warm-starts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--cg", type=int, default=400, help="CGLS iters for the W1 solve")
    ap.add_argument("--ridge", type=float, default=1e-12)
    ap.add_argument("--extra", type=int, default=0,
                    help="append this many extra random float64 queries (coverage "
                         "test: does more data drop the identifiability floor?)")
    args = ap.parse_args()
    dev = args.device

    ck = torch.load(args.xckpt, map_location="cpu", weights_only=False)
    dims = ck["dims"]; D, H, O = dims
    teacher = load64(dims, ck["teacher_state"], dev)
    X = ck["X"].double().to(dev)
    if args.extra > 0:
        sd = X.std(0, keepdim=True)                      # match query scale per-dim
        g = torch.Generator(device=dev).manual_seed(0)
        Xe = torch.randn(args.extra, D, generator=g, dtype=torch.float64,
                         device=dev) * sd
        X = torch.cat([X, Xe])
    with torch.no_grad():
        Y = teacher(X)                                   # exact float64 targets
    warm = torch.load(args.warm, map_location="cpu", weights_only=False)["state"] \
        if args.warm else ck["best_state"]
    net = load64(dims, warm, dev)
    e0 = param_errors(net, teacher)
    labels = ["L0.weight", "L0.bias", "L1.weight", "L1.bias"]
    print(f"[start] dims={dims} queries={len(X)} (+{args.extra} extra)  "
          f"max_eps={e0['max_eps']:.3e}", flush=True)
    print("  per-matrix max_eps: " + "  ".join(
        f"{n}={v:.2e}" for n, v in zip(labels, e0["max_eps_per_matrix"])), flush=True)

    ones = torch.ones(len(X), 1, dtype=torch.float64, device=dev)

    def dot(a, b):
        return sum((u * v).sum() for u, v in zip(a, b))

    for r in range(args.rounds):
        t0 = time.time()
        W1 = net.layers[0].weight.data; b1 = net.layers[0].bias.data
        W2 = net.layers[1].weight.data; b2 = net.layers[1].bias.data

        # --- sign pattern from current W1 ---
        z = X @ W1.T + b1                                # (N,H)
        a = torch.where(z >= 0, 1.0, ALPHA)             # gains (N,H)

        # --- exact last layer: y = W2 h + b2, h = a*z ---
        h = a * z
        Ha = torch.cat([h, ones], 1)                    # (N,H+1)
        A = Ha.T @ Ha
        A += args.ridge * A.diag().mean() * torch.eye(H + 1, dtype=torch.float64, device=dev)
        Wsol = torch.linalg.solve(A, Ha.T @ Y)          # (H+1,O)
        W2 = Wsol[:H].T.contiguous(); b2 = Wsol[H].contiguous()

        # --- exact input layer via CGLS: min ||L(W1,b1) - (Y-b2)|| ---
        #     L(W1,b1)_i = W2 @ ( a_i ⊙ (W1 x_i + b1) )   (linear, gains fixed)
        def L(w, bb):
            return (a * (X @ w.T + bb)) @ W2.T           # (N,O)

        def LT(rr):                                      # adjoint -> (dW1,db1)
            dg = a * (rr @ W2)                           # (N,H)
            return dg.T @ X, dg.sum(0)

        w = W1.clone(); bb = b1.clone()
        rr = (Y - b2) - L(w, bb)                         # (N,O)
        s = LT(rr); p = list(s); gamma = dot(s, s)
        for _ in range(args.cg):
            q = L(p[0], p[1])
            alpha = gamma / (q * q).sum().clamp_min(1e-300)
            w = w + alpha * p[0]; bb = bb + alpha * p[1]
            rr = rr - alpha * q
            s = LT(rr); g2 = dot(s, s)
            beta = g2 / gamma.clamp_min(1e-300)
            p = [s[0] + beta * p[0], s[1] + beta * p[1]]
            gamma = g2

        # --- commit, recompute signs, measure ---
        net.layers[0].weight.data = w; net.layers[0].bias.data = bb
        net.layers[1].weight.data = W2; net.layers[1].bias.data = b2
        znew = X @ w.T + bb
        flips = int(((znew >= 0) != (z >= 0)).sum())
        with torch.no_grad():
            mse = ((net(X) - Y) ** 2).mean().item()
        print(f"  [round {r+1}] max_eps={param_errors(net,teacher)['max_eps']:.3e}  "
              f"mse={mse:.3e}  sign_flips={flips}  ({time.time()-t0:.0f}s)", flush=True)
        if flips == 0 and r > 0:
            print("  [converged] sign pattern stable", flush=True)

    me = param_errors(net, teacher)["max_eps"]
    print(f"[done] final max_eps={me:.3e}", flush=True)
    import os
    out = os.path.join(os.path.dirname(args.xckpt), "sign_solved_3072x128x100.pt")
    torch.save({"dims": dims,
                "state": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "teacher_state": {k: v.detach().cpu() for k, v in teacher.state_dict().items()},
                "max_eps": me}, out)
    print(f"[save] -> {out}", flush=True)


if __name__ == "__main__":
    main()
