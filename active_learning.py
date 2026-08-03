"""Real (no-cheating) active-learning step to identify an under-determined unit.

Everything the attacker does uses ONLY the consensus estimate + black-box teacher
queries. Teacher weights are used solely to SCORE max_eps (eval), never to select
the unit, place queries, or solve.

  1. target selection : the unit with the largest COMMITTEE DISAGREEMENT
                         (members split on it) -- teacher-free.
  2. placement        : excite the unit by shifting queries along the CURRENT
                         estimate of its input row; re-placed each iteration as
                         the estimate improves (bootstrap).
  3. solve            : rank-1 alternating LSQ for the unit's input row AND output
                         column from R(x) = teacher(x) - recon_without_u(x)
                         ~= w_out_u * sigma(z_u(x))  (sigma=id in the active region).
"""
import argparse
import torch
from nets import MLP
from method import build_consensus
from align import scale_normalize_, greedy_perm, permute_layer_, param_errors

ALPHA = 0.01


def leaky(z):
    return torch.where(z >= 0, z, ALPHA * z)


def load_net(dims, state, dev):
    m = MLP(dims).to(dev); m.load_state_dict(state); return m


@torch.no_grad()
def committee_disagreement(pop, cons, dims):
    """Per consensus-unit max disagreement across committee members (teacher-free)."""
    H = dims[1]
    c = cons.clone(); scale_normalize_(c)
    Cf = torch.cat([c.layers[0].weight, c.layers[0].bias[:, None]], 1)
    dis = torch.zeros(H, device=c.layers[0].weight.device)
    for m in pop:
        r = m.clone(); scale_normalize_(r)
        Rf = torch.cat([r.layers[0].weight, r.layers[0].bias[:, None]], 1)
        permute_layer_(r, 0, greedy_perm(Cf, Rf))
        d = (c.layers[0].weight - r.layers[0].weight).abs().max(dim=1).values
        dis = torch.maximum(dis, d)
    return dis


@torch.no_grad()
def solve_unit(recon, teacher, X, u, iters=4, ridge=1e-4):
    """Rank-1 alternating LSQ for unit u's input row + output column on queries X."""
    dev = X.device
    Xa = torch.cat([X, torch.ones(len(X), 1, device=dev)], 1)
    A = (Xa.T @ Xa).double()
    A += ridge * A.diag().mean().clamp_min(1e-12) * torch.eye(
        Xa.shape[1], device=dev, dtype=torch.float64)
    L = torch.linalg.cholesky(A)
    H = leaky(X @ recon.layers[0].weight.T + recon.layers[0].bias)
    base = (H @ recon.layers[1].weight.T + recon.layers[1].bias) \
        - torch.outer(H[:, u], recon.layers[1].weight[:, u])   # recon without unit u
    R = teacher(X) - base                                      # ~ w_out_u * sigma(z_u)
    wout = recon.layers[1].weight[:, u].clone()
    w = None
    for _ in range(iters):
        a = (R @ wout) / wout.dot(wout).clamp_min(1e-12)       # activation estimate
        w = torch.cholesky_solve((Xa.T @ a).double()[:, None], L)[:, 0].float()
        afit = leaky(Xa @ w)                                   # realizable activation
        wout = (R.T @ afit) / afit.dot(afit).clamp_min(1e-12)  # refit output column
    recon.layers[0].weight.data[u] = w[:-1]
    recon.layers[0].bias.data[u] = w[-1]
    recon.layers[1].weight.data[:, u] = wout
    return recon


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--nt", type=int, default=60000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--shift", type=float, default=20.0,
                    help="max shift along the estimated unit direction")
    args = ap.parse_args()
    dev = args.device

    print(f"[load] {args.dump}", flush=True)
    ck = torch.load(args.dump, map_location="cpu", weights_only=False)
    dims = ck["dims"]; D = dims[0]
    teacher = load_net(dims, ck["teacher_state"], dev)
    pop = [load_net(dims, s, dev) for s in ck["pop_states"]]
    recon = build_consensus(pop, dims, quorum_ratio=0.625).to(dev)

    # (1) teacher-free target selection
    dis = committee_disagreement(pop, recon, dims)
    u = int(dis.argmax())
    print(f"[target] committee picks unit {u} "
          f"(disagreement={dis[u]:.3e}); start max_eps="
          f"{param_errors(recon, teacher)['max_eps']:.3e}", flush=True)

    g = torch.Generator(device=dev).manual_seed(0)
    for rnd in range(args.rounds):
        # (2) placement: excite unit u along its CURRENT estimated direction
        d = recon.layers[0].weight[u]
        dhat = d / d.norm().clamp_min(1e-12)
        x0 = torch.randn(args.nt, D, device=dev, generator=g) * 0.5
        c = torch.rand(args.nt, device=dev, generator=g) * args.shift
        X = x0 + c[:, None] * dhat[None, :]
        # (3) solve unit u from black-box outputs
        recon = solve_unit(recon, teacher, X, u)
        me = param_errors(recon, teacher)["max_eps"]
        print(f"  [round {rnd+1}] queries={(rnd+1)*args.nt}  "
              f"full-net max_eps={me:.3e}", flush=True)

    print(f"[done] final max_eps={param_errors(recon, teacher)['max_eps']:.3e}",
          flush=True)


if __name__ == "__main__":
    main()
