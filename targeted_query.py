"""Minimal targeted-query (active-learning) step to identify the one
non-identifiable unit. The dump gives us the teacher's weights so we can QUERY
it on new inputs (black-box: outputs only). Unit 508 is unrecoverable from the
original queries because its pre-activation is almost always deep-negative
(LeakyReLU's flat 0.01 region -> no signal). We synthesize queries that drive
z_508 > 0 (steep region), query the teacher, and re-solve just that unit's row.

Placements:
  control : re-solve 508 on the ORIGINAL queries        (expect ~0.755, no change)
  oracle  : queries placed with the teacher's true 508 direction  (does the
            information exist in the right queries?)
  proxy   : queries placed with the CONSENSUS's 508 estimate (can a real attacker,
            who doesn't know the teacher, get there?)

Re-solve (rank-1 peel, other 1023 units frozen at the consensus):
  R(x) = teacher(x) - consensus_without_508(x) ~= w_out_508 * sigma(z_508(x))
  on active queries sigma = identity, so a(x)=R·w_out_508/||w_out_508||^2 ~= z_508(x);
  regress a = w_in_508 . x + b_508.
"""
import argparse
import torch
from nets import MLP
from method import build_consensus
from align import scale_normalize_, greedy_perm, permute_layer_, param_errors

ALPHA = 0.01


def load_net(dims, state, dev):
    m = MLP(dims).to(dev); m.load_state_dict(state); return m


@torch.no_grad()
def unit508_err(r_fixed, teacher, wk):
    t = teacher.clone(); r = r_fixed.clone()
    scale_normalize_(t); scale_normalize_(r)
    A0 = torch.cat([t.layers[0].weight, t.layers[0].bias[:, None]], 1)
    B0 = torch.cat([r.layers[0].weight, r.layers[0].bias[:, None]], 1)
    permute_layer_(r, 0, greedy_perm(A0, B0))
    return max((t.layers[0].weight[wk] - r.layers[0].weight[wk]).abs().max().item(),
               (t.layers[0].bias[wk] - r.layers[0].bias[wk]).abs().item())


@torch.no_grad()
def solve_508(r, teacher_fn, X, wk, ridge=1e-4):
    """Re-solve unit wk's input row from queries X, freezing the other units."""
    wo = r.layers[1].weight[:, wk]
    n2 = wo.dot(wo).clamp_min(1e-12)
    H = ALPHA_leaky(X @ r.layers[0].weight.T + r.layers[0].bias)
    base = (H @ r.layers[1].weight.T + r.layers[1].bias) \
        - torch.outer(H[:, wk], wo)                      # consensus without unit wk
    R = teacher_fn(X) - base
    a = (R @ wo) / n2                                     # ~ sigma(z_508) = z_508 (active)
    Xa = torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1)
    A = (Xa.T @ Xa).double()
    A += ridge * A.diag().mean().clamp_min(1e-12) * torch.eye(
        Xa.shape[1], device=X.device, dtype=torch.float64)
    w = torch.linalg.solve(A, (Xa.T @ a).double()).float()
    r_fixed = r.clone()
    r_fixed.layers[0].weight.data[wk] = w[:-1]
    r_fixed.layers[0].bias.data[wk] = w[-1]
    return r_fixed


def ALPHA_leaky(z):
    return torch.where(z >= 0, z, ALPHA * z)


@torch.no_grad()
def make_targeted(x0, d, bth, lo=0.2, hi=12.0, gen=None):
    """Shift each base sample along d so that (d·x + bth) spans [lo, hi] > 0."""
    z0 = x0 @ d + bth
    tz = lo + (hi - lo) * torch.rand(len(x0), device=x0.device, generator=gen)
    return x0 + ((tz - z0) / d.dot(d).clamp_min(1e-12))[:, None] * d[None, :]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--nt", type=int, default=60000, help="# targeted queries")
    args = ap.parse_args()
    dev = args.device

    print(f"[load] {args.dump}", flush=True)
    ck = torch.load(args.dump, map_location="cpu", weights_only=False)
    dims = ck["dims"]; X, Y = ck["X"], ck["Y"]; D = dims[0]
    teacher = load_net(dims, ck["teacher_state"], dev)
    pop = [load_net(dims, s, dev) for s in ck["pop_states"]]
    cons = build_consensus(pop, dims, quorum_ratio=0.625).to(dev)

    # gauge-match consensus -> teacher; position i == teacher unit i
    r = cons.clone(); t = teacher.clone()
    scale_normalize_(t); scale_normalize_(r)
    A0 = torch.cat([t.layers[0].weight, t.layers[0].bias[:, None]], 1)
    B0 = torch.cat([r.layers[0].weight, r.layers[0].bias[:, None]], 1)
    permute_layer_(r, 0, greedy_perm(A0, B0))
    err0 = torch.tensor([
        max((t.layers[0].weight[i] - r.layers[0].weight[i]).abs().max().item(),
            (t.layers[1].weight[:, i] - r.layers[1].weight[:, i]).abs().max().item())
        for i in range(dims[1])])
    wk = int(err0.argmax())
    print(f"[before] worst unit {wk}: input-row max_eps={err0[wk]:.3e}  "
          f"(full-net max_eps={param_errors(r, teacher)['max_eps']:.3e})", flush=True)

    teacher_fn = lambda x: teacher(x)  # black-box query
    g = torch.Generator(device=dev).manual_seed(0)

    # --- control: original queries ---
    Xo = X[:args.nt].to(dev)
    rc = solve_508(r, teacher_fn, Xo, wk)
    print(f"[control ] original queries        -> unit {wk} "
          f"max_eps={unit508_err(rc, teacher, wk):.3e}", flush=True)

    x0 = torch.randn(args.nt, D, device=dev, generator=g) * 0.5

    # --- oracle placement: teacher's true 508 direction ---
    d_or = teacher.layers[0].weight[wk]
    b_or = teacher.layers[0].bias[wk]
    Xor = make_targeted(x0, d_or, b_or, gen=g)
    ro = solve_508(r, teacher_fn, Xor, wk)
    print(f"[oracle  ] queries exciting true u{wk} -> unit {wk} "
          f"max_eps={unit508_err(ro, teacher, wk):.3e}  "
          f"(full-net max_eps={param_errors(ro, teacher)['max_eps']:.3e})", flush=True)

    # --- proxy placement: consensus's (wrong) 508 estimate ---
    d_px = r.layers[0].weight[wk]
    b_px = r.layers[0].bias[wk]
    Xpx = make_targeted(x0, d_px, b_px, gen=g)
    rp = solve_508(r, teacher_fn, Xpx, wk)
    print(f"[proxy   ] queries via consensus est  -> unit {wk} "
          f"max_eps={unit508_err(rp, teacher, wk):.3e}  "
          f"(full-net max_eps={param_errors(rp, teacher)['max_eps']:.3e})", flush=True)


if __name__ == "__main__":
    main()
