"""Test the targeted-disagreement idea on a real saved committee.

Scenario: a committee where MOST hidden neurons reached consensus but a few did
NOT. Idea under test: instead of generic committee-disagreement queries, search
for inputs that make the committee disagree *on the non-consensus neurons
specifically*, label those with the black box (teacher), retrain, and see whether
those targeted queries resolve the stuck neurons better than (a) generic
disagreement queries and (b) just retraining on the existing data.

Everything is real: the committee, the 180k existing queries, and the teacher all
come from the saved checkpoint. New queries are labelled by the saved teacher.
"""
import argparse
import math
import time

import torch
import torch.nn.functional as F

from nets import MLP
from method import gen_queries, Cfg
from align import align_clone_to, scale_normalize_, greedy_perm

DEV = "cuda"
EPS = 0.02
QUORUM = 5          # of 8


def load_pop(states, dims):
    pop = []
    for s in states:
        m = MLP(dims).to(DEV)
        m.load_state_dict(s)
        pop.append(m)
    return pop


@torch.no_grad()
def aligned_to_ref(pop):
    """Align every member to member 0 (scale-norm + greedy perm). Returns aligned
    clones; member 0's native neuron indices are the reference frame."""
    ref = pop[0].clone()
    scale_normalize_(ref)
    return [ref] + [align_clone_to(m, ref)[1] for m in pop[1:]]


@torch.no_grad()
def neuron_spread(aligned, k):
    """Max pairwise inf-distance of hidden neuron k's (row,bias) across members
    (small => committee consensus on that neuron)."""
    feats = torch.stack([torch.cat([a.layers[0].weight[k], a.layers[0].bias[k:k + 1]])
                         for a in aligned])
    D = torch.cdist(feats, feats, p=float("inf"))
    return D.max().item()


@torch.no_grad()
def consensus_report(pop, dims, targets):
    aligned = aligned_to_ref(pop)
    H = dims[1]
    stuck = []
    for k in range(H):
        feats = torch.stack([torch.cat([a.layers[0].weight[k], a.layers[0].bias[k:k + 1]])
                             for a in aligned])
        D = torch.cdist(feats, feats, p=float("inf"))
        best = max(int((D[a] < EPS).sum()) for a in range(len(pop)))
        if best < QUORUM:
            stuck.append(k)
    tgt_spreads = {k: neuron_spread(aligned, k) for k in targets}
    tgt_resolved = [k for k in targets if tgt_spreads[k] < EPS]
    return {"n_stuck": len(stuck), "stuck": stuck,
            "tgt_spreads": tgt_spreads, "tgt_resolved": tgt_resolved}


@torch.no_grad()
def best_max_eps(pop, teacher, X, Y, bs=8192):
    from align import param_errors
    # pick lowest-MAE member on the query set, report its aligned max param error
    losses = []
    for m in pop:
        tot = n = 0
        for i in range(0, len(X), bs):
            xb, yb = X[i:i + bs].to(DEV), Y[i:i + bs].to(DEV)
            tot += (m(xb) - yb).abs().sum().item(); n += yb.numel()
        losses.append(tot / n)
    bi = min(range(len(pop)), key=lambda i: losses[i])
    return param_errors(pop[bi], teacher)["max_eps"], min(losses)


def targeted_queries(aligned, target_idx, K, din, steps=80, lr=0.1, init_std=0.5):
    """Inputs that MAXIMISE cross-member variance of the target neurons'
    activations -- i.e. queries the committee most disagrees about *on exactly
    the stuck neurons*."""
    Z = (torch.randn(K, din, device=DEV) * init_std).requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=lr)
    Ws = [a.layers[0].weight[target_idx].detach() for a in aligned]  # (|S|,din) each
    bs = [a.layers[0].bias[target_idx].detach() for a in aligned]
    for step in range(steps):
        if step in (steps // 2, int(steps * 0.8)):
            for g in opt.param_groups:
                g["lr"] /= 10
        acts = torch.stack([F.leaky_relu(Z @ W.T + b, 0.01)
                            for W, b in zip(Ws, bs)])       # (p,K,|S|)
        loss = -acts.var(dim=0).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return Z.detach()


def retrain(states, dims, X, Y, epochs, lr, bs=512, seed=0):
    pop = load_pop(states, dims)
    opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in pop]
    g = torch.Generator().manual_seed(seed)
    N = len(X)
    for ep in range(epochs):
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            xb, yb = X[idx].to(DEV), Y[idx].to(DEV)
            for m, o in zip(pop, opts):
                o.zero_grad()
                ((m(xb) - yb) ** 2).mean().backward()
                o.step()
    return pop


@torch.no_grad()
def teacher_kink_rate(teacher, aligned0, targets, Xq):
    """Map the stuck neurons (member-0 frame) to the teacher's neurons, then report
    how often the query set Xq crosses those teacher neurons' kinks (activation
    rate + whether both signs occur) -- the mechanism check."""
    t = teacher.clone(); scale_normalize_(t)
    r0 = aligned0.clone()  # already scale-normalised member 0
    TF = torch.cat([t.layers[0].weight, t.layers[0].bias[:, None]], 1)
    RF = torch.cat([r0.layers[0].weight[targets], r0.layers[0].bias[targets, None]], 1)
    D = torch.cdist(RF, TF, p=1)
    tgt_teacher = D.argmin(1).tolist()      # teacher neuron for each stuck one
    W = teacher.layers[0].weight[tgt_teacher]  # (|S|,din) in teacher's native scale
    b = teacher.layers[0].bias[tgt_teacher]
    Z = Xq.to(DEV) @ W.T + b                # (K,|S|)
    frac_pos = (Z > 0).float().mean(0)      # activation rate per stuck neuron
    both = ((Z.min(0).values < 0) & (Z.max(0).values > 0))
    return tgt_teacher, frac_pos.tolist(), int(both.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="recon/v18_lbfgs__784x256x10__s0.pt")
    ap.add_argument("--K", type=int, default=15000, help="new queries per branch")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()

    print(f"[load] {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    dims = ck["dims"]; din = dims[0]
    teacher = MLP(dims).to(DEV); teacher.load_state_dict(ck["teacher_state"])
    X0, Y0 = ck["X"], ck["Y"]                       # CPU, existing queries + labels
    pop0 = load_pop(ck["pop_states"], dims)
    print(f"[load] dims={dims} members={len(pop0)} existing_queries={len(X0)}", flush=True)

    base = consensus_report(pop0, dims, targets=[])
    targets = base["stuck"]
    print(f"[state] non-consensus neurons: {len(targets)}/{dims[1]} -> {targets}", flush=True)
    b_eps0, b_loss0 = best_max_eps(pop0, teacher, X0, Y0)
    print(f"[state] best-member max_eps={b_eps0:.3e}  MAE(D)={b_loss0:.3e}", flush=True)

    aligned0 = aligned_to_ref(pop0)
    tgt = torch.tensor(targets, device=DEV)

    # --- generate the two query sets ---
    print(f"\n[gen] {args.K} TARGETED queries (disagreement on the {len(targets)} stuck neurons)...", flush=True)
    Xt = targeted_queries(aligned0, tgt, args.K, din)
    print(f"[gen] {args.K} GENERIC queries (standard median-pair output disagreement)...", flush=True)
    cfg = Cfg(p=len(pop0), q=args.K, disagree="median_pair")
    ggen = torch.Generator(device=DEV).manual_seed(0)
    Xg = gen_queries(pop0, cfg, din, DEV, ggen)

    # --- mechanism check: do targeted queries excite the stuck neurons? ---
    tgt_teacher, fpt, both_t = teacher_kink_rate(teacher, aligned0[0], tgt, Xt)
    _,           fpg, both_g = teacher_kink_rate(teacher, aligned0[0], tgt, Xg)
    print(f"\n[mechanism] teacher activation rate on the {len(targets)} stuck neurons:")
    print(f"   TARGETED queries: mean {sum(fpt)/len(fpt):.3f}  kinks-crossed {both_t}/{len(targets)}")
    print(f"   GENERIC  queries: mean {sum(fpg)/len(fpg):.3f}  kinks-crossed {both_g}/{len(targets)}")

    # --- label with the black box (teacher) ---
    with torch.no_grad():
        Yt = torch.cat([teacher(Xt[i:i+8192]) for i in range(0, len(Xt), 8192)]).cpu()
        Yg = torch.cat([teacher(Xg[i:i+8192]) for i in range(0, len(Xg), 8192)]).cpu()
    Xt = Xt.cpu(); Xg = Xg.cpu()

    Xall_t = torch.cat([X0, Xt]); Yall_t = torch.cat([Y0, Yt])
    Xall_g = torch.cat([X0, Xg]); Yall_g = torch.cat([Y0, Yg])

    # --- three branches from the SAME snapshot ---
    branches = {
        "none    (retrain on existing D only)": (X0, Y0),
        "generic (D + generic disagreement)  ": (Xall_g, Yall_g),
        "TARGETED(D + targeted on stuck)     ": (Xall_t, Yall_t),
    }
    print(f"\n[retrain] {args.epochs} epochs, lr={args.lr}, from the same snapshot\n", flush=True)
    print(f"{'branch':<40}{'stuck->':>8}{'resolved':>10}{'best max_eps':>14}{'wall':>7}")
    print("-" * 79)
    print(f"{'(before)':<40}{len(targets):>8}{'0':>10}{b_eps0:>14.3e}{'-':>7}")
    for name, (X, Y) in branches.items():
        t0 = time.time()
        pop = retrain(ck["pop_states"], dims, X, Y, args.epochs, args.lr)
        rep = consensus_report(pop, dims, targets)
        beps, _ = best_max_eps(pop, teacher, X0, Y0)
        n_res = len(targets) - len([k for k in targets if rep["tgt_spreads"][k] >= EPS])
        print(f"{name:<40}{rep['n_stuck']:>8}{n_res:>10}{beps:>14.3e}{time.time()-t0:>6.0f}s",
              flush=True)

    print(f"\n[targets] per-neuron committee spread before -> after (targeted branch):")
    pop_t = retrain(ck["pop_states"], dims, Xall_t, Yall_t, args.epochs, args.lr)
    rep_t = consensus_report(pop_t, dims, targets)
    for k in targets:
        print(f"   neuron {k:4d}: {neuron_spread(aligned0, k):.3e} -> "
              f"{rep_t['tgt_spreads'][k]:.3e}  "
              f"{'RESOLVED' if rep_t['tgt_spreads'][k] < EPS else 'still stuck'}")


if __name__ == "__main__":
    main()
