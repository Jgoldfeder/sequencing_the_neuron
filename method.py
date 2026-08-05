"""Reconstruction algorithm (paper Appendix B, Algorithms 1 & 2) with variant
hooks for benchmarking methodological improvements.

Baseline (paper-faithful):
  - Committee Disagreement Sampling: learnable input tensor optimized by
    gradient ascent on normalized pairwise L1 output disagreement (App. C).
  - Population of p surrogates trained on ALL accumulated samples each outer
    iteration; LR step-decay; return lowest-loss member.

Variant hooks:
  disagree      : 'mean_pair' (paper) | 'min_pair' | 'median_pair' | 'variance'
  query_box     : bound queries into [-box, box]^d via tanh parameterization
  query_div_w   : weight of inter-query repulsion penalty (query diversity)
  window        : keep only the last `window` outer-iterations of samples
                  (0 = keep all, paper behavior)
  maint_every   : every k iters, align population to best member, try soup,
                  restart worst members (0 = never)
  restart_worst : number of worst members restarted during maintenance
  polish_f64    : final float64 polish phase on CPU
"""
import json
import math
import time
from dataclasses import dataclass, field, asdict

import torch

from align import (scale_normalize_, greedy_perm, permute_layer_,
                   param_errors, align_clone_to)
from nets import MLP


@dataclass
class Cfg:
    # population / budget
    p: int = 8                 # population size
    q: int = 1500              # new queries per outer iteration
    outer: int = 40            # outer iterations
    epochs: int = 10         # training epochs per outer iteration
                           # (VALIDATED: must be high enough to fit D tightly;
                           #  epochs=2 stalls completely - the underfit trap)
    lr: float = 1e-3           # surrogate training lr (Adam)
    lr_sched: tuple = (0.6, 0.85)  # decay lr /10 at these outer-iter fracs
    batch: int = 512
    # query generation
    qg_steps: int = 30         # gradient steps optimizing the query tensor
    qg_lr: float = 0.1
    qg_sched: tuple = (0.5, 0.8)
    qg_init_std: float = 0.5
    # variant switches
    disagree: str = "mean_pair"
    query_box: float = 0.0     # 0 = unbounded (paper); else tanh box bound
    query_div_w: float = 0.0
    window: int = 0
    maint_every: int = 0
    restart_worst: int = 0
    polish_f64: bool = False
    polish_epochs: int = 3
    warmstart_iters: int = 0   # first k iters use random Gaussian queries
    fit_delta: float = 0.0     # >0: stop inner epochs early when batch L1 < delta
    stop_loss: float = 0.0     # >0: early-stop outer when best_loss < stop_loss
    stop_agree: float = 0.0    # >0: ...AND population weight dispersion < this
    stop_patience: int = 3
    # batch-2 hypotheses
    fit_loss: str = "l1"       # 'l1' (paper) | 'mse' (Opus A2: residual-proportional grads)
    lastlayer_every: int = 0   # >0: closed-form ridge LS solve of last layer every k iters (Opus A12)
    popavg_kappa: float = 0.0  # >0: aligned averaging of members within kappa x best loss (Opus A9)
    lbfgs_polish: bool = False # float64 LBFGS squared-loss endgame (Opus A2+A3-lite)
    gate_kappa: float = 0.0    # >0: exclude members with loss > kappa x best from disagreement (Opus T7)
    # cluster-consensus diagnostic
    cluster_quorum: float = 0.625  # quorum as a ratio of the committee (5/8);
                                  # a unit needs >= ceil(ratio*p) members to
                                  # agree, else the whole consensus is n/a.
                                  # Matches the --fast dump trigger so the
                                  # printed `cluster` column (and --combine)
                                  # key off the same consensus --fast stops on.
    cluster_gate: float = 10.0    # NOTE: no longer applied to the logged/combine
                                  # consensus (that path is now ungated, to match
                                  # the --fast trigger). Only the dead diagnostic
                                  # wrapper cluster_consensus() still references it.
    combine: bool = False         # the FIRST outer iter the consensus becomes
                                  # available (n/a -> answer), replace the worst
                                  # committee member with the (polished)
                                  # consensus net
    solver_polish: bool = False   # at the end of each outer iter, tighten each
                                  # member with the staged LBFGS recipe
                                  # (MSE then MAE) -> faster committee agreement
    solverwindow: int = 0         # every query-solver (in-loop polish, --fast
                                  # endgame, lbfgs endgame) fits on the last
                                  # solverwindow outer iters of queries; 0 = all
    verbose: bool = False         # print per-member polish detail (loss
                                  # before->after, #evals, time)
    dump_at_iter: int = 0         # >0: torch.save population+queries at this
    dump_path: str = ""           # outer iter (real schedule intact) and stop
    stop_on_consensus: bool = False  # dump+stop the FIRST log-iter a consensus
                                     # (ungated 5/8) forms, instead of a fixed iter
    combine_polish_lr: float = 1e-4   # low-LR polish of the consensus before
    combine_polish_steps: int = 200   # injecting: removes the averaging /
                                      # assembly artifact (loss ~200x lower) so
                                      # the injected member wins loss-selection,
                                      # without drifting its recovered params
    # logging
    log_every: int = 5
    eval_pts: int = 2000
    pop_save_every: int = 0       # >0: snapshot the population every k iters
    pop_save_path: str = ""       # (inspect mid-run; overwrites, no queries)


# ---------------------------------------------------------------- queries --
def _normalize(v):
    return v / v.abs().sum(dim=-1, keepdim=True).clamp_min(1e-12)


def disagreement(outs, mode):
    """outs: (p, q, out_dim) frozen committee outputs. Return loss to
    MINIMIZE (negative disagreement)."""
    f = _normalize(outs)
    if mode == "variance":
        mean = f.mean(dim=0, keepdim=True)
        var = ((f - mean) ** 2).sum(-1).mean(dim=0)  # per sample
        return -var.mean()
    # pairwise distance matrix per sample: (p, p, q)
    D = (f.unsqueeze(1) - f.unsqueeze(0)).abs().sum(-1)
    p = f.shape[0]
    i, j = torch.triu_indices(p, p, offset=1, device=f.device)
    pairs = D[i, j, :]  # (n_pairs, q)
    if mode == "mean_pair":
        per_sample = pairs.mean(dim=0)
    elif mode == "min_pair":
        per_sample = pairs.min(dim=0).values
    elif mode == "median_pair":
        per_sample = pairs.median(dim=0).values
    else:
        raise ValueError(mode)
    return -per_sample.mean()


def _repulsion(Z, n_pairs=65536):
    """Mean squared cosine similarity over random query pairs (to minimize)."""
    q = Z.shape[0]
    flat = Z.reshape(q, -1).float()
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    i = torch.randint(0, q, (n_pairs,), device=Z.device)
    j = torch.randint(0, q, (n_pairs,), device=Z.device)
    cos = (flat[i] * flat[j]).sum(-1)
    return (cos ** 2).mean()


def gen_queries(pop, cfg, input_dim, device, gen, losses=None):
    """Paper Algorithm 2 (+ box / diversity / fit-gating variants)."""
    members = pop
    if cfg.gate_kappa > 0 and losses is not None:
        bl = min(losses)
        gated = [m for m, l in zip(pop, losses)
                 if l <= bl * cfg.gate_kappa]
        if len(gated) >= 2:
            members = gated
    q = cfg.q
    if cfg.query_box > 0:
        raw = torch.randn(q, input_dim, generator=gen, device=device)
        raw.requires_grad_(True)
        Z = raw
        feats = lambda: cfg.query_box * torch.tanh(raw)  # noqa: E731
    else:
        Z = torch.randn(q, input_dim, generator=gen,
                        device=device) * cfg.qg_init_std
        Z.requires_grad_(True)
        feats = lambda: Z  # noqa: E731
    opt = torch.optim.Adam([Z], lr=cfg.qg_lr)
    sched = {int(s * cfg.qg_steps) for s in cfg.qg_sched}
    for step in range(cfg.qg_steps):
        if step in sched:
            for g in opt.param_groups:
                g["lr"] /= 10
        I = feats()
        outs = torch.stack([net(I) for net in members])  # (p', q, out)
        loss = disagreement(outs, cfg.disagree)
        if cfg.query_div_w > 0:
            loss = loss + cfg.query_div_w * _repulsion(I)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return feats().detach()


# ------------------------------------------------------------- maintenance --
@torch.no_grad()
def solve_last_layer_(net, X, Y, ridge=1e-6, chunk=8192):
    """Closed-form ridge least-squares solve of the final layer given the
    penultimate features (Opus A12: the last layer is a convex quadratic;
    variable projection instead of first-order SGD)."""
    dev = net.layers[-1].weight.device
    Hs = []
    for i in range(0, len(X), chunk):
        h = X[i:i + chunk].to(dev)
        for l in net.layers[:-1]:
            h = net.act(l(h))
        Hs.append(h)
    H = torch.cat(Hs).cpu().double()
    Yd = Y.cpu().double()
    Ha = torch.cat([H, torch.ones(len(H), 1, dtype=torch.float64)], dim=1)
    A = Ha.T @ Ha
    A = A + ridge * A.diag().mean().clamp_min(1e-12) * torch.eye(
        A.shape[0], dtype=torch.float64)
    W = torch.linalg.solve(A, Ha.T @ Yd)
    last = net.layers[-1]
    dev = last.weight.device
    last.weight.copy_(W[:-1].T.float().to(dev))
    last.bias.copy_(W[-1].float().to(dev))


@torch.no_grad()
def aligned_pop_average(pop, losses, kappa):
    """Aligned coordinate-wise average of members within kappa x best loss
    (Opus A9). Members are aligned into the best member's frame with
    function-preserving transforms; returns (avg_net, k)."""
    bi = min(range(len(pop)), key=lambda i: losses[i])
    thresh = losses[bi] * kappa
    idxs = [i for i in range(len(pop)) if losses[i] <= thresh]
    if len(idxs) < 2:
        return None, len(idxs)
    anchor = pop[bi].clone()
    aligned = [anchor]
    for i in idxs:
        if i == bi:
            continue
        m = pop[i].clone()
        align_member_to(anchor, m)
        aligned.append(m)
    return soup_of(aligned), len(idxs)


@torch.no_grad()
def align_member_to(best, member):
    """Function-preserving alignment of member into best's frame (in-place
    on member)."""
    scale_normalize_(best)  # scale-norm is function preserving
    scale_normalize_(member)
    for l in range(len(best.layers) - 1):
        Af = torch.cat([best.layers[l].weight,
                        best.layers[l].bias.unsqueeze(1)], dim=1)
        Bf = torch.cat([member.layers[l].weight,
                        member.layers[l].bias.unsqueeze(1)], dim=1)
        perm = greedy_perm(Af, Bf)
        permute_layer_(member, l, perm)


@torch.no_grad()
def soup_of(members):
    avg = members[0].clone()
    for avg_p, *ps in zip(avg.parameters(),
                          *[m.parameters() for m in members]):
        avg_p.copy_(torch.stack(list(ps)).mean(0))
    return avg


@torch.no_grad()
def build_consensus(pop, dims, losses=None, eps=0.02,
                    quorum_ratio=0.75, gate_kappa=10.0):
    """Teacher-free tight-cluster alignment across the committee, then per-unit
    consensus. Returns the consensus NET (reconstructed from committee
    agreement instead of the single best member), or None (n/a) if the
    committee doesn't back it strongly enough. Fully teacher-free.
    Single-hidden-layer nets only.

    Robustness (prefer n/a over a shaky consensus):
      - loss-gate: only members within `gate_kappa` x the best training loss
        vote, so a collapsed/correlated majority of stuck members can't form a
        spurious cluster (needs `losses`; without it, all members vote).
      - quorum: a unit is only trusted if >= ceil(quorum_ratio * p) members
        agree tightly (default 0.75 => 6 of 8). If fewer well-fit members than
        the quorum exist, or any unit falls short, the whole thing is n/a.

    Method: scale-normalize gated members, connect units that are mutual
    nearest neighbours AND within `eps` (max-element distance), union-find into
    clusters (one per teacher unit), average each cluster. Stragglers are loners
    below quorum; collapsed members are gated out entirely."""
    import math
    if len(dims) != 3 or len(pop) < 2:
        return None
    P, H = len(pop), dims[1]
    quorum = math.ceil(quorum_ratio * P)          # ratio of the FULL committee

    idxs = list(range(P))
    if losses is not None and gate_kappa > 0:
        bl = min(losses)
        idxs = [i for i in range(P) if losses[i] <= gate_kappa * bl]
    if len(idxs) < quorum:
        return None  # too few well-fit members to reach quorum -> n/a

    mem = [pop[i].clone() for i in idxs]
    for r in mem:
        scale_normalize_(r)
    g = len(mem)
    F = [torch.cat([r.layers[0].weight, r.layers[0].bias[:, None]], 1)
         for r in mem]
    parent = list(range(g * H))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(g):
        for b in range(a + 1, g):
            D = torch.cdist(F[a].float(), F[b].float(), p=float("inf"))
            nn_ab, nn_ba = D.argmin(1), D.argmin(0)
            for i in range(H):
                j = int(nn_ab[i])
                if int(nn_ba[j]) == i and D[i, j] < eps:
                    ra, rb = find(a * H + i), find(b * H + j)
                    if ra != rb:
                        parent[ra] = rb

    from collections import defaultdict
    clusters = defaultdict(list)
    for node in range(g * H):
        clusters[find(node)].append(node)
    good = sorted((v for v in clusters.values() if len(v) >= quorum),
                  key=len, reverse=True)[:H]
    if len(good) < H:
        return None  # some unit lacks a quorum -> n/a (prefer n/a over shaky)

    net = mem[0].clone()
    for k, comp in enumerate(good):
        rows = torch.stack([mem[n // H].layers[0].weight[n % H] for n in comp])
        bias = torch.stack([mem[n // H].layers[0].bias[n % H] for n in comp])
        outs = torch.stack([mem[n // H].layers[1].weight[:, n % H]
                            for n in comp])
        net.layers[0].weight.data[k] = rows.mean(0)
        net.layers[0].bias.data[k] = bias.mean(0)
        net.layers[1].weight.data[:, k] = outs.mean(0)
    net.layers[1].bias.data = mem[0].layers[1].bias.clone()
    return net


@torch.no_grad()
def consensus_neuron_stats(pop, teacher, dims, eps=0.02, quorum_ratio=0.625):
    """Per-neuron consensus diagnostic (single hidden layer, ungated -- matches
    the logged consensus). Instead of the all-or-nothing full consensus, report
    how many hidden neurons reached a quorum consensus across the committee, and
    the max/mean parameter error on JUST those neurons (each averaged over its
    cluster, then aligned to the teacher). Teacher used only for scoring.
    Returns {n_consensus, n_total, max_eps, mean_eps} or None."""
    import math
    from collections import defaultdict
    if len(dims) != 3 or len(pop) < 2:
        return None
    P, H = len(pop), dims[1]
    quorum = math.ceil(quorum_ratio * P)
    mem = [m.clone() for m in pop]
    for r in mem:
        scale_normalize_(r)
    F = [torch.cat([r.layers[0].weight, r.layers[0].bias[:, None]], 1) for r in mem]
    parent = list(range(P * H))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a in range(P):
        for b in range(a + 1, P):
            D = torch.cdist(F[a].float(), F[b].float(), p=float("inf"))
            ab, ba = D.argmin(1), D.argmin(0)
            for i in range(H):
                j = int(ab[i])
                if int(ba[j]) == i and D[i, j] < eps:
                    ra, rb = find(a * H + i), find(b * H + j)
                    if ra != rb:
                        parent[ra] = rb
    clusters = defaultdict(list)
    for node in range(P * H):
        clusters[find(node)].append(node)
    good = [v for v in clusters.values() if len(v) >= quorum]
    if not good:
        return {"n_consensus": 0, "n_total": H, "max_eps": None, "mean_eps": None}

    rows = torch.stack([torch.stack([mem[n // H].layers[0].weight[n % H]
                                     for n in c]).mean(0) for c in good])
    bias = torch.stack([torch.stack([mem[n // H].layers[0].bias[n % H]
                                     for n in c]).mean(0) for c in good])
    outs = torch.stack([torch.stack([mem[n // H].layers[1].weight[:, n % H]
                                     for n in c]).mean(0) for c in good])
    # align consensus neurons to teacher (scale-normalized); greedy L1 match
    t = teacher.clone(); scale_normalize_(t)
    TF = torch.cat([t.layers[0].weight, t.layers[0].bias[:, None]], 1)
    CF = torch.cat([rows, bias[:, None]], 1)
    Dm = torch.cdist(CF.float(), TF.float(), p=1)
    l0d, l1d = [], []                       # L0 = input weights+bias, L1 = output col
    match_ti = [0] * len(good)              # teacher hidden neuron matched to cluster ci
    for _ in range(len(good)):
        flat = int(Dm.argmin()); ci, ti = flat // H, flat % H
        l0d.append((rows[ci] - t.layers[0].weight[ti]).abs())
        l0d.append((bias[ci] - t.layers[0].bias[ti]).abs().reshape(1))
        l1d.append((outs[ci] - t.layers[1].weight[:, ti]).abs())
        match_ti[ci] = ti
        Dm[ci, :] = float("inf"); Dm[:, ti] = float("inf")
    L0 = torch.cat(l0d); L1 = torch.cat(l1d); alld = torch.cat([L0, L1])
    out = {"n_consensus": len(good), "n_total": H,
           "max_eps": alld.max().item(), "mean_eps": alld.mean().item(),
           "l0_max": L0.max().item(), "l0_mean": L0.mean().item(),
           "l1_max": L1.max().item(), "l1_mean": L1.mean().item()}

    # --- output-unit consensus: for each of the O output units, do the members
    #     agree on its bias AND its weights to the consensus hidden neurons? ---
    O = dims[2]
    bias_out = torch.stack([r.layers[1].bias for r in mem])          # (P, O)
    agree = (bias_out.max(0).values - bias_out.min(0).values) < eps  # (O,)
    for c in good:
        w = torch.stack([mem[n // H].layers[1].weight[:, n % H] for n in c])  # (|c|,O)
        agree = agree & ((w.max(0).values - w.min(0).values) < eps)
    n_out = int(agree.sum())
    out["n_out_consensus"] = n_out
    out["out_total"] = O
    if n_out > 0:
        ti_idx = torch.tensor(match_ti, device=outs.device)
        bias_err = (bias_out.mean(0) - t.layers[1].bias).abs()       # (O,)
        w_err = (outs.T - t.layers[1].weight[:, ti_idx]).abs()       # (O, K)
        od = torch.cat([bias_err[agree].reshape(-1), w_err[agree].reshape(-1)])
        out["out_max"] = od.max().item()
        out["out_mean"] = od.mean().item()
    else:
        out["out_max"] = None
        out["out_mean"] = None
    return out


@torch.no_grad()
def cluster_consensus(pop, teacher, dims, losses=None, eps=0.02,
                      quorum_ratio=0.75, gate_kappa=10.0):
    """Diagnostic wrapper around build_consensus: returns the consensus net's
    max parameter error (teacher used only to score), or None (n/a)."""
    net = build_consensus(pop, dims, losses=losses, eps=eps,
                          quorum_ratio=quorum_ratio, gate_kappa=gate_kappa)
    return None if net is None else param_errors(net, teacher)["max_eps"]


def solver_polish_(net, X, Y, mse_steps=15, mae_steps=15,
                   verbose=False, tag="", bs=8192):
    """Tighten a single net's fit in-place with the staged LBFGS recipe:
    MSE (descend into the basin) then MAE (constant gradient finishes the flat
    directions MSE's vanishing gradient abandons). Modifies net.parameters()
    in place so the caller's optimizer stays valid. GPU float32.

    X/Y may live on CPU: the full query matrix can exceed GPU memory at large
    input dims, so the closure streams `bs`-row chunks to the net's device and
    accumulates the gradient across all chunks into .grad before each LBFGS
    step -- identical to one full-batch backward, just bounded in peak memory."""
    dev = next(net.parameters()).device
    dt = next(net.parameters()).dtype     # follow the net's precision (fp32/fp64)
    denom = X.shape[0] * Y.shape[1]
    for kind, steps in (("mse", mse_steps), ("mae", mae_steps)):
        opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=20,
                                history_size=20, line_search_fn="strong_wolfe")
        n_eval = [0]

        def closure():
            n_eval[0] += 1
            opt.zero_grad()
            total = 0.0
            for i in range(0, X.shape[0], bs):
                xb, yb = X[i:i + bs].to(dev, dt), Y[i:i + bs].to(dev, dt)
                r = net(xb) - yb
                loss = (r ** 2).sum() if kind == "mse" else r.abs().sum()
                (loss / denom).backward()
                total += loss.item() / denom
            return total
        if verbose:
            t0 = time.time()
            l0 = l1_on([net], X, Y)[0]
        for _ in range(steps):
            opt.step(closure)
        if verbose:
            print(f"    [polish]{tag} {kind}: L1 {l0:.3e}->"
                  f"{l1_on([net], X, Y)[0]:.3e}  {n_eval[0]} evals  "
                  f"{time.time() - t0:.1f}s", flush=True)
    return net


def polish_consensus(net, X, Y, lr=1e-4, steps=200, max_samples=30000):
    """Gentle low-LR polish of a consensus net on query MSE. The consensus has
    good parameters but a small function-assembly artifact from averaging
    scale-normalized members (=> high loss); a low-LR pass removes it (loss
    drops ~100-200x) while the recovered parameters hold (max_eps stable). A
    higher LR lowers loss faster but starts pulling weakly-identified units back
    toward their query-loss-optimal (wrong) values, so keep it low. Returns a
    fresh polished clone; does not mutate `net`."""
    net = net.clone()
    dev = next(net.parameters()).device
    Xs = (X[-max_samples:] if len(X) > max_samples else X).to(dev)
    Ys = (Y[-max_samples:] if len(Y) > max_samples else Y).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((net(Xs) - Ys) ** 2).mean()
        loss.backward()
        opt.step()
    return net


# -------------------------------------------------------------- evaluation --
@torch.no_grad()
def batched(net, X, bs=4096):
    return torch.cat([net(X[i:i + bs]) for i in range(0, len(X), bs)])


@torch.no_grad()
def l1_on(pop, X, Y, bs=4096):
    losses = []
    for net in pop:
        dev = next(net.parameters()).device
        dt = next(net.parameters()).dtype
        tot, n = 0.0, 0
        for i in range(0, len(X), bs):
            xb, yb = X[i:i + bs].to(dev, dt), Y[i:i + bs].to(dev, dt)
            tot += (net(xb) - yb).abs().sum().item()
            n += yb.numel()
        losses.append(tot / n)
    return losses


@torch.no_grad()
def agreement(net, teacher, X, bs=4096):
    pa = batched(net, X, bs).argmax(1)
    ta = batched(teacher, X, bs).argmax(1)
    return (pa == ta).float().mean().item()


# ------------------------------------------------------------ main routine --
def reconstruct(teacher, dims, cfg: Cfg, device, teacher_eval_pts, seed=0,
                save_recon=None):
    gen = torch.Generator(device=device).manual_seed(seed)
    pop = [MLP(dims).to(device) for _ in range(cfg.p)]
    opts = [torch.optim.Adam(n.parameters(), lr=cfg.lr) for n in pop]
    # query buffer lives on CPU/host RAM; minibatches are streamed to `device`
    # during training so the full accumulated query set never sits on the GPU.
    X = torch.empty(0, dims[0])
    Y = torch.empty(0, dims[-1])
    decay_at = {int(s * cfg.outer) for s in cfg.lr_sched}
    log = []
    t0 = time.time()
    best = None
    stop_hits = 0
    combined_done = False

    for t in range(cfg.outer):
        # --- query the blackbox ---
        if t < cfg.warmstart_iters:
            I = torch.randn(cfg.q, dims[0], generator=gen,
                            device=device) * cfg.qg_init_std
        else:
            losses_qg = (l1_on(pop, X, Y)
                         if cfg.gate_kappa > 0 and t > 0 else None)
            I = gen_queries(pop, cfg, dims[0], device, gen,
                            losses=losses_qg)
        with torch.no_grad():
            T = teacher(I)
        X = torch.cat([X, I.cpu()])
        Y = torch.cat([Y, T.cpu()])
        if cfg.window > 0:
            keep = cfg.window * cfg.q
            X, Y = X[-keep:], Y[-keep:]

        # --- lr schedule ---
        if t in decay_at:
            for o in opts:
                for g in o.param_groups:
                    g["lr"] /= 10

        # --- train population on D ---
        n = len(X)
        for ep in range(cfg.epochs):
            perm = torch.randperm(n, generator=gen, device=device)
            ep_loss = 0.0
            nb = 0
            for i in range(0, n, cfg.batch):
                idx = perm[i:i + cfg.batch].cpu()
                xb, yb = X[idx].to(device), Y[idx].to(device)
                bl = 0.0
                for net, opt in zip(pop, opts):
                    opt.zero_grad()
                    if cfg.fit_loss == "mse":
                        loss = ((net(xb) - yb) ** 2).mean()
                    else:
                        loss = (net(xb) - yb).abs().mean()
                    loss.backward()
                    opt.step()
                    bl += loss.item()
                ep_loss += bl / cfg.p
                nb += 1
            if cfg.fit_delta > 0 and ep_loss / max(nb, 1) < cfg.fit_delta:
                break

        # --- per-iteration solver polish: tighten every member with the
        #     staged MSE->MAE LBFGS recipe on the recent solverwindow of
        #     queries (in place, so the Adam optimizers stay valid) ---
        if cfg.solver_polish:
            keep = cfg.solverwindow * cfg.q
            Xp, Yp = ((X[-keep:], Y[-keep:]) if keep and len(X) > keep
                      else (X, Y))
            if cfg.verbose:
                print(f"  [polish] it {t + 1}: {cfg.p} members on {len(Xp)} "
                      f"queries", flush=True)
            for mi, net in enumerate(pop):
                solver_polish_(net, Xp, Yp, verbose=cfg.verbose,
                               tag=f" m{mi}")

        # --- closed-form last-layer solve (variable projection) ---
        if cfg.lastlayer_every and (t + 1) % cfg.lastlayer_every == 0:
            for net in pop:
                solve_last_layer_(net, X, Y)

        # --- early stopping (App F signals, actually wired in) ---
        if cfg.stop_loss > 0 and (t + 1) % cfg.log_every == 0:
            losses_now = l1_on(pop, X, Y)
            bl = min(losses_now)
            disp = 0.0
            if cfg.stop_agree > 0:
                bi = min(range(cfg.p), key=lambda i: losses_now[i])
                anchor = pop[bi].clone()
                disp = max(
                    (anchor.layers[0].weight - m.layers[0].weight)
                    .abs().max().item()
                    for m in pop if m is not pop[bi])
            if bl < cfg.stop_loss and (cfg.stop_agree <= 0 or
                                       disp < cfg.stop_agree):
                stop_hits += 1
                if stop_hits >= cfg.stop_patience:
                    best = pop[min(range(cfg.p), key=lambda i: losses_now[i])]
                    print(f"  [early-stop] iter {t + 1}: loss {bl:.2e} "
                          f"disp {disp:.2e}", flush=True)
                    break
            else:
                stop_hits = 0

        # --- committee maintenance (alignment/soup/restart) ---
        souped = None
        if cfg.maint_every and (t + 1) % cfg.maint_every == 0 and t >= 5:
            losses = l1_on(pop, X, Y)
            order = sorted(range(cfg.p), key=lambda i: losses[i])
            best_now = pop[order[0]].clone()
            for i in order[1:]:
                align_member_to(best_now, pop[i])
            cand = soup_of([best_now] + [pop[i] for i in order[1:]])
            closs, wloss = l1_on([cand], X, Y)[0], losses[order[-1]]
            if closs < wloss:
                pop[order[-1]] = cand.to(device)
                opts[order[-1]] = torch.optim.Adam(
                    pop[order[-1]].parameters(),
                    lr=opts[order[-1]].param_groups[0]["lr"])
            for k in range(min(cfg.restart_worst, cfg.p - 1)):
                idx = order[-1 - k]
                pop[idx] = MLP(dims).to(device)
                opts[idx] = torch.optim.Adam(
                    pop[idx].parameters(),
                    lr=opts[idx].param_groups[0]["lr"])
            souped = closs

        # --- logging (consensus + combine run FIRST so this iter's max_eps
        #     reflects any injected member) ---
        if (t + 1) % cfg.log_every == 0 or t == cfg.outer - 1:
            losses = l1_on(pop, X, Y)
            # ungated 5/8 — identical to the --fast dump trigger, so the printed
            # `cluster` column and --combine act on the same consensus --fast does
            cnet = build_consensus(pop, dims, quorum_ratio=cfg.cluster_quorum)
            cc = (param_errors(cnet, teacher)["max_eps"]
                  if cnet is not None else None)
            cstats = consensus_neuron_stats(pop, teacher, dims,
                                            quorum_ratio=cfg.cluster_quorum)
            combined_now = None
            if cfg.combine and not combined_done and cnet is not None:
                polished = polish_consensus(
                    cnet, X, Y, lr=cfg.combine_polish_lr,
                    steps=cfg.combine_polish_steps)
                wi = max(range(cfg.p), key=lambda i: losses[i])
                pop[wi] = polished.to(device)
                opts[wi] = torch.optim.Adam(
                    pop[wi].parameters(),
                    lr=opts[wi].param_groups[0]["lr"])
                combined_done = True
                combined_now = t + 1
                pe = param_errors(polished, teacher)["max_eps"]
                pl = l1_on([polished], X, Y)[0]
                print(f"  [combine] iter {t + 1}: replaced worst member {wi} "
                      f"(loss {losses[wi]:.2e}) with polished consensus "
                      f"(max_eps {cc:.2e}->{pe:.2e}, loss {pl:.2e})",
                      flush=True)
                losses = l1_on(pop, X, Y)  # reflect the injected member
            bi = min(range(cfg.p), key=lambda i: losses[i])
            best = pop[bi]
            errs = param_errors(best, teacher)
            rec = {
                "iter": t + 1,
                "queries": (t + 1) * cfg.q,
                "best_loss": losses[bi],
                "med_loss": sorted(losses)[cfg.p // 2],
                "worst_loss": max(losses),
                "max_eps": errs["max_eps"],
                "mean_eps": sum(errs["mean_eps_per_matrix"]) /
                len(errs["mean_eps_per_matrix"]),
                "agree": agreement(best, teacher, teacher_eval_pts),
                "wall_s": round(time.time() - t0, 1),
                "cluster_max_eps": cc,
            }
            if cstats is not None:
                rec["n_consensus"] = cstats["n_consensus"]
                rec["n_total"] = cstats["n_total"]
                rec["consensus_max_eps"] = cstats["max_eps"]
                rec["consensus_mean_eps"] = cstats["mean_eps"]
                for k in ("l0_max", "l0_mean", "l1_max", "l1_mean",
                          "n_out_consensus", "out_total", "out_max", "out_mean"):
                    if k in cstats:
                        rec["consensus_" + k] = cstats[k]
            if souped is not None:
                rec["soup_loss"] = souped
            if combined_now is not None:
                rec["combined_iter"] = combined_now
            log.append(rec)
            cc_str = f"{cc:.2e}" if cc is not None else "  n/a  "
            if cstats is not None and cstats["max_eps"] is not None:
                cons_str = (f"{cstats['n_consensus']}/{cstats['n_total']} "
                            f"L0[max {cstats['l0_max']:.2e} "
                            f"mean {cstats['l0_mean']:.2e}] "
                            f"L1[max {cstats['l1_max']:.2e} "
                            f"mean {cstats['l1_mean']:.2e}]")
                om = (f"max {cstats['out_max']:.2e} mean {cstats['out_mean']:.2e}"
                      if cstats.get("out_max") is not None else "n/a")
                cons_str += (f" | out {cstats['n_out_consensus']}/"
                             f"{cstats['out_total']} [{om}]")
            else:
                cons_str = f"0/{dims[1]}"
            print(f"  it {t + 1:3d} | q {(t + 1) * cfg.q:6d} | "
                  f"loss {losses[bi]:.2e} | max_eps {errs['max_eps']:.2e} | "
                  f"mean_eps {rec['mean_eps']:.2e} | "
                  f"cluster {cc_str} | cons {cons_str} | "
                  f"agree {rec['agree']:.4f} | {rec['wall_s']}s", flush=True)

        # periodic population snapshot (inspect mid-run; no queries, overwrites)
        if (cfg.pop_save_every and cfg.pop_save_path
                and (t + 1) % cfg.pop_save_every == 0):
            torch.save({
                "dims": dims, "iter": t + 1,
                "pop_states": [{k: v.detach().cpu() for k, v in
                                m.state_dict().items()} for m in pop],
                "teacher_state": {k: v.detach().cpu() for k, v in
                                  teacher.state_dict().items()},
            }, cfg.pop_save_path)
            print(f"  [pop-save] iter {t + 1}: {cfg.p} members -> "
                  f"{cfg.pop_save_path}", flush=True)

        # dump population + queries at a target iter (or first consensus), stop
        _hit = ((cfg.dump_at_iter and (t + 1) == cfg.dump_at_iter) or
                (cfg.stop_on_consensus and (t + 1) % cfg.log_every == 0 and
                 build_consensus(pop, dims, quorum_ratio=0.625) is not None))
        if cfg.dump_path and _hit:
            torch.save({
                "dims": dims, "iter": t + 1,
                "pop_states": [{k: v.detach().cpu() for k, v in
                                m.state_dict().items()} for m in pop],
                "teacher_state": {k: v.detach().cpu() for k, v in
                                  teacher.state_dict().items()},
                "X": X.detach().cpu(), "Y": Y.detach().cpu(),
            }, cfg.dump_path)
            print(f"  [dump] iter {t + 1} population + {len(X)} queries "
                  f"-> {cfg.dump_path}", flush=True)
            break

    # --- final selection / averaging / polish ---
    if best is None:
        losses = l1_on(pop, X, Y)
        best = pop[min(range(cfg.p), key=lambda i: losses[i])]
    queries_used = (t + 1) * cfg.q
    extra = {}

    # --- checkpoint the raw reconstruction BEFORE the endgame solvers, so the
    #     LBFGS / last-layer polish can be re-tried offline via polish.py
    #     without re-running the whole query loop. ---
    if save_recon is not None:
        torch.save({
            "dims": dims,
            "best_state": {k: v.detach().cpu()
                           for k, v in best.state_dict().items()},
            "pop_states": [{k: v.detach().cpu()
                            for k, v in m.state_dict().items()} for m in pop],
            "teacher_state": {k: v.detach().cpu()
                              for k, v in teacher.state_dict().items()},
            "X": X.detach().cpu(),
            "Y": Y.detach().cpu(),
            "cfg": asdict(cfg),
            "seed": seed,
            "queries": queries_used,
            "pre_endgame_max_eps": param_errors(best, teacher)["max_eps"],
        }, save_recon)
        print(f"  [save] reconstruction checkpoint -> {save_recon}", flush=True)

    if cfg.lastlayer_every:
        solve_last_layer_(best, X, Y)
    if cfg.popavg_kappa > 0:
        losses = l1_on(pop, X, Y)
        bi = min(range(cfg.p), key=lambda i: losses[i])
        avg, k = aligned_pop_average(pop, losses, cfg.popavg_kappa)
        if avg is not None:
            avg_loss = l1_on([avg], X, Y)[0]
            extra["popavg"] = {
                "k": k,
                "avg_loss": avg_loss,
                "best_loss": losses[bi],
                "avg_max_eps": param_errors(avg, teacher)["max_eps"],
                "best_max_eps": param_errors(pop[bi], teacher)["max_eps"],
            }
            # observable selection rule: lower training loss wins
            best = avg if avg_loss <= losses[bi] else pop[bi]
        else:
            best = pop[bi]
            extra["popavg"] = {"k": k}
    if cfg.polish_f64:
        best = polish_f64(best, X, Y, cfg, gen)
    if cfg.lbfgs_polish:
        best = polish_lbfgs(best, X, Y, cfg)

    errs = param_errors(best, teacher)
    final = {
        "final_max_eps": errs["max_eps"],
        "final_mean_eps": sum(errs["mean_eps_per_matrix"]) /
        len(errs["mean_eps_per_matrix"]),
        "final_max_eps_per_matrix": errs["max_eps_per_matrix"],
        "final_agree": agreement(best, teacher, teacher_eval_pts),
        "queries": queries_used,
        "wall_s": round(time.time() - t0, 1),
        **extra,
    }
    return best, log, final


def polish_f64(best, X, Y, cfg, gen):
    """Last-mile refinement in float64 on CPU (MPS lacks float64)."""
    dev = next(best.parameters()).device
    net = best.clone().cpu().double()
    Xc, Yc = X.detach().cpu().double(), Y.detach().cpu().double()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    n = len(Xc)
    for ep in range(cfg.polish_epochs):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad()
            loss = (net(Xc[idx]) - Yc[idx]).abs().mean()
            loss.backward()
            opt.step()
    return net.float().to(dev)


def polish_lbfgs(best, X, Y, cfg, max_samples=12000, steps=60):
    """float64 LBFGS endgame with squared loss (Opus A2 + A3-lite):
    residual-proportional gradients + curvature, at float64 precision."""
    dev = next(best.parameters()).device
    net = best.clone().cpu().double()
    # solver window: restrict to the last cfg.solverwindow outer iters of queries
    # (the most-recent tail), then cap to max_samples for float64 tractability.
    # solverwindow=0 keeps the original behavior (random subsample of all).
    Xw, Yw = X, Y
    if cfg.solverwindow and cfg.solverwindow > 0:
        keep = cfg.solverwindow * cfg.q
        if keep < len(X):
            Xw, Yw = X[-keep:], Y[-keep:]
    n = len(Xw)
    if n > max_samples:
        idx = torch.randperm(n)[:max_samples]
        Xs, Ys = Xw[idx].cpu().double(), Yw[idx].cpu().double()
    else:
        Xs, Ys = Xw.cpu().double(), Yw.cpu().double()
    opt = torch.optim.LBFGS(net.parameters(), lr=0.5, max_iter=20,
                            history_size=10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ((net(Xs) - Ys) ** 2).mean()
        loss.backward()
        return loss

    for _ in range(steps):
        opt.step(closure)
    return net.float().to(dev)
