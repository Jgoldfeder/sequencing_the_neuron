"""Informed query design WITHOUT disagreement (--query-design leverage).

Motivation (query analysis, 3072x200x200x200x100 leaky ReLU): disagreement queries win not by covering kinks better
but by exciting every layer's INPUT evenly -- rare units switched on 2-6x more often, and h2's covariance 17x better
conditioned than under Gaussian queries. This sampler targets that directly with ONE model (the committee's best
member), no committee disagreement:

    score(x) = sum_l  a_{l-1}(x)^T A_l^{-1} a_{l-1}(x) / dim(a_{l-1})

a_{l-1} = [h_{l-1}(x), 1] is layer l's (augmented) input under the guess (h_0 = x), and A_l its second moment over
the queries already collected (the input factor of the K-FAC Fisher). A query scores high when it moves some layer's
input along directions the buffer has barely explored; a rarely active unit has ~no variance in the next layer's input,
so the score pushes it on. (The K-FAC gradient factor is piecewise constant in x for leaky ReLU -> no gradient; omitted.)
Each term is normalised by its dimension, so a typical buffer point scores ~1 per layer.

Search: like the disagreement search -- start from the variant's init, `steps` Adam steps on x maximising the score,
each step projected into a ball of radius `radius` (no projection = the score grows without bound with |x|).
"""
import torch


@torch.no_grad()
def _moments(member, Xb, device, max_rows=20000, gen=None):
    """(sums of a a^T per layer, count) over (a subsample of) the points Xb, under `member`"""
    if len(Xb) > max_rows:
        idx = torch.randperm(len(Xb), generator=gen)[:max_rows]
        Xb = Xb[idx]
    acc = None; n = 0
    for i in range(0, len(Xb), 4096):
        h = Xb[i:i + 4096].to(device).float()
        ins = []
        for L in member.layers:
            ins.append(torch.cat([h, torch.ones(len(h), 1, device=device)], 1))
            h = member.act(L(h)) if L is not member.layers[-1] else L(h)
        if acc is None:
            acc = [torch.zeros(a.shape[1], a.shape[1], device=device, dtype=torch.float64) for a in ins]
        for k, a in enumerate(ins):
            acc[k] += a.double().T @ a.double()
        n += len(h)
    return acc, n


def _invert(acc, n, ridge=1e-3):
    """per layer: (V, 1/lambda) of the ridged second moment, so a^T A^-1 a = sum_i (v_i.a)^2 / lambda_i"""
    inv = []
    for M in acc:
        M = M / n
        M = M + ridge * (M.trace() / M.shape[0]) * torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
        ev, V = torch.linalg.eigh(M)
        inv.append((V.float(), (1.0 / ev).float()))
    return inv


def _score(member, x, Ainv, skip_input=False, topk=0):
    """topk = 0: per layer, the whole whitened norm a^T A^-1 a / dim (averages over all directions).
    topk = k: per layer, the mean of the k LARGEST whitened components (v_i.a)^2 / lambda_i -- one under-explored
    direction (e.g. a single rare unit) can then dominate a query's score instead of being averaged away."""
    h = x; s = 0.0
    for k, (L, (V, il)) in enumerate(zip(member.layers, Ainv)):
        a = torch.cat([h, torch.ones(len(h), 1, device=h.device)], 1)
        if not (skip_input and k == 0):              # skip_input: drop the raw-input (layer-1) term
            c = (a @ V) ** 2 * il
            s = s + (c.topk(min(topk, c.shape[1]), dim=1).values.mean(1) if topk > 0 else c.sum(1) / a.shape[1])
        h = member.act(L(h)) if L is not member.layers[-1] else L(h)
    return s


def leverage_queries(members, Xbuf, n, d, device, gen, init="uniform", init_scale=1.0, steps=30, lr=0.1,
                     radius=None, chunk=8192, verbose=False, seq=False, max_rows=20000, skip_input=False,
                     init_mean=None, init_cov_sqrt=None, topk=0, seeds=None, seed_noise=0.0):
    """n queries (n, d) on `device`, each optimised to maximise the leverage score. `members`: one net, or a list
    whose scores are averaged (robust to a poor single guess; still no disagreement term). seq=True: batch design --
    the batch is built in chunks and each chunk's optimised points are added to the moments before the next chunk,
    so later chunks go after the directions earlier ones left unexplored (instead of all piling into the same ones)."""
    members = list(members) if isinstance(members, (list, tuple)) else [members]
    for m_ in members:
        for p in m_.parameters():
            p.requires_grad_(False)
    _g = torch.Generator().manual_seed(int(torch.randint(1 << 30, (1,), generator=gen, device=device)))
    mom = [_moments(m_, Xbuf, device, max_rows=max_rows, gen=_g) for m_ in members]
    # buffer subsample size vs batch: weight the batch's own points as if the subsample were the full buffer
    wbuf = max(1.0, len(Xbuf) / max(1, mom[0][1]))
    Ainvs = [_invert(a, k) for a, k in mom]
    if radius is None:
        radius = 1.4 * 0.5 * d ** 0.5                      # ~ where the disagreement search's queries end up
    out = torch.empty(n, d, device=device)
    s0 = s1 = 0.0
    i = 0
    while i < n:
        m = min(chunk, n - i)
        if seeds is not None and i < len(seeds):         # residual seeds: stored queries the model gets most wrong
            m = min(m, len(seeds) - i)
            x = seeds[i:i + m].to(device).float() + seed_noise * torch.randn(m, d, generator=gen, device=device)
        elif init == "datacov" and init_cov_sqrt is not None:
            x = torch.randn(m, d, generator=gen, device=device) @ init_cov_sqrt.T + init_mean
        elif init == "uniform":
            x = (torch.rand(m, d, generator=gen, device=device) * 2 - 1) * init_scale
        else:
            x = torch.randn(m, d, generator=gen, device=device) * init_scale
        x.requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        for k in range(steps):
            opt.zero_grad()
            sc = sum(_score(m_, x, Ai, skip_input, topk) for m_, Ai in zip(members, Ainvs)) / len(members)
            if k == 0:
                s0 += float(sc.detach().sum())
            (-sc.sum()).backward()
            opt.step()
            with torch.no_grad():                            # project into the ball
                nr = x.norm(dim=1, keepdim=True)
                x.mul_(torch.clamp(radius / nr.clamp_min(1e-12), max=1.0))
        with torch.no_grad():
            s1 += float((sum(_score(m_, x, Ai, skip_input, topk) for m_, Ai in zip(members, Ainvs)) / len(members)).sum())
        out[i:i + m] = x.detach()
        i_next = i + m
        if seq and i + m < n:                                # batch design: fold this chunk into the moments
            for k_, m_ in enumerate(members):
                a_c, n_c = _moments(m_, x.detach(), device, max_rows=len(x))
                acc, nb = mom[k_]
                mom[k_] = ([A + B / wbuf for A, B in zip(acc, a_c)], nb + n_c / wbuf)
            Ainvs = [_invert(a, k) for a, k in mom]
        i = i_next
    if verbose:
        print(f"  [leverage] {n} queries, {len(members)} member(s){', batch design' if seq else ''}: mean score "
              f"{s0 / n:.2f} -> {s1 / n:.2f} (~{len(Ainvs[0])} per typical buffer point), radius {radius:.1f}", flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# --query-design kink: kink-coverage design (one model, no disagreement).
# Query analysis (16k/iter, data-cov start): leverage pushed pre-activations AWAY from the kinks (3-6x fewer near-kink
# queries than disagreement in layers 2-3) and escaped along junk low-variance input directions (34x the data spread,
# every query at the norm cap). Disagreement concentrates queries NEAR kinks while staying near the data's spread.
# This design does that explicitly:
#   score(x) = sum_layers mean_j  w_j / (1 + (z_j(x) / (tau * s_j))^2)     (Cauchy: a Gaussian kernel at tau 0.05 has
#                                                                       no gradient ~20 bands away from the kink)
# z_j = the best member's pre-activation of hidden unit j, s_j its spread over the buffer, w_j = 1 / (coverage_j + eps)
# with coverage_j = the fraction of stored queries within tau*s_j of the kink on its SCARCER side (normalised to mean 1
# per layer) -> under-covered units, typically the rare ones, dominate. Coverage is updated chunk by chunk (batch
# design). The search runs in the data's whitened coordinates u (x = mean + u S, S S^T = cov + ridge) inside the
# ball |u| <= wradius (disagreement's queries: whitened norm median 21, 10-90% 17-28).
# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def _hidden_z(member, X, device):
    out = [[] for _ in member.layers[:-1]]
    for i in range(0, len(X), 8192):
        h = X[i:i + 8192].to(device).float()
        for l, L in enumerate(member.layers[:-1]):
            z = L(h); out[l].append(z); h = member.act(z)
    return [torch.cat(z) for z in out]


def kink_queries(member, Xbuf, n, d, device, gen, mean, S, wradius=25.0, tau=0.05, steps=30, lr=0.1, chunk=2048,
                 eps=1e-3, max_rows=20000, verbose=False, mix=0.0, seeds=None, seed_noise=0.1):
    """member: one net or a list (the score is summed over them -- their differing versions of a wrongly-learned unit put
    kinks in several places, one of which may be the true one; no disagreement term). mix: fraction of the n queries
    left as plain data-covariance draws (no search)."""
    members = list(member) if isinstance(member, (list, tuple)) else [member]
    for m_ in members:
        for p in m_.parameters():
            p.requires_grad_(False)
    _g = torch.Generator().manual_seed(int(torch.randint(1 << 30, (1,), generator=gen, device=device)))
    Xs = Xbuf[torch.randperm(len(Xbuf), generator=_g)[:max_rows]] if len(Xbuf) > max_rows else Xbuf
    nb = len(Xs)
    # per member, per layer: spread and near-kink counts on each side (layers of all members concatenated as units)
    Zb = [z for m_ in members for z in _hidden_z(m_, Xs, device)]
    nl = len(members[0].layers) - 1
    s = [z.std(0).clamp_min(1e-6) for z in Zb]
    near_pos = [((z.abs() < tau * sl) & (z > 0)).float().sum(0) for z, sl in zip(Zb, s)]
    near_neg = [((z.abs() < tau * sl) & (z <= 0)).float().sum(0) for z, sl in zip(Zb, s)]
    n_mix = int(mix * n)
    wbuf = max(1.0, len(Xbuf) / nb)                      # the batch's points count as if the subsample were the buffer
    mean = mean.to(device).float(); S = S.to(device).float()
    out = torch.empty(n, d, device=device)
    sc0 = sc1 = 0.0
    if n_mix:                                            # plain data-covariance draws, no search
        out[:n_mix] = mean + torch.randn(n_mix, d, generator=gen, device=device) @ S.T
    for i in range(n_mix, n, chunk):
        m = min(chunk, n - i)
        cov = [torch.minimum(p_, q_) / nb for p_, q_ in zip(near_pos, near_neg)]
        w = [1.0 / (c + eps) for c in cov]
        w = [wl / wl.mean() for wl in w]
        u = torch.randn(m, d, generator=gen, device=device)
        if seeds is not None and i - n_mix < len(seeds):   # start from given points (e.g. worst-fit stored queries)
            k_ = min(m, len(seeds) - (i - n_mix))
            sx = seeds[i - n_mix:i - n_mix + k_].to(device).float()
            u[:k_] = torch.linalg.solve(S, (sx - mean).T).T + seed_noise * u[:k_]   # whitened coords of the seeds
        u.requires_grad_(True)
        opt = torch.optim.Adam([u], lr=lr)
        for k in range(steps):
            opt.zero_grad()
            x_ = mean + u @ S.T; sc = 0.0
            for mi, m_ in enumerate(members):
                h = x_
                for l, L in enumerate(m_.layers[:-1]):
                    k_ = mi * nl + l
                    z = L(h); h = m_.act(z)
                    sc = sc + (w[k_] / (1.0 + (z / (tau * s[k_])) ** 2)).mean(1)   # Cauchy kernel: peak at the kink, gradient everywhere
            if k == 0:
                sc0 += float(sc.detach().sum())
            (-sc.sum()).backward()
            opt.step()
            with torch.no_grad():
                nr = u.norm(dim=1, keepdim=True)
                u.mul_(torch.clamp(wradius / nr.clamp_min(1e-12), max=1.0))
        with torch.no_grad():
            x = mean + u.detach() @ S.T
            out[i:i + m] = x
            sc1 += float(sc.detach().sum())
            zs = [z for m_ in members for z in _hidden_z(m_, x, device)]
            for l, z in enumerate(zs):                                # batch design: update the coverage counts
                near = z.abs() < tau * s[l]
                near_pos[l] += ((near & (z > 0)).float().sum(0)) / wbuf
                near_neg[l] += ((near & (z <= 0)).float().sum(0)) / wbuf
            nb_eff = nb + m / wbuf
            nb = nb_eff
    if verbose:
        print(f"  [kink-design] {n} queries ({n_mix} plain draws), {len(members)} member(s): mean score "
              f"{sc0 / max(1, n - n_mix):.2f} -> {sc1 / max(1, n - n_mix):.2f}, whitened radius {wradius:g}, tau {tau:g}", flush=True)
    return out
