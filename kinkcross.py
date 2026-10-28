"""Model-free kink-crossing query sampler (--random-queries kinkcross).

A (leaky) ReLU black box is piecewise linear along any line, so its kinks along a line can be found from forward
queries alone: query a fine grid on the line, and a kink sits wherever the second difference of the outputs is not
(numerically) zero. Every kink found then gets extra queries on BOTH sides of it, at a few points spread around the
crossing (small perpendicular jitter), so each found kink is crossed many times at different places of its surface.

Exploration (Rolnick & Kording 2020): kink surfaces intersect, so new lines are drawn THROUGH previously found kink
points (in fresh random directions) as well as through fresh random base points. Walking outward from found kinks
reaches the neighbouring surfaces, including those of units that are rarely active under random inputs. Found kink
points persist across iterations (a module-level pool, reset at the start of each run).

No committee, no guess: the only information used is the black box's outputs. EVERY point sent to the black box
(grid points and kink-straddling points) is returned as a query and counted once in the budget.
"""
import torch

_POOL = {"x": None}            # found kink points (CPU, float32), shared across iterations of one run


def reset():
    _POOL["x"] = None


@torch.no_grad()
def kinkcross_queries(teacher, n, d, device, gen, base_std=0.5, grid=512, span=2.0, pairs=4, explore=0.5,
                      lines_per_batch=8, tol=1e-3, pool_cap=50000, out_device=None):
    """n queries (n, d) on out_device (default `device`): grid points on random lines plus pairs of points
    straddling every kink found on them. Lines: x(t) = b + t * R * u, t in [-span, span], u a random unit
    direction, R = base_std * sqrt(d) (the typical norm of a base point); b is a fresh N(0, base_std^2) point or,
    with probability `explore`, a previously found kink point."""
    out_device = out_device or device
    out = torch.empty(n, d, device=out_device)
    filled = 0
    R = base_std * d ** 0.5
    ts = torch.linspace(-span, span, grid, device=device)
    h = float(ts[1] - ts[0])                                   # grid step in t
    while filled < n:
        L = lines_per_batch
        base = torch.randn(L, d, generator=gen, device=device) * base_std
        pool = _POOL["x"]
        if pool is not None and len(pool) > 0:
            use = torch.rand(L, generator=gen, device=device) < explore
            if bool(use.any()):
                idx = torch.randint(len(pool), (int(use.sum()),), generator=gen, device=device)
                base[use] = pool[idx.cpu()].to(device)
        u = torch.randn(L, d, generator=gen, device=device)
        u = u / u.norm(dim=1, keepdim=True)
        X = base[:, None, :] + (ts[None, :, None] * R) * u[:, None, :]          # (L, grid, d)
        Y = teacher(X.reshape(L * grid, d)).reshape(L, grid, -1).float()
        # kinks: second difference of the outputs along each line, relative to the typical first difference
        D1 = (Y[:, 1:] - Y[:, :-1]).norm(dim=2)                                 # (L, grid-1)
        D2 = (Y[:, 2:] - 2 * Y[:, 1:-1] + Y[:, :-2]).norm(dim=2)                # (L, grid-2), centred at 1..grid-2
        scale = D1.median(dim=1, keepdim=True).values.clamp_min(1e-12)
        flag = D2 > tol * scale                                                 # a kink lies in (t_{i-1}, t_{i+1})
        # one kink estimate per run of consecutive flagged indices: the run's centre
        new_pts, straddle = [], []
        for li in range(L):
            f = flag[li].nonzero().flatten().tolist()
            groups, cur = [], []
            for i in f:
                if cur and i != cur[-1] + 1:
                    groups.append(cur); cur = []
                cur.append(i)
            if cur:
                groups.append(cur)
            if not groups:
                continue
            tk = torch.tensor([ts[g[0] + 1].item() * 0.5 + ts[g[-1] + 1].item() * 0.5 for g in groups], device=device)
            half = torch.tensor([(len(g) + 1) * h for g in groups], device=device)   # the kink is within +-half of tk
            kx = base[li] + (tk[:, None] * R) * u[li]                                 # kink estimates (K, d)
            new_pts.append(kx)
            # `pairs` pairs per kink: t* -+ (half + jittered extra), each point jittered perpendicular to the line
            K = len(groups)
            off = (half[:, None] * (1.0 + torch.rand(K, pairs, generator=gen, device=device)))   # (K, pairs) in t
            for sgn in (-1.0, 1.0):
                p = kx[:, None, :] + (sgn * off[:, :, None] * R) * u[li]                # (K, pairs, d)
                jit = torch.randn(K, pairs, d, generator=gen, device=device)
                jit = jit - (jit * u[li]).sum(-1, keepdim=True) * u[li]                  # perpendicular to the line
                jit = jit / jit.norm(dim=-1, keepdim=True) * (off[:, :, None] * R * 0.5)
                straddle.append((p + jit).reshape(-1, d))
        # collect this batch's queries: grid points first, then straddles; truncate to the remaining budget
        batch = [X.reshape(L * grid, d)] + straddle
        for b in batch:
            take = min(len(b), n - filled)
            if take <= 0:
                break
            out[filled:filled + take].copy_(b[:take]); filled += take
        if new_pts:
            kp = torch.cat(new_pts).float().cpu()
            _POOL["x"] = kp if _POOL["x"] is None else torch.cat([_POOL["x"], kp])[-pool_cap:]
    return out
