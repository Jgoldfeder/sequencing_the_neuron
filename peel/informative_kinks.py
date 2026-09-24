"""Experimental forward-query recovery with reconstruction-designed samples.

No teacher weights, activations or derivatives are used. Isolation is a soft
preference: noisy guesses cannot certify teacher isolation, so every selected
bracket is still scanned, fingerprinted and verified by kink_solve.locate.
"""
import copy
import math
import time
import numpy as np
import torch
from scipy.linalg import qr
from threadpoolctl import threadpool_limits
import kink_solve as K
from verify_layer1 import _Oracle


@torch.no_grad()
def candidate_pool(guess, l, channels, gen, pool, scales, eps):
    dev = guess.layers[0].weight.device
    nscale = pool // len(scales)
    per = nscale * len(scales)
    j = torch.tensor(channels, device=dev).repeat_interleave(per)
    scale = torch.tensor(scales, device=dev, dtype=torch.float64).repeat_interleave(nscale).repeat(len(channels))
    erow = torch.tensor([eps[c] for c in channels], device=dev, dtype=torch.float64).repeat_interleave(per)
    # Only the *guessed* surface is needed for candidate generation. Newton
    # plus a residual check avoids 50 bisection evaluations of a noisy model;
    # the actual teacher kink is subsequently located by the verified scan.
    d = guess.layers[0].weight.shape[1]
    X = scale[:, None] * torch.randn(len(j), d, device=dev,
                                     dtype=torch.float64, generator=gen)
    wg = guess.layers[l].weight[j]; bg = guess.layers[l].bias[j]
    for _ in range(12):
        g, normal = K._g_and_normal(guess, X, l, wg, bg)
        X -= (g / normal.square().sum(1).clamp_min(1e-30))[:, None]*normal
    g, normal = K._g_and_normal(guess, X, l, wg, bg)
    norm = normal.norm(dim=1)
    H = K._phi(guess, X, l)
    keep = (norm > 1e-8) & (g.abs() < 1e-10*(1+H.norm(dim=1)*wg.norm(dim=1)+bg.abs()))
    U = normal/norm.clamp_min(1e-30)[:, None]
    R = 3*erow*(H.norm(dim=1)+1)/(H.shape[1]**.5)/norm.clamp_min(1e-30)
    X, U, R, H, j = (v[keep] for v in (X, U, R, H, j))
    zm = K._preacts(guess, X-R[:, None]*U, l)
    zp = K._preacts(guess, X+R[:, None]*U, l)
    crossings = torch.zeros(len(X), device=dev)
    for i, (a, b) in enumerate(zip(zm, zp)):
        cross = a*b < 0
        if i == l:
            cross[torch.arange(len(j), device=dev), j] = False
        crossings += cross.sum(1)
    return {c: tuple(v[j == c] for v in (X, U, R, H, crossings)) for c in channels}


@torch.no_grad()
def select_candidates(guess, l, c, gen, history, pool=768, batch=96,
                      scales=(1., 4., 16.), eps=.02, selection='design', candidates=None):
    if candidates is None:
        candidates = candidate_pool(guess, l, [c], gen, pool, scales, {c: eps})[c]
    X, U, R, H, crossings = candidates
    if not len(X):
        return X, U, R, 0, 0
    isolated = crossings == 0
    n = min(batch, len(X))
    if selection == 'random':
        idx = torch.randperm(len(X), device=X.device, generator=gen)[:n]
    else:
        # Pin the guessed normal's largest coordinate; the remaining augmented
        # coordinates span the hyperplane. Whiten against accumulated samples
        # to prioritize directions not yet measured. SVD avoids normal equations.
        k = int(guess.layers[l].weight[c].abs().argmax())
        A = torch.cat([H, torch.ones_like(H[:, :1])], 1).cpu().numpy()
        A = np.delete(A, k, axis=1)
        if history is None or len(history) < A.shape[1]:
            design = A
        else:
            h = history.cpu().numpy()
            design = np.delete(np.column_stack([h, np.ones(len(h))]), k, axis=1)
        _, s, vh = np.linalg.svd(design, full_matrices=False)
        floor = max(s[0]*1e-8, 1e-14)
        B = (A @ vh.T) / np.maximum(s, floor)
        # Prefer model-isolated brackets without discarding all crowded regions.
        quality = 1 / np.sqrt(1 + .25*crossings.cpu().numpy())
        if selection == 'isolated':
            quality = isolated.cpu().numpy().astype(float)
            n = min(n, int(quality.sum()))
        _, _, piv = qr((B*quality[:, None]).T, pivoting=True, mode='economic',
                       check_finite=False)
        idx = torch.as_tensor(piv[:n].copy(), device=X.device, dtype=torch.long)
    return X[idx], U[idx], R[idx], int(isolated.sum()), len(X)


# Small per-neuron SVD/QR fits become much slower with large BLAS pools.
# Scope the limit to refinement; restore training thread settings on exit.
@torch.no_grad()
@threadpool_limits.wrap(limits=1)
def recover_layer(teacher, cons, frontier, device, only_channels=None, gen=None,
                  verbose=True, need=400, pool=768, batch=96, max_rounds=None,
                  scales=(1.,4.,16.), selection='design', diagnostics=None,
                  angle_gate=12., retry_strategy="adaptive", failure_dir=None, **kwargs):
    """Experimental drop-in recovery; refined means passed heuristic checks,
    NOT a certified parameter error bound when the prefix is approximate.
    """
    if retry_strategy not in {"adaptive", "legacy"}:
        raise ValueError("retry_strategy must be adaptive or legacy")
    if failure_dir is not None and diagnostics is None:
        diagnostics = {}
    output_dtype = cons.layers[frontier].weight.dtype
    cons = copy.deepcopy(cons).double().to(device).eval()
    if isinstance(teacher, torch.nn.Module):
        teacher = copy.deepcopy(teacher).double().to(device).eval()
    if selection not in {'design', 'random', 'isolated'}:

        raise ValueError(f'Unknown selection mode: {selection}')
    if not scales or min(scales) <= 0 or pool < len(scales) or batch < 1:
        raise ValueError('Positive scales, pool >= number of scales, and batch >= 1 required')
    l = frontier
    minimum_fit = max(cons.layers[l].weight.shape[1]+48,
                      math.ceil((cons.layers[l].weight.shape[1]+16)*7/6))
    need = max(need, minimum_fit)
    # A fixed 24 rounds capped each neuron at 2304 attempted brackets and
    # prematurely rejected low-yield deep neurons. Budget for a 10% yield,
    # with extra room for rare deep crossings; finished neurons leave todo.
    if max_rounds is None:
        available = min(batch, (pool // len(scales))*len(scales))
        max_rounds = max(160, math.ceil(10*need/available))
    if max_rounds < 0:
        raise ValueError('max_rounds must be nonnegative or None')
    W = cons.layers[l].weight.detach().clone()
    b = cons.layers[l].bias.detach().clone()
    channels = list(range(len(W))) if only_channels is None else list(only_channels)
    if not channels:
        return W.to(output_dtype), b.to(output_dtype), torch.zeros(len(W), dtype=torch.bool, device=device), 0
    gen = gen or torch.Generator(device=device).manual_seed(0)
    initial_rng = gen.get_state().clone()
    oracle = _Oracle(teacher)
    Hs = {c: [] for c in channels}
    count = {c: 0 for c in channels}
    eps = {c: .02 for c in channels}
    # A low yield can mean either a missed surface OR an overcrowded bracket.
    # Explore narrower as well as wider brackets, rather than permanently
    # ratcheting to .16. Keep successful scales, periodically revisit alternatives.
    widths = (.02, .01, .005, .0025, .04, .08, .16)
    trial = {c: 0 for c in channels}
    history_rounds = {c: [] for c in channels}
    isolated_count = candidate_count = 0
    start = time.time()
    for rnd in range(max_rounds):
        todo = [c for c in channels if count[c] < need]
        if not todo:
            break
        for offset in range(0, len(todo), 8):
            group = todo[offset:offset+8]
            pools = candidate_pool(cons, l, group, gen, pool, scales, eps)
            batches = []
            for c in group:
                history = torch.cat(Hs[c]) if Hs[c] else None
                X, U, R, isolated, nc = select_candidates(
                    cons, l, c, gen, history, pool, batch, scales, eps[c], selection,
                    candidates=pools[c])
                isolated_count += isolated; candidate_count += nc
                if len(X):
                    batches.append((c, X, U, R))
            if not batches:
                continue
            X = torch.cat([v[1] for v in batches])
            U = torch.cat([v[2] for v in batches])
            R = torch.cat([v[3] for v in batches])
            js = torch.cat([torch.full((len(v[1]),),v[0], device=device, dtype=torch.long) for v in batches])
            debug = [] if diagnostics is not None else None
            Xs, ok = K.locate(oracle, X, U, R, gen, cons, l, W[js], K=25+12*l, debug=debug)
            H = K._phi(cons, Xs, l)
            for c, _, _, _ in batches:
                selected = js == c
                good = selected & ok
                if good.any():
                    Hs[c].append(H[good]); count[c] += int(good.sum())
                gained, attempted = int(good.sum()), int(selected.sum())
                entry = dict(round=rnd, eps=eps[c], attempted=attempted, found=gained)
                if debug is not None:
                    for name in ("single", "straddle", "good"):
                        entry[name] = sum(int((d[name] & (js[d["sub"]] == c)).sum()) for d in debug)
                history_rounds[c].append(entry)
                if retry_strategy == "legacy":
                    if gained < .1*attempted:
                        eps[c] = min(.16, 2*eps[c])
                elif gained < .1*attempted or (rnd+1) % 8 == 0:
                    trial[c] += 1
                    eps[c] = widths[trial[c] % len(widths)]
        if verbose:
            print(f'[design] L{l} round={rnd} done={sum(count[c]>=need for c in channels)}/{len(channels)} '
                  f'min_points={min(count.values())} queries={oracle.n} elapsed={time.time()-start:.1f}s', flush=True)
    refined = torch.zeros(len(W), dtype=torch.bool, device=device)
    for c in channels:
        if count[c] < minimum_fit:
            if diagnostics is not None:
                diagnostics[c] = dict(points=count[c], accepted=False, reason='too_few_points',
                                      round_budget=max_rounds)
            if verbose:
                print(f'[design] reject c={c} only {count[c]}/{need} points', flush=True)
            continue
        if verbose and count[c] < need:
            print(f'[design] c={c}: fitting {count[c]}/{need} target points '
                  f'(minimum {minimum_fit}); full validation still required', flush=True)
        H = torch.cat(Hs[c])
        # Interleave held-out points across independent candidate rounds.
        hold = torch.arange(len(H), device=device) % 7 == 0
        train, valid = H[~hold], H[hold]
        w, bias, kept, gap = K.solve_from_points(train, W[c])
        r = (train@w+bias).abs()
        rv = (valid@w+bias).abs()
        threshold = max(100*r.median().item(), 1e-12)
        inlier = r < threshold
        if int(inlier.sum()) < W.shape[1]+8:
            if diagnostics is not None:
                diagnostics[c] = dict(points=len(H), accepted=False, reason='too_few_inliers')
            continue
        A = torch.cat([train[inlier], torch.ones_like(train[inlier, :1])], 1).cpu()
        s = torch.linalg.svdvals(A)
        cond = (s[0]/s[-2]).item()
        uncertainty = (r.median().item()+torch.finfo(H.dtype).eps*train.norm(dim=1).median().item())*math.sqrt(len(train))/s[-2].item()
        gv = torch.cat([W[c], b[c:c+1]]); v = torch.cat([w, bias.reshape(1)])
        cosine = abs(float((gv @ v) / (gv.norm()*v.norm())))
        angle = math.degrees(math.acos(min(1., cosine)))
        valid_fraction = float((rv < threshold).double().mean())
        # gap/uncertainty floors: a wrong lock gives gap ~1e-2; a correct solve gives
        # gap ~ the prefix error (1e-13 exact prefix, ~1e-6 for a 1e-8 prefix as in
        # the real pipeline), so the thresholds are set between, NOT at the exact-
        # prefix floor (1e-7 / 1e-9 rejected every channel of the real checkpoint).
        ok = (int(inlier.sum()) >= W.shape[1]+8 and valid_fraction >= .9
              and angle <= angle_gate and gap < 1e-4 and uncertainty < 1e-5)
        if diagnostics is not None:
            diagnostics[c] = dict(points=len(H), round_budget=max_rounds, cond=cond, gap=gap,
                                  uncertainty=uncertainty, validation_fraction=valid_fraction,
                                  residual=r.median().item(), accepted=ok)
        if ok:
            W[c] = w; b[c] = bias; refined[c] = True
        elif verbose:
            print(f'[design] reject c={c} gap={gap:.2e} cond={cond:.2e} '
                  f'uncertainty={uncertainty:.2e} validation={valid_fraction:.2f}', flush=True)
    if diagnostics is not None:
        for c in channels:
            diagnostics[c]["sampling_rounds"] = history_rounds[c]
    if verbose:
        print(f'[design] L{l} accepted={int(refined.sum())}/{len(channels)} '
              f'isolated_candidates={isolated_count}/{candidate_count} queries={oracle.n} '
              f'time={time.time()-start:.1f}s', flush=True)
    if failure_dir is not None and any(not bool(refined[c]) for c in channels):
        # Diagnostic replay only; teacher parameters never enter the solve.
        from pathlib import Path
        directory = Path(failure_dir); directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"layer{l}_{time.time_ns()}.pt"
        torch.save(dict(frontier=l, dims=cons.dims, act=getattr(cons, "act_name", "leaky_relu"),
                        student={k:v.detach().cpu() for k,v in cons.state_dict().items()},
                        teacher=({k:v.detach().cpu() for k,v in teacher.state_dict().items()}
                                 if isinstance(teacher, torch.nn.Module) else None),
                        channels=channels, failed=[c for c in channels if not bool(refined[c])],
                        rng=initial_rng.cpu(), diagnostics=diagnostics,
                        options=dict(need=need, pool=pool, batch=batch, max_rounds=max_rounds,
                                     scales=scales, selection=selection, angle_gate=angle_gate,
                                     retry_strategy=retry_strategy)), path)
        print(f"[design] incomplete recovery: replay snapshot -> {path}", flush=True)
    return W.to(output_dtype), b.to(output_dtype), refined, oracle.n
