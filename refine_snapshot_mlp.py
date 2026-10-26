"""refine_snapshot_mlp.py -- exact recovery of a (leaky) ReLU MLP from a saved committee snapshot.

    python refine_snapshot_mlp.py <pop snapshot .pt> [--out out.pt]

Same refiner as `run.py --fast --design-refine` (method._mlp_refine_layer, design sampling, layer by layer, rows
scale-matched to the guess, closed-form output layer), run on the best member of a saved population.  One addition:
a unit the refiner cannot solve from the best member's row is retried from the committee's other copies -- rows of
any member that match none of the layer's refined units, clustered across members, most-supported first (the
best-output member can hold a wrong copy of a unit that the other members learned).  The black box is queried by
forward evaluation only; teacher parameters are read only for the printed error scores.
"""
import argparse, copy, time, torch
from nets import MLP
from method import param_errors, _mlp_refine_layer
from align import align_clone_to, scale_normalize_

ap = argparse.ArgumentParser()
ap.add_argument("snapshot"); ap.add_argument("--out", default=""); ap.add_argument("--device", default="cuda")
ap.add_argument("--head-points", type=int, default=20000)
ap.add_argument("--unit-report", action="store_true")
ap.add_argument("--resume", default="", help="saved output of an earlier run: take its layers < --start-layer as refined")
ap.add_argument("--start-layer", type=int, default=1, help="1-indexed hidden layer to start refining at (with --resume)")
ap.add_argument("--gf-lines", type=int, default=20)
ap.add_argument("--track-step", type=float, default=2.0); ap.add_argument("--track-need-mult", type=int, default=2)
ap.add_argument("--stop-after", type=int, default=0)
ap.add_argument("--local-refine", action="store_true", help="bounded independent eight-query kink brackets before tracking/design")
ap.add_argument("--local-rounds", type=int, default=24)
ap.add_argument("--fallback-rounds", type=int, default=16, help="design rounds per fallback when --local-refine is enabled")
ap.add_argument("--member", default="consensus", help="guess member: 'consensus' (teacher-free vote) or an index")
a = ap.parse_args(); dev = a.device
d = torch.load(a.snapshot, map_location=dev, weights_only=False)
dims, act = d["dims"], d["act"]
T = MLP(dims, act=act).to(dev); T.load_state_dict(d["teacher_state"]); T = T.double()
for p in T.parameters():
    p.requires_grad_(False)
mems = []
for s in d["pop_states"]:
    m = MLP(dims, act=act).to(dev); m.load_state_dict(s); mems.append(m.double())
X = torch.randn(4096, dims[0], device=dev, dtype=torch.float64) * 0.5
# guess member, teacher-free: the one whose hidden rows the rest of the committee reproduces most often (cos > 0.999).
# The lowest-output-error member is NOT a good choice: on the 9-layer run it was the worst member in layers 4-8
# (independently trained members that agree on a row have both found the teacher's row).
def _rows(n, l):
    R = torch.cat([n.layers[l].weight, n.layers[l].bias[:, None]], 1); return R / R.norm(dim=1, keepdim=True)
score = []
for i, ref in enumerate(mems):
    r0 = MLP(dims, act=act).to(dev).double(); r0.load_state_dict(ref.state_dict()); scale_normalize_(r0)
    agree = 0
    for k, m in enumerate(mems):
        if k != i:
            _, r = align_clone_to(m, r0)
            agree += sum(int(((_rows(r, l) * _rows(r0, l)).sum(1) > 0.999).sum()) for l in range(len(dims) - 2))
    score.append(agree)
bi = max(range(len(mems)), key=lambda i: score[i]) if a.member == "consensus" else int(a.member)
print(f"[refine] committee row agreement per member {score}", flush=True)
nq = 0
net = mems[bi]
for p in net.parameters():
    p.requires_grad_(False)
print(f"[refine] guess = member {bi} of {len(mems)} | dims {dims} | start max_eps {param_errors(net, T)['max_eps']:.2e}", flush=True)

# committee copies in a common (scale-normalised) gauge, for the fallback guesses
with torch.no_grad():
    nz = MLP(dims, act=act).to(dev).double(); nz.load_state_dict(net.state_dict()); scale_normalize_(nz)
    col_s, s_prev = [], torch.ones(dims[0], device=dev, dtype=torch.float64)
    for l in range(len(dims) - 2):                          # column scales nz applies to layer l's inputs
        col_s.append(s_prev); s_prev = (net.layers[l].weight * s_prev[None]).norm(dim=1)
    alt = [align_clone_to(m, nz)[1] for k, m in enumerate(mems) if k != bi]


def rows(n, l):
    R = torch.cat([n.layers[l].weight, n.layers[l].bias[:, None]], 1)
    return R / R.norm(dim=1, keepdim=True)


def orphans(l, Wr, br, ok, cos=0.99):
    """[(support, row in nz gauge)]: member rows matching no refined row of layer l, clustered, most members first"""
    Rr = torch.cat([Wr[ok] * col_s[l][None], br[ok][:, None]], 1)
    Rr = Rr / Rr.norm(dim=1, keepdim=True)
    pool = []
    for k, m in enumerate([nz] + alt):
        R = rows(m, l)
        pool += [(r, k) for r in R[(R @ Rr.T).abs().amax(1) < cos]]
    out, used = [], [False] * len(pool)
    for i, (ri, _) in enumerate(pool):
        if used[i]:
            continue
        g = [k for k, (rk, _) in enumerate(pool) if not used[k] and float(rk @ ri) > cos]
        for k in g:
            used[k] = True
        support=len({pool[k][1] for k in g})
        group_rows=torch.stack([pool[k][0] for k in g])
        centrality=(group_rows@group_rows.T).sum(1)
        # Try actual committee guesses: lower-median aggregation of two rows
        # takes the coordinatewise minimum and can destroy their closeness.
        for rank in centrality.argsort(descending=True)[:3].tolist():
            out.append((support,group_rows[rank]))
    return sorted(out, key=lambda t: -t[0])


def distinct_candidate(j, w, b, Wr, br, ok, tol=1e-5):
    """Do not count the same recovered hyperplane twice (sign invariant)."""
    others=ok.clone();others[j]=False
    if not bool(others.any()):return True
    r=torch.cat([w.flatten(), b.reshape(1)]).double();r=r/r.norm()
    rows=torch.cat([Wr[others],br[others,None]],1).double()
    rows=rows/rows.norm(dim=1,keepdim=True)
    distance=torch.minimum((rows-r).norm(dim=1),(rows+r).norm(dim=1))
    return bool((distance>tol).all())


def install(l, idx, Wr, br):
    """scale-match refined unit rows to the guess (as run.py --fast)"""
    Lf = net.layers[l]; gw = Lf.weight[idx]; gb = Lf.bias[idx]
    uw = Wr[idx].double(); ub = br[idx].double()
    proj = (gw * uw).sum(1) + gb * ub; gn = (gw.pow(2).sum(1) + gb.pow(2)).sqrt()
    cs = torch.where(proj > 1e-6 * gn, proj, gn).clamp_min(1e-8)
    Lf.weight[idx] = cs[:, None] * uw; Lf.bias[idx] = cs * ub


def polish(l, chans=None):
    """light kink-point polish of layer l's rows (kink_solve.polish_layer); installs rows that moved < 1e-4,
    returns {unit: moved}"""
    global nq
    import copy, kink_solve
    from verify_layer1 import _Oracle
    orc = _Oracle(copy.deepcopy(T).eval())
    chans = list(range(net.layers[l].weight.shape[0])) if chans is None else chans
    out = kink_solve.polish_layer(orc, net, l, chans, torch.Generator(device=dev).manual_seed(55 + l), max_rows=20000)
    nq += orc.n
    moved = {}
    with torch.no_grad():
        for j, (w, b, _) in out.items():
            v = torch.cat([w.double().flatten(), torch.as_tensor(b, device=dev, dtype=torch.float64).flatten()]); v = v / v.norm()
            cur = torch.cat([net.layers[l].weight[j], net.layers[l].bias[j:j + 1]]); cur = cur / cur.norm()
            if float(v @ cur) < 0:
                v = -v
            dv = float((v - cur).norm())
            if dv < 1e-4:
                moved[j] = dv
                install(l, torch.tensor([j], device=dev), v[None, :-1], v[-1:])
    print(f"[refine] L{l + 1} polish: {len(moved)}/{len(chans)} rows, max move "
          f"{max(moved.values()) if moved else 0:.1e}, {orc.n:,} queries", flush=True)
    return moved


completed_layers=a.start_layer-1
TRK = dict(track_step=a.track_step, track_need=None)
t0 = time.time()
if a.resume:
    _r = torch.load(a.resume, map_location=dev, weights_only=False)
    with torch.no_grad():
        for l in range(a.start_layer - 1):
            net.layers[l].weight.copy_(_r["state_dict"][f"layers.{l}.weight"]); net.layers[l].bias.copy_(_r["state_dict"][f"layers.{l}.bias"])
    nq += _r["queries"]
    e = param_errors(net, T)["max_eps_per_matrix"]
    print(f"[refine] resumed L1-L{a.start_layer - 1} from {a.resume} ({_r['queries']:,} queries) | "
          f"max eps {' '.join(f'{v:.1e}' for v in e[:2 * (a.start_layer - 1)])}", flush=True)
def design_refine(layer, channels):
    if not a.local_refine:
        return _mlp_refine_layer(T,net,layer,dev,act,only_channels=channels,angle_gate=12.,design=True)
    from peel.informative_kinks import recover_layer
    return recover_layer(T,net,layer,dev,only_channels=channels,angle_gate=12.,
                         max_rounds=a.fallback_rounds,failure_dir=None)

for l in range(a.start_layer - 1, len(dims) - 2):
    TRK["track_need"] = a.track_need_mult * dims[l]
    tl = time.time()
    # layer 1: design sampler (wide input); deeper: cheap surface tracking (~1.3M q/layer), design only for the
    # units tracking leaves unsolved
    if l == 0:
        Wr, br, ok, q = _mlp_refine_layer(T, net, l, dev, act, angle_gate=12.0, design=True)
    elif a.local_refine:
        from peel.local_kink_refine import recover_layer
        archive={}
        Wr,br,ok,q=recover_layer(T,net,l,rounds=a.local_rounds,archive=archive)
        if a.out:torch.save(archive,a.out+f".L{l+1}.points.pt")
    else:
        # tracking with spread-out points (step 2, 2*Din points): exact-prefix L3 test 3.5e-10 -> 1.6e-11 max error,
        # and the better-conditioned fit amplifies the prefix's error far less (defaults: L2 3e-9 -> L3 1.5e-6)
        import kink_solve
        Wr, br, ok, q = kink_solve.recover_layer(T, net, l, dev, gen=torch.Generator(device=dev).manual_seed(1234 + l),
                                                 angle_gate=12.0, sampling="track", verbose=False, **TRK)
    nq += q
    Wr, br = Wr.double(), br.double()
    # Later candidate isolation must see the rows we actually recovered.
    with torch.no_grad():install(l,ok.nonzero().flatten(),Wr,br)
    left = (~ok).nonzero().flatten().tolist()
    if left and l > 0:
        W2, b2, ok2, q = design_refine(l,left); nq += q
        for j in left:
            if bool(ok2[j]) and distinct_candidate(j,W2[j],b2[j],Wr,br,ok):
                Wr[j] = W2[j].double(); br[j] = b2[j].double(); ok[j] = True
        print(f"[refine] L{l + 1}: track left {len(left)}, design fallback solved {sum(bool(ok2[j]) for j in left)}", flush=True)
        left = (~ok).nonzero().flatten().tolist()
    if a.out:
        torch.save(dict(dims=dims,act=act,state_dict=net.state_dict(),queries=nq,
                        frontier=l,refined_weight=Wr,refined_bias=br,refined_mask=ok),
                   a.out + f".L{l+1}.partial.pt")
    if left:
        # cluster at cos 0.98 and keep single-member rows too: a unit only 3 members learned, at cos 0.984-0.997,
        # formed three singleton clusters at 0.99 and was filtered out by a support >= 2 rule (L6, 9-layer run)
        oc = orphans(l, Wr, br, ok, cos=0.98)[:3 * len(left) + 3]
        cands = [r for sup, r in oc]
        print(f"[refine] L{l + 1}: {len(left)} unsolved -> {len(cands)} committee orphan rows, support {[sup for sup, _ in oc]}", flush=True)
        for j in left:
            g0 = (net.layers[l].weight[j].clone(), net.layers[l].bias[j].clone())
            for r in cands:
                with torch.no_grad():
                    w_ = r[:-1] / col_s[l]; b_ = r[-1]; mag = float(torch.cat([g0[0], g0[1][None]]).norm()) / float(torch.cat([w_, b_[None]]).norm())
                    net.layers[l].weight[j] = mag * w_; net.layers[l].bias[j] = mag * b_
                W2, b2, ok2, q = design_refine(l,[j]); nq += q
                if bool(ok2[j]) and distinct_candidate(j,W2[j],b2[j],Wr,br,ok):
                    Wr[j] = W2[j].double(); br[j] = b2[j].double(); ok[j] = True
                    print(f"[refine]   unit {j}: solved from a committee orphan row", flush=True)
                    break
            else:
                with torch.no_grad():
                    net.layers[l].weight[j] = g0[0]; net.layers[l].bias[j] = g0[1]
                print(f"[refine]   unit {j}: no committee row works", flush=True)
        left = (~ok).nonzero().flatten().tolist()
    if left and l > 0:
        # guess-free: every other unit of layers <= l is exact, so a black-box kink they do not explain belongs to
        # an unsolved unit (or a deeper layer; the global kink test rejects those).  Its row (prefix-Jacobian
        # mapback, coarse at depth) is the guess for an exact design solve.
        import guessfree
        # Exclusion must use the recovered rows, not stale committee guesses.
        gf_net=copy.deepcopy(net)
        with torch.no_grad():
            idx=ok.nonzero().flatten()
            gf_net.layers[l].weight[idx]=Wr[idx]
            gf_net.layers[l].bias[idx]=br[idx]
        gf_teacher=T
        if a.local_refine and dims[0]>dims[1]:
            from peel.input_subspace import project_first_rowspace
            gf_teacher,gf_net,_=project_first_rowspace(T,gf_net)
            print(f"[refine] guess-free input subspace: {dims[0]} -> {dims[1]} dimensions",flush=True)
        rows_gf, q, info = guessfree.recover_frontier_rows(gf_teacher, gf_net, l, ok.nonzero().flatten().tolist(), lines=a.gf_lines,
                                                           device=dev, seed=11 + l, stop_at=len(left))
        nq += q
        print(f"[refine] L{l + 1}: guess-free scan {len(rows_gf)} candidate rows, {q:,} queries, clusters {info['clusters'][:8]}", flush=True)
        for j in left:
            g0 = (net.layers[l].weight[j].clone(), net.layers[l].bias[j].clone())
            for r in rows_gf:
                with torch.no_grad():
                    r = r.double(); mag = float(torch.cat([g0[0], g0[1][None]]).norm())
                    net.layers[l].weight[j] = mag * r[:-1]; net.layers[l].bias[j] = mag * r[-1]
                W2, b2, ok2, q = design_refine(l,[j]); nq += q
                if bool(ok2[j]) and distinct_candidate(j,W2[j],b2[j],Wr,br,ok):
                    Wr[j] = W2[j].double(); br[j] = b2[j].double(); ok[j] = True
                    print(f"[refine]   unit {j}: solved from a guess-free row", flush=True)
                    break
            else:
                with torch.no_grad():
                    net.layers[l].weight[j] = g0[0]; net.layers[l].bias[j] = g0[1]
                print(f"[refine]   unit {j}: unsolved", flush=True)
    with torch.no_grad():
        install(l, ok.nonzero().flatten(), Wr, br)
    if l > 0 and bool(ok.all()) and not a.local_refine:
        # exactness check: the track gate (12 deg) can pass a row 1e-2 off (L3 unit, 9-layer run). Re-solve every
        # row with the light polish (fresh kink points bracketed at ~1e-5 around the row, 8 queries each); a row
        # that polishes to within 1e-4 takes the polished value, one that cannot be polished goes to design.
        import kink_solve
        W3, b3, m3, q = kink_solve.recover_layer(T, net, l, dev, gen=torch.Generator(device=dev).manual_seed(777 + l),
                                                 angle_gate=12.0, sampling="track", verbose=False, **TRK); nq += q
        v1 = torch.cat([Wr, br[:, None]], 1); v1 = v1 / v1.norm(dim=1, keepdim=True)
        v3 = torch.cat([W3.double(), b3.double()[:, None]], 1); v3 = v3 / v3.norm(dim=1, keepdim=True)
        v3 = v3 * torch.sign((v1 * v3).sum(1, keepdim=True))
        gap = (v1 - v3).abs().amax(1)
        # tracking's own noise grows with depth (two-pass median gap: L2 1e-9, L3 5e-7), so flag OUTLIERS only:
        # rows the second pass failed, or whose gap exceeds both 1e-7 and 50x the layer's median gap
        thr = max(1e-7, 50 * float(gap[m3].median())) if bool(m3.any()) else 1e-7
        bad = [j for j in range(ok.numel()) if not bool(m3[j]) or float(gap[j]) > thr]
        print(f"[refine] L{l + 1}: second independent track ({q:,} q): {ok.numel() - len(bad)}/{ok.numel()} rows agree "
              f"to {thr:.0e} (median gap {float(gap[m3].median()):.1e})", flush=True)
        if bad:
            W2, b2, ok2, q = design_refine(l,bad); nq += q
            with torch.no_grad():
                idx = torch.tensor([j for j in bad if bool(ok2[j])], device=dev, dtype=torch.long)
                if len(idx):
                    install(l, idx, W2.double(), b2.double())
            ok[[j for j in bad if not bool(ok2[j])]] = False
            print(f"[refine] L{l + 1}: track disagreement {len(bad)} rows -> design solved {int(len(idx))}", flush=True)
    if a.unit_report:                                      # scoring only: worst units of this layer vs the teacher
        t_, r_ = align_clone_to(net, T)
        ew = (t_.layers[l].weight - r_.layers[l].weight).abs().amax(1); eb = (t_.layers[l].bias - r_.layers[l].bias).abs()
        print(f"[refine]   worst units (aligned idx): W {[(int(i), f'{float(ew[i]):.1e}') for i in ew.topk(3).indices]} "
              f"b {[(int(i), f'{float(eb[i]):.1e}') for i in eb.topk(3).indices]}", flush=True)
    e = param_errors(net, T)["max_eps_per_matrix"]
    print(f"[refine] L{l + 1}: {int(ok.sum())}/{ok.numel()} rows | {nq:,} queries so far | {time.time() - tl:.0f}s | "
          f"L{l + 1} max eps W {e[2 * l]:.2e} b {e[2 * l + 1]:.2e}", flush=True)
    if bool(ok.all()):completed_layers=l+1
    if a.out:
        torch.save(dict(dims=dims,act=act,state_dict=net.state_dict(),queries=nq,
                        completed_layers=completed_layers,frontier=l,refined_mask=ok),
                   a.out+f".L{l+1}.pt")
    if a.stop_after and l + 1 >= a.stop_after:
        break
    if not bool(ok.all()):
        print("[refine] incomplete layer: stopping (an unsolved unit poisons every layer above)", flush=True)
        break
if completed_layers == len(dims)-2:
    with torch.no_grad():                                        # output layer: closed-form least squares
        Xh = torch.randn(a.head_points, dims[0], device=dev, dtype=torch.float64); Yh = T(Xh); nq += len(Xh)
        H = Xh
        for lay in net.layers[:-1]:
            H = net.act(lay(H))
        sol = torch.linalg.lstsq(torch.cat([H, torch.ones(len(H), 1, device=dev, dtype=H.dtype)], 1), Yh).solution
        net.layers[-1].weight.copy_(sol[:-1].T); net.layers[-1].bias.copy_(sol[-1])
else:
    print(f"[refine] INCOMPLETE: {completed_layers}/{len(dims)-2} hidden layers accepted; output fit deferred",flush=True)

e = param_errors(net, T)
print(f"[refine] FINAL max_eps {e['max_eps']:.3e} mean {sum(e['mean_eps_per_matrix']) / len(e['mean_eps_per_matrix']):.3e} | "
      f"per-matrix max {' '.join(f'{v:.1e}' for v in e['max_eps_per_matrix'])}", flush=True)
print(f"[refine] {nq:,} black-box queries (refine + head), {time.time() - t0:.0f}s", flush=True)
if a.out:
    torch.save({"dims": dims, "act": act, "state_dict": net.state_dict(), "queries": nq, "completed_layers": completed_layers,
                "complete": completed_layers == len(dims)-2}, a.out)
    print("[refine] saved", a.out, flush=True)
