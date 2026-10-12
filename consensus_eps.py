"""Per-consensus-neuron weight error against the black box, from a population
snapshot. For every weight matrix: units in isolated consensus (rows / cols, the
same rule as the [consensus/matrix] log line) and the error of the CONSENSUS VALUE
(the loss-ordered seed's unit vector) against the teacher's matched unit, in the
canonical gauge, over the identified entries. Non-consensus units are scored too
(their reference member's own unit) for contrast.

usage: python consensus_eps.py recon/_pop__....pt [--probe N]
"""
import sys, math, argparse, types, torch
from nets import MLP
from align import scale_normalize_, _preacts, pairing_preacts, sign_align_clone
from head_refine import _act_perm
from reverse_peel import ReversePeel


@torch.no_grad()
def consensus_eps(pop, teacher, losses, cfg, device, n_probe=8192):
    dims = pop[0].dims; nW = len(dims) - 1; nh = nW - 1
    rp = ReversePeel(dims, cfg, device)
    P = len(pop); quorum = math.ceil(rp.quorum_ratio * P)
    ref = min(range(P), key=lambda i: losses[i])
    order = sorted(range(P), key=lambda i: losses[i])
    probe = rp._probe(torch.empty(1, dims[0]))
    leaky = isinstance(pop[0].act, torch.nn.LeakyReLU)
    def _rown(Rk):                                   # tanh/sigmoid: no scale gauge -> compare RAW weights
        return Rk / Rk.norm(dim=1, keepdim=True).clamp_min(1e-12) if leaky else Rk
    def _coln(Ck):
        return Ck / Ck.norm(dim=0, keepdim=True).clamp_min(1e-12) if leaky else Ck
    Zref = _preacts(pop[ref], probe)
    canon, perms, corrs = [], [], []
    for m in pop:                                   # paired to the reference; polarity aligned by activation (tanh/sigmoid)
        mc, ps, cs = sign_align_clone(m, Zref, probe)
        if leaky:
            scale_normalize_(mc)                    # scale gauge (leaky-ReLU only; it would re-flip tanh by the max-entry rule)
        canon.append(mc); perms.append(ps); corrs.append(cs)
    ident = [(torch.stack([corrs[i][l] for i in range(P)]) > rp.id_corr).sum(0) >= quorum for l in range(nh)]
    T, tperm, _ = sign_align_clone(teacher, Zref, probe)                # ref unit i <-> teacher unit tperm[i]
    if leaky:
        scale_normalize_(T)
    out = {}
    row_ok_all = {}
    for k in range(nh):                                                  # pass 1: rows of every hidden layer
        n_units = dims[k + 1]
        in_mask = torch.ones(dims[k], dtype=torch.bool, device=device) if k == 0 else ident[k - 1]
        if int(in_mask.sum()) < rp.min_ident * dims[k]:            # inputs not identified yet: use all paired columns
            in_mask = torch.ones(dims[k], dtype=torch.bool, device=device)
        cols_in = in_mask.nonzero().flatten()
        # ---- rows
        R = []
        for mi, mc in enumerate(canon):
            W, b = mc.layers[k].weight, mc.layers[k].bias
            Wk = W[perms[mi][k]]
            if k > 0:
                Wk = Wk[:, perms[mi][k - 1]]
            Rk = torch.cat([Wk[:, cols_in], b[perms[mi][k]][:, None]], 1)
            R.append(_rown(Rk))
        U = torch.stack(R)
        Wt = T.layers[k].weight[tperm[k]]
        if k > 0:
            Wt = Wt[:, tperm[k - 1]]
        Tt = torch.cat([Wt[:, cols_in], T.layers[k].bias[tperm[k]][:, None]], 1)
        Tt = _rown(Tt)
        enough = int(in_mask.sum()) >= rp.min_ident * dims[k]
        def _rowtest(U, Tt):
            cons = torch.zeros(n_units, dtype=torch.bool, device=device); err = torch.zeros(n_units, device=device)
            for j in range(n_units):
                seed = None
                if enough:
                    for r in order:
                        if int(rp._agree(U, j, r).sum()) >= quorum:
                            seed = r; break
                v = U[seed, j] if seed is not None else U[ref, j]
                cons[j] = seed is not None
                err[j] = (v - Tt[j]).abs().max()
            return cons, err
        row_cons, row_err = _rowtest(U, Tt)
        row_dir = None; row_mag = None
        if not leaky:                                    # tanh/sigmoid: ALSO direction-only and magnitude-only views
            Ud = U / U.norm(dim=2, keepdim=True).clamp_min(1e-12); Td = Tt / Tt.norm(dim=1, keepdim=True).clamp_min(1e-12)
            row_dir = _rowtest(Ud, Td)
            logn = U[:, :, :-1].norm(dim=2).clamp_min(1e-12).log(); logt = Tt[:, :-1].norm(dim=1).clamp_min(1e-12).log()
            mcons = torch.zeros(n_units, dtype=torch.bool, device=device); merr = torch.zeros(n_units, device=device)
            for j in range(n_units):
                seed = None
                for r in order:
                    if int(((logn[:, j] - logn[r, j]).abs() <= 0.02).sum()) >= quorum:
                        seed = r; break
                mcons[j] = (seed is not None) and bool(row_dir[0][j])     # only meaningful once the DIRECTION is in consensus
                merr[j] = (logn[seed if seed is not None else ref, j] - logt[j]).abs()      # |log(norm ratio)| ~ relative error
            row_mag = (mcons, merr)
        row_ok_all[k] = (row_cons.clone(), row_err.clone(), Tt.abs().mean().item(), (row_dir, row_mag))
    for k in range(nh):                                                  # pass 2: columns, gated on the NEXT layer's row consensus
        n_units = dims[k + 1]
        row_cons, row_err, row_scale, (row_dir, row_mag) = row_ok_all[k]
        # ---- cols (outgoing: columns of W_{k+2} for this layer's units). A column entry counts
        #      only in rows whose UNIT passed its own row consensus: paired is not enough -- one
        #      wrong (but paired) layer-k+1 row otherwise poisons every column (tanh 5-layer: 166/200
        #      W3 columns at 0.1 error from a single 0.27 row; same gating as entry_coverage).
        is_head = (k + 1 == nh)
        C = []
        out_mask = None if is_head else (ident[k + 1] & row_ok_all[k + 1][0])
        if out_mask is not None and int(out_mask.sum()) < rp.min_ident * dims[k + 2]:
            out_mask = row_ok_all[k + 1][0].clone()        # fallback: all next-layer units whose rows passed
        for mi, mc in enumerate(canon):
            Wn = mc.layers[k + 1].weight[:, perms[mi][k]]
            if not is_head:
                Wn = Wn[perms[mi][k + 1]][out_mask]
            C.append(_coln(Wn).T)
        Uc = torch.stack(C)
        Wtn = T.layers[k + 1].weight[:, tperm[k]]
        if not is_head:
            Wtn = Wtn[tperm[k + 1]][out_mask]
        Tc = _coln(Wtn).T
        col_cons = torch.zeros(n_units, dtype=torch.bool, device=device); col_err = torch.zeros(n_units, device=device)
        n_out_id = Uc.shape[-1]; enough_c = n_out_id >= rp.min_ident * (dims[-1] if is_head else dims[k + 2])
        for j in range(n_units):
            seed = None
            if enough_c:
                for r in order:
                    if int(rp._agree(Uc, j, r).sum()) >= quorum:
                        seed = r; break
            v = Uc[seed, j] if seed is not None else Uc[ref, j]
            col_cons[j] = seed is not None
            col_err[j] = (v - Tc[j]).abs().max() if Tc.shape[1] > 0 else float("nan")   # no gated output rows yet
        out[k] = dict(row_cons=row_cons, row_err=row_err, col_cons=col_cons, col_err=col_err,
                      row_scale=row_scale, col_scale=Tc.abs().mean().item(), is_head=is_head)
        if row_dir is not None:
            out[k]["row_dir_cons"], out[k]["row_dir_err"] = row_dir
            out[k]["row_mag_cons"], out[k]["row_mag_err"] = row_mag
    return out


def _q(x):
    if x.numel() == 0:
        return "n/a"
    return f"mean {x.mean():.1e} max {x.max():.1e}"


def format_by_matrix(res, dims):
    """One block per weight matrix W_k with BOTH views of its entries: rows (grouped by the
    units it feeds; from layer k-1's report) and cols (grouped by the units feeding it; from
    layer k-2's report). W1 has only rows (inputs are pixels); the head only cols."""
    nW = len(dims) - 1; lines = []
    for k in range(1, nW + 1):
        lines.append(f"W{k}{'=head' if k == nW else ''}:")
        if k - 1 in res:                                      # rows view: layer k-1's rows
            v = res[k - 1]; rc, re_ = v["row_cons"], v["row_err"]
            lines.append(f"    rows (per unit of L{k}):   consensus {int(rc.sum()):3d}/{len(rc)} -> {_q(re_[rc])}"
                         f" | non-consensus {int((~rc).sum()):3d} -> {_q(re_[~rc])}")
            if "row_dir_cons" in v:                               # tanh/sigmoid: direction-only and magnitude-only views as well
                dc, de = v["row_dir_cons"], v["row_dir_err"]
                lines.append(f"      direction only:          consensus {int(dc.sum()):3d}/{len(dc)} -> {_q(de[dc])}"
                             f" | non-consensus {int((~dc).sum()):3d} -> {_q(de[~dc])}")
                mc, me = v["row_mag_cons"], v["row_mag_err"]
                lines.append(f"      magnitude only (2% rel): consensus {int(mc.sum()):3d}/{len(mc)} -> rel err {_q(me[mc])}"
                             f" | non-consensus {int((~mc).sum()):3d} -> rel err {_q(me[~mc])}")
        if k - 2 in res:                                      # cols view: layer k-2's outgoing cols
            v = res[k - 2]; cc, ce = v["col_cons"], v["col_err"]
            lines.append(f"    cols (per unit of L{k-1}): consensus {int(cc.sum()):3d}/{len(cc)} -> {_q(ce[cc])}"
                         f" | non-consensus {int((~cc).sum()):3d} -> {_q(ce[~cc])}")
    return "\n".join(lines)




@torch.no_grad()
def entry_coverage(pop, teacher, losses, cfg, device, ref=None, cert_gauge=False, sticky=None, value="seed"):
    """ENTRY-level union of the two views. Common frame = canonical gauge (hidden rows
    unit-norm), teacher in the same gauge, units paired to the reference. An entry
    W_k[j, i] is covered by the ROW view if row j is in consensus and the cluster agrees
    on that entry (spread <= eps), by the COL view likewise for column i; the value is
    the seed member's canonical entry. Returns per matrix: masks rows/cols and the
    per-entry |value - teacher| for the union, by class."""
    dims = pop[0].dims; nW = len(dims) - 1; nh = nW - 1
    rp = ReversePeel(dims, cfg, device)
    P = len(pop); quorum = math.ceil(rp.quorum_ratio * P)
    ref = min(range(P), key=lambda i: losses[i]) if ref is None else ref
    order = sorted(range(P), key=lambda i: losses[i])
    probe = rp._probe(torch.empty(1, dims[0]))
    leaky = isinstance(pop[0].act, torch.nn.LeakyReLU)
    Zref = _preacts(pop[ref], probe)
    aligned, perms, corrs = [], [], []
    for m in pop:                                   # paired to the reference; polarity aligned by activation (tanh/sigmoid)
        mc, ps, cs = sign_align_clone(m, Zref, probe); aligned.append(mc); perms.append(ps); corrs.append(cs)
    ident = [(torch.stack([corrs[i][l] for i in range(P)]) > rp.id_corr).sum(0) >= quorum for l in range(nh)]
    T_aligned, tperm, _ = (sign_align_clone(teacher, Zref, probe) if teacher is not None else (None, None, None))
    # cert_gauge: the canonical gauge must not depend on UNCERTIFIED units. In the plain
    # gauge a hidden unit's scale is its full row norm, which includes entries at uncertified
    # (still-moving) input units; those scales then leak into every certified row above
    # (7e-3 per entry observed with 2 moving units) and flip borderline rows in and out of
    # consensus although their weights are frozen. Here the scale of a unit is the norm of
    # its row over CERTIFIED input units only (a valid per-unit gauge, applied identically
    # to every member and the teacher), and the row test is restricted to certified inputs.
    # Certification is sequential by layer, so layer k-1's row_ok is known when layer k is
    # canonicalized. `sticky` (per hidden layer, ref frame) adds units certified earlier
    # whose rows are frozen (pin mode) to the certified set used for the gauge.
    if not cert_gauge:
        canon = []
        for mc in aligned:
            if leaky:
                scale_normalize_(mc)
            canon.append(mc)
        if teacher is not None:
            T = T_aligned
            if leaky:
                scale_normalize_(T)
    state = [None] * P; tstate = None; row_ok_prev = None
    out = {}
    for k in range(nW):                                   # matrix index k: W_{k+1}, rows = layer k units (or outputs)
        n_out, n_in = dims[k + 1], dims[k]
        is_head = (k == nW - 1)
        # canonical matrices in the reference frame (rows: layer k units unless head; cols: layer k-1 units unless k==0)
        Ms, biases = [], []
        prev_state = list(state)
        if cert_gauge:
            odd = not isinstance(pop[0].act, torch.nn.LeakyReLU)       # tanh/sigmoid: polarity gauge, no scale gauge
            def _chain(W, b, st, perm_rows, perm_cols):
                We = W if st is None else W * st[None, :]
                if not is_head:
                    if odd:                                            # polarity already aligned by activation (sign_align_clone); no scale gauge
                        bc = b; st = None
                        We = We[perm_rows]; bc = bc[perm_rows]
                        if k > 0:
                            We = We[:, perm_cols]
                        return We, bc, st
                    if k == 0:
                        cm = torch.ones(n_in, dtype=torch.bool, device=device)
                    else:
                        cm = torch.zeros(n_in, dtype=torch.bool, device=device); cm[perm_cols] = row_ok_prev
                    nr = (We * cm[None, :]).norm(dim=1).clamp_min(1e-12)
                    We = We / nr[:, None]; bc = b / nr; st = nr
                    We = We[perm_rows]; bc = bc[perm_rows]
                else:
                    bc = b
                if k > 0:
                    We = We[:, perm_cols]
                return We, bc, st
            for mi, m in enumerate(aligned):
                We, bc, state[mi] = _chain(m.layers[k].weight, m.layers[k].bias, state[mi],
                                           perms[mi][k] if not is_head else None, perms[mi][k - 1] if k > 0 else None)
                Ms.append(We); biases.append(bc)
            if teacher is not None:
                Wt, _, tstate = _chain(T_aligned.layers[k].weight, T_aligned.layers[k].bias, tstate,
                                       tperm[k] if not is_head else None, tperm[k - 1] if k > 0 else None)
            else:
                Wt = torch.zeros(n_out, n_in, device=device)
        else:
            for mi, mc in enumerate(canon):
                W = mc.layers[k].weight
                if not is_head:
                    W = W[perms[mi][k]]
                if k > 0:
                    W = W[:, perms[mi][k - 1]]
                Ms.append(W)
                biases.append(mc.layers[k].bias if is_head else mc.layers[k].bias[perms[mi][k]])
            if teacher is not None:
                Wt = T.layers[k].weight
                if not is_head:
                    Wt = Wt[tperm[k]]
                if k > 0:
                    Wt = Wt[:, tperm[k - 1]]
            else:
                Wt = torch.zeros(n_out, n_in, device=device)   # no scoring
        M = torch.stack(Ms)                               # (P, n_out, n_in)
        # input scales of this matrix per member (member unit order): the gauge the stored
        # values V live in; holders convert V -> member gauge with THESE, not a live chain
        in_scales = ([None] * P if k == 0 else
                     (prev_state if cert_gauge else [rp._in_scales(m, k) for m in aligned]))
        in_mask = torch.ones(n_in, dtype=torch.bool, device=device) if k == 0 else ident[k - 1]
        if cert_gauge and k > 0:
            in_mask = in_mask & row_ok_prev
        out_mask = torch.ones(n_out, dtype=torch.bool, device=device) if is_head else ident[k]
        row_cov = torch.zeros(n_out, n_in, dtype=torch.bool, device=device); val_r = torch.zeros(n_out, n_in, device=device)
        bias_ref = biases[ref]
        bias_v = bias_ref.clone(); bias_cov = torch.zeros(n_out, dtype=torch.bool, device=device)
        col_cov = torch.zeros(n_out, n_in, dtype=torch.bool, device=device); val_c = torch.zeros(n_out, n_in, device=device)
        # ---- row view (not for the head: its rows are outputs, no pairing needed but rows span all L units -> allowed)
        cols_in = in_mask.nonzero().flatten()
        if cols_in.numel() >= rp.min_ident * n_in:
            bias = torch.stack(biases)
            R = torch.cat([M[:, :, cols_in], bias[:, :, None]], 2)
            if leaky:                                        # tanh/sigmoid: raw weights (no scale gauge)
                R = R / R.norm(dim=2, keepdim=True).clamp_min(1e-12)
            for j in range(n_out):
                for r in order:
                    mem = rp._agree(R, j, r)
                    if int(mem.sum()) >= quorum:
                        spread = (R[mem, j] - R[r, j]).abs().amax(0)[:-1]
                        ok = cols_in[spread <= rp.eps]
                        row_cov[j, ok] = True
                        if value == "mean":                       # quorum MEAN of the canonical entries (noise averages out)
                            val_r[j, ok] = M[mem, j][:, ok].mean(0); bias_v[j] = bias[mem, j].mean()
                        else:
                            val_r[j, ok] = M[r, j, ok]; bias_v[j] = bias[r, j]
                        bias_cov[j] = True
                        break
        # ---- col view: like the log line, a column entry counts only in rows whose UNIT
        #      passed its own row consensus (paired is not enough: an unsolved rare unit
        #      can be paired at 0.99 and still be wrong; members then share the error)
        # (not for the head: its rows are OUTPUTS, whose consensus needs the whole last
        #  hidden layer solved; the column view is the gauge-correct head measure and is
        #  not gated by rows -- same as the [consensus/matrix] log line)
        row_ok = (row_cov.any(1) if (bool(row_cov.any()) and not is_head)
                  else torch.ones(n_out, dtype=torch.bool, device=device))
        if not is_head:
            row_ok_prev = row_ok | (sticky[k] if sticky is not None else False)   # feeds layer k+1's gauge (cert_gauge)
        rows_out = (out_mask & row_ok).nonzero().flatten()
        if rows_out.numel() >= rp.min_ident * n_out:
            C = M[:, rows_out, :]                          # (P, n_out_id, n_in)
            if leaky:
                C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)
            Ct = C.transpose(1, 2)                         # (P, n_in, n_out_id)
            for i in range(n_in):
                for r in order:
                    mem = rp._agree(Ct, i, r)
                    if int(mem.sum()) >= quorum:
                        spread = (Ct[mem, i] - Ct[r, i]).abs().amax(0)
                        ok = rows_out[spread <= rp.eps]
                        col_cov[ok, i] = True
                        val_c[ok, i] = M[mem][:, ok, i].mean(0) if value == "mean" else M[r, ok, i]
                        break
        E_r = (val_r - Wt).abs(); E_c = (val_c - Wt).abs()
        # union value: row view where available, else col view, else the reference member
        union = row_cov | col_cov
        E_u = torch.where(row_cov, E_r, E_c)
        V = torch.where(row_cov, val_r, torch.where(col_cov, val_c, M[ref]))
        out[k] = dict(row_cov=row_cov, col_cov=col_cov, E_r=E_r, E_c=E_c, E_u=E_u, scale=Wt.abs().mean().item(),
                      E_ref=(M[ref] - Wt).abs(), is_head=is_head, V=V, b=bias_v, b_cov=bias_cov, M_ref=M[ref],
                      perms=perms, ref=ref, in_mask=in_mask, in_scales=in_scales, b_ref=bias_ref.clone())
    return out


def format_entries(res, dims):
    nW = len(dims) - 1; lines = []
    for k, v in res.items():
        rc, cc = v["row_cov"], v["col_cov"]; both = rc & cc; ro = rc & ~cc; co = cc & ~rc; nei = ~(rc | cc)
        n = rc.numel()
        def cls(mask, E):
            return f"{int(mask.sum())/n:6.1%} (mean {E[mask].mean():.1e} max {E[mask].max():.1e})" if bool(mask.any()) else "  0.0%"
        lines.append(f"W{k+1}{'=head' if v['is_head'] else ''} entries ({n}, teacher mean |entry| {v['scale']:.3f}):")
        lines.append(f"    rows view {cls(rc, v['E_r'])} | cols view {cls(cc, v['E_c'])} | UNION {cls(rc | cc, v['E_u'])}")
        lines.append(f"    both {cls(both, v['E_u'])} | rows-only {cls(ro, v['E_r'])} | cols-only {cls(co, v['E_c'])} | neither {cls(nei, v['E_ref'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--entries", action="store_true")
    a = ap.parse_args(); dev = a.device; torch.manual_seed(0)              # reproducible probe
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); dims = ck["dims"]
    t = MLP(dims, act=ck.get("act", "leaky_relu")).to(dev); t.load_state_dict(ck["teacher_state"])
    pop = []
    for sd in ck["pop_states"]:
        m = MLP(dims, act=ck.get("act", "leaky_relu")).to(dev); m.load_state_dict(sd); pop.append(m)
    X = (torch.rand(4000, dims[0], device=dev) * 2 - 1)
    with torch.no_grad():
        yt = t(X); losses = [float((m(X) - yt).abs().mean()) for m in pop]
    cfg = types.SimpleNamespace(reverse_eps=0.02, reverse_thresh=1.0, cluster_quorum=0.625, qg_init="uniform", qg_range=1.0)
    res = consensus_eps(pop, t, losses, cfg, dev)
    print(f"{a.ckpt}  iter {ck.get('iter')}  dims {dims}  member MAE {[f'{l:.3f}' for l in losses]}  ref m{min(range(len(pop)), key=lambda i: losses[i])}")
    print("max-abs error of the consensus unit vector vs the black box (canonical gauge, identified entries)")
    print(format_by_matrix(res, dims))
    if a.entries:
        print("\nENTRY-level coverage (union of the two views; value = seed member's canonical entry; error vs teacher, canonical gauge):")
        print(format_entries(entry_coverage(pop, t, losses, cfg, dev), dims))


@torch.no_grad()
def build_consensus_net(pop, teacher, losses, cfg, device, cert_gauge=False, ref=None, value="seed"):
    """Assemble ONE network from the committee: canonical gauge, reference member's unit
    order; every union-covered entry at its consensus value, uncovered entries and
    uncovered biases from the reference member. Returns (net, coverage dict)."""
    ent = entry_coverage(pop, teacher, losses, cfg, device, ref=ref, cert_gauge=cert_gauge, value=value)
    dims = pop[0].dims
    net = MLP(dims, act=pop[0].act_name).to(device)
    cov = {}
    for k in range(len(dims) - 1):
        e = ent[k]
        net.layers[k].weight.copy_(e["V"]); net.layers[k].bias.copy_(e["b"])
        cov[k] = dict(entries=float((e["row_cov"] | e["col_cov"]).float().mean()), bias=float(e["b_cov"].float().mean()))
    return net, cov, ent
