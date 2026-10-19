"""--reverse-peel: top-down committee peel (v2: frame-free pins, partial rows).

Train the committee as usual, but track consensus on the LAST layer first, in the
head's gauge-correct frame (columns per last-hidden unit, members' units paired by
activation matching, unit-normalized columns, the code's cluster/quorum rule), peel
it, then recurse backwards one layer at a time.

  head  : consensus columns -> EXACT columns by the kink-jump refiner
          (head_refine.refine_head, black-box, no prefix needed) -> pinned in every
          member at that member's own column magnitude (scale = gauge), gradient
          masked. The head is complete when >= thresh of the columns are pinned.
  L_k   : rows compared in the frame the layer above induces (row identity = the
          unit's pairing to the reference; input columns = activation pairing of
          the layer below), unit-normalized, the same cluster/quorum rule, but
          the agreement test and the pin only cover the IDENTIFIED input units
          (those every member pairs to the reference at correlation > id_corr);
          entries for unidentified inputs stay trainable (partial-row pin). This
          is what stops ONE unsolved input unit from vetoing the consensus of the
          whole layer above. Below the head the pins are at consensus precision
          (no suffix refiner yet); they are a training scaffold and their true
          error is logged against the teacher.

Pins are FRAME-FREE: the cache stores exact values by id; every member records
which unit holds each id, so a change of reference member (the most-pinned member
is always the reference) costs nothing, a replaced member gets every cached value
re-injected by activation pairing to the reference, and a fully fresh committee
gets the cache injected under the identity assignment. Pinned entries are written,
gradient hook-masked, Adam state zeroed and re-written at every check.
"""
import math
import torch
from align import _preacts, scale_normalize_, pairing_preacts, sign_align_clone, _scale_power
from head_refine import head_consensus_columns, refine_head, _act_perm


class ReversePeel:
    def __init__(self, dims, cfg, device):
        self.dims = list(dims)
        self.nW = len(dims) - 1                 # weight layers; head index = nW-1
        self.frontier = self.nW - 1
        self.eps = cfg.reverse_eps
        self.thresh = cfg.reverse_thresh
        self.quorum_ratio = cfg.cluster_quorum
        self.id_corr = 0.99                     # input unit "identified" if paired at corr > this in every member
        self.pin_cos = 0.9
        self.trim = 0.03                        # fraction of entries a cluster may disagree on
        self.min_ident = 0.1                    # min fraction of identified neighbor units for a row/col consensus to count
        self.force = getattr(cfg, 'reverse_force', False)  # force pins into units that match nothing (breaks mispaired members)
        self.device = device
        self.cache = {}                         # layer -> list of {"vec", "mask"(rows only)}
        self.pins = {}                          # (mi, l) -> state
        self.teacher64 = None
        self.queries = 0
        self.n_forced = 0
        self.done = False
        self.history = []
        self.by_id = {}                         # id(member) -> member index (for project)
        self.qg_init = getattr(cfg, 'qg_init', 'uniform'); self.qg_range = getattr(cfg, 'qg_range', 1.0)
        self.qg_init_std = getattr(cfg, 'qg_init_std', 0.5)

    def _probe(self, X):
        """Probe for pairing/consensus, drawn FRESH at every check from a FIXED
        distribution (the query-init distribution: uniform in the box, or N(0, std)),
        not from the training buffer. The buffer is a sliding window of disagreement
        queries whose composition drifts, so a buffer-drawn probe measured the units
        on different input regions at different checks and the counts moved with it.
        8192 points: the correlation standard error at 0.99 is ~2e-4, negligible."""
        n_probe, d = 8192, self.dims[0]
        if self.qg_init == "uniform":
            return (torch.rand(n_probe, d, device=self.device) * 2 - 1) * self.qg_range
        return torch.randn(n_probe, d, device=self.device) * self.qg_init_std

    # ------------------------------------------------------------- pins --
    def _agree(self, U, j, r):
        """Members whose unit-j vector agrees with seed r on all but a `trim`
        fraction of entries (a few ill-determined entries, e.g. columns of rarely
        active inputs, must not veto the whole row)."""
        D = (U[:, j] - U[r, j]).abs()
        k = max(1, int(round((1 - self.trim) * D.shape[-1])))
        return D.kthvalue(k, dim=-1).values <= self.eps

    def _n_units(self, l):
        return self.dims[l] if l == self.nW - 1 else self.dims[l + 1]

    def _st(self, mi, m, l):
        key = (mi, l); st = self.pins.get(key)
        if st is None or st["hooked"] != id(m):
            W, b = m.layers[l].weight, m.layers[l].bias
            keep_w = torch.zeros_like(W, dtype=torch.bool)
            keep_b = torch.zeros(W.shape[0], dtype=torch.bool, device=W.device)
            st = {"hooked": id(m), "keep_w": keep_w, "keep_b": keep_b, "unit_of": {}, "vals": {}}
            self.pins[key] = st
        return st

    def _units_pinned(self, st, l):
        return st["keep_w"].any(0) if l == self.nW - 1 else st["keep_b"]

    @torch.no_grad()
    def _snap(self, net, st, l):
        """Project the member's pinned entries back onto their pinned DIRECTIONS,
        keeping the radial component (the scale keeps training; it is gauge for a
        consistent member and the only thing a bad member needs to be able to fix).
        Floored at 10% of the current norm so a sign-flipped member can't collapse."""
        W, b = net.layers[l].weight, net.layers[l].bias
        for cid, (w, bb, u, cols) in st["vals"].items():
            if l == self.nW - 1:
                d = w / w.norm().clamp_min(1e-12)
                col = W[:, u]
                r = torch.maximum((col * d).sum(), 0.1 * col.norm())
                W[:, u] = d * r
            else:
                e = self.cache[l][cid]; vec = e["vec"].to(W.dtype)
                ref_cols = bb                                       # stored with the pin
                d = self._row_dir(net, l, vec, ref_cols, cols)
                cur = torch.cat([W[u, cols], b[u:u + 1]])
                r = torch.maximum((cur * d).sum(), 0.1 * cur.norm())
                W[u, cols] = (d[:-1] * r).to(W.dtype); b[u] = (d[-1] * r).to(b.dtype)

    @torch.no_grad()
    def _rewrite(self, pop):
        for (mi, l), st in self.pins.items():
            if mi >= len(pop) or st["hooked"] != id(pop[mi]):
                continue
            self._snap(pop[mi], st, l)

    @staticmethod
    def _zero_adam(opt, p, idx, column):
        st = opt.state.get(p)
        if not st:
            return
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st:
                if column:
                    st[k][:, idx] = 0
                else:
                    st[k][idx] = 0

    @torch.no_grad()
    def _pin_head(self, mi, m, opt, cid, u):
        st = self._st(mi, m, self.nW - 1)
        W = m.layers[-1].weight
        vec = self.cache[self.nW - 1][cid]["vec"].to(W.dtype)
        col = W[:, u]
        if float((col * vec).sum()) / float(col.norm().clamp_min(1e-12)) < self.pin_cos:
            self.n_forced += 1
        newcol = vec * col.norm().clamp_min(1e-6)
        W[:, u] = newcol
        st["keep_w"][:, u] = True
        st["unit_of"][cid] = u
        st["vals"][cid] = (newcol.clone(), None, u, None)

    @torch.no_grad()
    def _pin_row(self, mi, m, opt, l, cid, u, in_perm):
        """Pin cache entry cid of layer l into member unit u. in_perm maps the
        cache's (reference) input order to this member's input order (None = identity)."""
        st = self._st(mi, m, l)
        W, b = m.layers[l].weight, m.layers[l].bias
        e = self.cache[l][cid]; vec = e["vec"].to(W.dtype); mask = e["mask"]
        n_in = self.dims[l]
        mask = mask.clone()
        if l > 0 and getattr(self, "member_ok", None) is not None and mi < len(self.member_ok):
            mask &= self.member_ok[mi][l - 1]                     # inputs THIS member pairs confidently
        ref_cols = mask.nonzero().flatten()
        if ref_cols.numel() == 0:
            return
        cols = in_perm[ref_cols] if in_perm is not None else ref_cols
        d = self._row_dir(m, l, vec, ref_cols, cols)              # pinned direction, member gauge
        cur_id = torch.cat([W[u][cols], b[u:u + 1]])              # identified entries only
        if float((cur_id * d).sum()) / float(cur_id.norm().clamp_min(1e-12)) < self.pin_cos:
            self.n_forced += 1
        mag = cur_id.norm().clamp_min(1e-6)                      # same energy as the part it replaces
        W[u, cols] = (d[:-1] * mag).to(W.dtype); b[u] = (d[-1] * mag).to(b.dtype)
        st["keep_w"][u, cols] = True; st["keep_b"][u] = True
        st["unit_of"][cid] = u
        st["vals"][cid] = (cid, ref_cols.clone(), u, cols.clone())

    @torch.no_grad()
    def _row_dir(self, m, l, vec, ref_cols, cols):
        """Unit direction (identified entries + bias) of a cached canonical row,
        expressed in member m's LIVE gauge: entry i divided by the input scale n_i."""
        n_in = self.dims[l]
        n = self._in_scales(m, l)
        w = vec[ref_cols] if n is None else vec[ref_cols] / n[cols].to(vec.dtype)
        d = torch.cat([w, vec[n_in:n_in + 1]])
        return d / d.norm().clamp_min(1e-12)

    # ------------------------------------------------------------- step --
    def step(self, pop, opts, teacher, X, ref_hint, t, forward_masks=None, losses=None):
        if self.done:
            return None
        dev = self.device; L = self.nW - 1; nh = self.nW - 1
        self._rewrite(pop)
        self.by_id = {id(m): mi for mi, m in enumerate(pop)}
        for mi, m in enumerate(pop):
            for l in range(self.nW):
                self._st(mi, m, l)
        # reference = most-pinned member (head), ties -> the caller's best member
        counts = [int(self._units_pinned(self.pins[(mi, L)], L).sum()) for mi in range(len(pop))]
        best = max(counts)
        cands = [mi for mi in range(len(pop)) if counts[mi] == best]
        if losses is not None:
            ref_idx = min(cands, key=lambda mi: losses[mi])     # ties -> lowest training loss
        else:
            ref_idx = ref_hint if ref_hint in cands else cands[0]
        self.losses = losses
        if getattr(self, "force_ref", None) is not None:
            ref_idx = self.force_ref
        probe = self._probe(X)
        Z = [pairing_preacts(m, probe) for m in pop]
        perms, corrs = [], []
        for i in range(len(pop)):
            ps, cs = [], []
            for l in range(nh):
                p, c = _act_perm(Z[ref_idx][l], Z[i][l]); ps.append(p); cs.append(c)
            perms.append(ps); corrs.append(cs)
        # identified hidden units (per hidden layer, reference order): paired at
        # corr > id_corr in at least a QUORUM of members (min over all members would
        # let a few garbage members blank out whole layers)
        quorum = math.ceil(self.quorum_ratio * len(pop))
        Cm = [torch.stack([corrs[i][l] for i in range(len(pop))]) for l in range(nh)]   # (P, n)
        ident = [(Cm[l] > self.id_corr).sum(0) >= quorum for l in range(nh)]
        self.member_ok = [[Cm[l][i] > self.id_corr for l in range(nh)] for i in range(len(pop))]
        fwd = forward_masks or {}
        info = {"iter": t + 1, "frontier": self.frontier, "ref": ref_idx, "layers": {}}
        for l in range(L, self.frontier - 1, -1):
            n_units = self._n_units(l)
            # a fully fresh committee with a cache: identity injection everywhere
            if self.cache.get(l) and all(not self._units_pinned(self.pins[(mi, l)], l).any() for mi in range(len(pop))):
                for mi, m in enumerate(pop):
                    for cid, e in enumerate(self.cache[l]):
                        if l == L:
                            self._pin_head(mi, m, opts[mi], cid, cid)
                        else:
                            self._pin_row(mi, m, opts[mi], l, cid, cid, None)
            if l == L:
                li = self._step_head(pop, opts, X, probe, ref_idx, perms, teacher)
            else:
                li = self._step_rows(pop, opts, l, ref_idx, perms, ident, fwd.get(l))
            st_ref = self.pins[(ref_idx, l)]
            solved = self._units_pinned(st_ref, l).clone()
            if l != L and fwd.get(l) is not None:
                solved |= fwd[l].to(dev)
            li["pinned_ref"] = int(solved.sum()); li["n_units"] = n_units
            li["complete"] = li["pinned_ref"] >= self.thresh * n_units
            li.update(self._solved_eps(pop[ref_idx], teacher, probe, l, solved))
            info["layers"][l] = li
        info["report"] = self.layer_report(pop, ref_idx, perms, ident, losses)
        while self.frontier >= 0 and info["layers"].get(self.frontier, {}).get("complete", False):
            self.frontier -= 1
            info["advanced_to"] = self.frontier
        if self.frontier < 0:
            self.done = True
        self.history.append(info)
        return info

    # ------------------------------------------------------------- head --
    def _apply_cache(self, pop, opts, l, ref_idx, perms):
        """Inject every cached entry into every member that lacks it, via the
        reference's unit for that entry and the member's activation pairing."""
        st_ref = self.pins[(ref_idx, l)]
        L = self.nW - 1
        canon = {}                                          # member -> canonical clone (rows only)
        for cid, e in enumerate(self.cache.get(l, [])):
            if cid not in st_ref["unit_of"]:
                continue                                    # reference lacks it: skip this check
            u_ref = st_ref["unit_of"][cid]
            for mi, m in enumerate(pop):
                st = self.pins[(mi, l)]
                if cid in st["unit_of"]:
                    continue
                hl = l - 1 if l == L else l                    # hidden layer whose units are the row/column ids
                u = int(perms[mi][hl][u_ref])
                pinned_u = self._units_pinned(st, l)
                in_perm = perms[mi][l - 1] if (l != L and l > 0) else None
                # does the activation-paired unit already hold this value? if not, the
                # pairing is wrong for this unit (swapped / unformed): look for the
                # member's best-matching UNPINNED unit by weight similarity; force onto
                # the paired unit only when nothing matches (fresh member).
                sims = self._unit_sims(m, mi, l, cid, in_perm, canon)        # (n_units,)
                if sims[u] < self.pin_cos:
                    sims_free = sims.masked_fill(pinned_u, -2.0)
                    ub = int(sims_free.argmax())
                    if sims_free[ub] >= self.pin_cos:
                        u = ub
                    elif not self.force:
                        continue                            # no unit of this member matches: leave it
                                                            # unpinned here, retried at every check
                if bool(pinned_u[u]):
                    continue                                # unit already holds another entry
                if l == L:
                    self._pin_head(mi, m, opts[mi], cid, u)
                else:
                    self._pin_row(mi, m, opts[mi], l, cid, u, in_perm)

    @torch.no_grad()
    def _unit_sims(self, m, mi, l, cid, in_perm, canon):
        """Cosine of every unit of member m (layer l) with cache entry cid: head ->
        column cosine; rows -> cosine over the identified entries (+bias) in the
        member's canonical gauge, with the member's inputs mapped to the cache order."""
        e = self.cache[l][cid]; vec = e["vec"]
        if l == self.nW - 1:
            W = m.layers[l].weight
            return (W * vec[:, None].to(W.dtype)).sum(0) / W.norm(dim=0).clamp_min(1e-12)
        W, b = m.layers[l].weight, m.layers[l].bias
        ref_cols = e["mask"].nonzero().flatten()
        cols = in_perm[ref_cols] if in_perm is not None else ref_cols
        R = torch.cat([W[:, cols], b[:, None]], 1)
        v = self._row_dir(m, l, vec.to(W.dtype), ref_cols, cols)
        return (R * v[None, :]).sum(1) / R.norm(dim=1).clamp_min(1e-12)

    def _step_head(self, pop, opts, X, probe, ref_idx, perms, teacher):
        L = self.nW - 1; dev = self.device
        self.cache.setdefault(L, [])
        self._apply_cache(pop, opts, L, ref_idx, perms)
        cons, _, sizes = head_consensus_columns(pop, probe, ref_idx=ref_idx,
                                                eps=self.eps, quorum_ratio=self.quorum_ratio)
        n3 = cons.shape[1]
        st_ref = self.pins[(ref_idx, L)]
        pinned = self._units_pinned(st_ref, L)
        todo = ((sizes > 0) & ~pinned).nonzero().flatten()
        out = {"consensus": int((sizes > 0).sum()), "refined_now": 0, "queries_now": 0}
        if todo.numel() == 0:
            return out
        if self.teacher64 is None:
            import copy
            self.teacher64 = copy.deepcopy(teacher).double().to(dev).eval()
            self.teacher64.requires_grad_(False)
        oracle = lambda Xq: self.teacher64(Xq.to(torch.float64))      # noqa: E731
        ref64 = pop[ref_idx].clone().double().eval()
        n_pool = min(20000, len(X))
        X_pool = X[torch.randperm(len(X))[:n_pool]].to(dev, torch.float64)
        res = refine_head(oracle, ref64, cons, X_pool, units=todo,
                          seed=int(torch.randint(0, 10 ** 6, (1,))))
        self.queries += res["queries"]; out["queries_now"] = res["queries"]
        with torch.no_grad():
            for j in res["ok"].nonzero().flatten().tolist():
                self.cache[L].append({"vec": res["cols"][:, j].detach().clone(), "mask": None})
                self._pin_head(ref_idx, pop[ref_idx], opts[ref_idx], len(self.cache[L]) - 1, j)
        out["refined_now"] = int(res["ok"].sum())
        self._apply_cache(pop, opts, L, ref_idx, perms)
        return out

    # ------------------------------------------------------------- rows --
    @torch.no_grad()
    def _step_rows(self, pop, opts, l, ref_idx, perms, ident, fwd_mask):
        dev = self.device; n_out, n_in = self.dims[l + 1], self.dims[l]
        P = len(pop); quorum = math.ceil(self.quorum_ratio * P)
        self.cache.setdefault(l, [])
        self._apply_cache(pop, opts, l, ref_idx, perms)
        # identified input units (reference order); layer 0's inputs are the raw inputs
        in_mask = torch.ones(n_in, dtype=torch.bool, device=dev) if l == 0 else ident[l - 1].clone()
        cols_ref = in_mask.nonzero().flatten()
        if cols_ref.numel() < self.min_ident * n_in:
            return {"consensus": 0, "refined_now": 0, "queries_now": 0,
                    "identified_inputs": int(cols_ref.numel()), "n_inputs": n_in}
        rows = []
        for mi, m in enumerate(pop):
            mc = m.clone(); scale_normalize_(mc)                  # canonical gauge: unit rows below
            W, b = mc.layers[l].weight, mc.layers[l].bias
            pr = perms[mi][l]
            R = W[pr]
            if l > 0:
                R = R[:, perms[mi][l - 1]]
            R = torch.cat([R[:, cols_ref], b[pr][:, None]], 1)
            rows.append(R / R.norm(dim=1, keepdim=True).clamp_min(1e-12))
        U = torch.stack(rows)                                   # (P, n_out, k+1)
        st_ref = self.pins[(ref_idx, l)]
        pinned = self._units_pinned(st_ref, l).clone()
        if fwd_mask is not None:
            pinned |= fwd_mask.to(dev)
        n_cons = 0; new = 0
        # seeds in LOSS order: the first seed whose eps-ball reaches quorum defines
        # the consensus, and its own row is the value. (The largest cluster need not
        # contain the best members: on an uneven committee two 5e-4 members can sit
        # outside a bigger cluster of 2e-2 members, and a cluster mean is dragged by
        # its worst members either way.)
        order = (sorted(range(P), key=lambda mi: self.losses[mi]) if self.losses is not None
                 else list(range(P)))
        for j in range(n_out):
            seed = None
            for r in order:
                mem = self._agree(U, j, r)
                if int(mem.sum()) >= quorum:
                    seed = r; break
            if seed is None:
                continue
            n_cons += 1
            if bool(pinned[j]):
                continue
            c = U[seed, j].clone()
            c = c / c.norm()
            # entries the cluster does NOT agree on stay trainable (not pinned)
            spread = (U[mem, j] - U[seed, j]).abs().amax(0)          # (k+1,)
            ok_entries = spread[:-1] <= self.eps
            vec = torch.zeros(n_in + 1, device=dev, dtype=c.dtype)
            vec[cols_ref] = c[:-1]; vec[n_in] = c[-1]
            mask = torch.zeros(n_in, dtype=torch.bool, device=dev)
            mask[cols_ref[ok_entries]] = True
            self.cache[l].append({"vec": vec, "mask": mask})
            cid = len(self.cache[l]) - 1
            self._pin_row(ref_idx, pop[ref_idx], opts[ref_idx], l, cid, j, None)
            new += 1
        self._apply_cache(pop, opts, l, ref_idx, perms)
        return {"consensus": n_cons, "refined_now": new, "queries_now": 0,
                "identified_inputs": int(in_mask.sum()), "n_inputs": n_in}

    # ------------------------------------------------------------ gauge --
    @torch.no_grad()
    def _in_scales(self, m, l):
        """Live canonical scale of every INPUT unit of layer l in member m, i.e.
        the bottom-up factors align.scale_normalize_ would push into layer l's
        columns (n_k = norms of layer k's rows after scaling its columns by n_{k-1},
        raised to the activation's scale power: 1 for leaky-ReLU, 2 for square).
        Canonical row entries of layer l are W_m[j, i] * n[i]; the member is NOT
        modified (in-place canonicalization is function-preserving but rescales the
        parametrization and wrecks the optimizer's conditioning)."""
        n = None
        p = _scale_power(m) or 1
        for k in range(l):
            W = m.layers[k].weight
            We = W if n is None else W * n[None, :]
            n = We.norm(dim=1).clamp_min(1e-12) ** p
        return n                                               # None for l == 0

    @torch.no_grad()
    def _canon_below(self, m, l):
        """Function-preserving canonical gauge BELOW layer l: bottom-up, unit-norm
        the rows of every hidden layer k < l and push each scale into layer k+1's
        columns (leaky-ReLU homogeneity). Must match align.scale_normalize_, which
        the consensus clones use: the gauge of layer k's rows depends on the push
        from layer k-1, so normalizing layer l-1 alone is a DIFFERENT gauge."""
        p = _scale_power(m) or 1
        for k in range(l):
            Wp, bp = m.layers[k].weight, m.layers[k].bias
            n = Wp.norm(dim=1).clamp_min(1e-12)
            Wp.div_(n[:, None]); bp.div_(n)
            m.layers[k + 1].weight.mul_((n ** p)[None, :])

    @torch.no_grad()
    def project(self, net):
        """Called after every optimizer step: keep hidden layer k unit-norm whenever
        layer k+1 holds pinned rows in this member (a pinned row is expressed in the
        canonical gauge of its inputs, so the inputs' scale is no longer free)."""
        mi = self.by_id.get(id(net))
        if mi is None:
            return
        for l in range(self.nW):
            st = self.pins.get((mi, l))
            if st is not None and st["vals"]:
                self._snap(net, st, l)
    # ------------------------------------------- isolated per-layer consensus --
    @torch.no_grad()
    def report_only(self, pop, X, ref_hint, losses=None):
        """The isolated two-sided consensus report without any pinning (for runs
        that only want the diagnostic): pair every hidden layer by activation to
        the lowest-loss member and call layer_report."""
        dev = self.device; nh = self.nW - 1
        ref_idx = (min(range(len(pop)), key=lambda mi: losses[mi]) if losses is not None else ref_hint)
        probe = self._probe(X)
        Zref = _preacts(pop[ref_idx], probe)
        canon, perms, corrs = [], [], []
        for m in pop:                                   # paired to the reference; polarity aligned by activation (tanh/sigmoid)
            mc, ps, cs = sign_align_clone(m, Zref, probe)
            if _scale_power(m) is not None:            # leaky-ReLU / square: scale gauge (sign already aligned)
                scale_normalize_(mc)
            canon.append(mc); perms.append(ps); corrs.append(cs)
        quorum = math.ceil(self.quorum_ratio * len(pop))
        ident = [(torch.stack([corrs[i][l] for i in range(len(pop))]) > self.id_corr).sum(0) >= quorum
                 for l in range(nh)]
        return self.layer_report(pop, ref_idx, perms, ident, losses, canon=canon)

    @torch.no_grad()
    def layer_report(self, pop, ref_idx, perms, ident, losses=None, canon=None):
        """Teacher-free, frontier-independent consensus of EVERY hidden layer from
        both sides: rows = incoming weights in the canonical gauge of the layer
        below (its unidentified units masked), cols = outgoing weights (columns of
        the next layer) in the canonical gauge, rows of the next layer paired by
        activation. Same cluster/quorum rule and loss-ordered seeds as the pins.
        Returns {hidden layer k: dict(rows, cols, n, ident_in, ident_out)}."""
        P = len(pop); quorum = math.ceil(self.quorum_ratio * P); nh = self.nW - 1
        order = sorted(range(P), key=lambda mi: losses[mi]) if losses is not None else list(range(P))
        if canon is None:                               # (step path) clones in the canonical gauge
            canon = []
            for m in pop:
                mc = m.clone(); scale_normalize_(mc); canon.append(mc)
        # tanh/sigmoid have NO scale gauge: row/column magnitudes are real parameters, so the tests
        # compare RAW weights (inf-norm eps, like the chained counter). Unit-normalising hid 1.0-1.8x
        # row-norm disagreements in layer 1 of the 6-layer tanh run (direction spread 0.007, raw 0.047).
        scale_free = _scale_power(pop[0]) is not None
        def _rown(Rk):
            return Rk / Rk.norm(dim=1, keepdim=True).clamp_min(1e-12) if scale_free else Rk
        def _coln(Ck):
            return Ck / Ck.norm(dim=0, keepdim=True).clamp_min(1e-12) if scale_free else Ck
        def n_cons(U):
            if U.shape[-1] == 0:
                return 0
            cnt = 0
            for j in range(U.shape[1]):
                for r in order:
                    if int(self._agree(U, j, r).sum()) >= quorum:
                        cnt += 1; break
            return cnt
        def cons_mask(U):
            ok = torch.zeros(U.shape[1], dtype=torch.bool, device=U.device)
            if U.shape[-1] == 0:
                return ok
            for j in range(U.shape[1]):
                for r in order:
                    if int(self._agree(U, j, r).sum()) >= quorum:
                        ok[j] = True; break
            return ok
        out = {}
        row_ok = {}; row_dir = {}; row_mag = {}
        # pass 1: rows of every layer (inputs = identified units of the layer below)
        for k in range(nh):
            in_mask = (torch.ones(self.dims[k], dtype=torch.bool, device=self.device) if k == 0
                       else ident[k - 1])
            if int(in_mask.sum()) < self.min_ident * self.dims[k]:
                # inputs not (yet) activation-identified: the Hungarian pairing still exists, so test
                # the rows over ALL paired input columns (mispaired columns can only make rows DISagree,
                # beyond the 3% trim; they cannot create a false consensus). This is what the chained
                # counter does implicitly; without it the isolated view printed 0 for every layer
                # above an unconverged layer 1 during a tanh transition.
                cols_in = torch.arange(self.dims[k], device=self.device)
            else:
                cols_in = in_mask.nonzero().flatten()
            Rraw = []
            for mi, mc in enumerate(canon):
                W, b = mc.layers[k].weight, mc.layers[k].bias
                pr = perms[mi][k]
                Wk = W[pr]
                if k > 0:
                    Wk = Wk[:, perms[mi][k - 1]]
                Rraw.append(torch.cat([Wk[:, cols_in], b[pr][:, None]], 1))
            Rraw = torch.stack(Rraw)
            Rdir = Rraw / Rraw.norm(dim=2, keepdim=True).clamp_min(1e-12)
            # leaky-ReLU: scale is a gauge -> the direction test IS the test. tanh/sigmoid: the
            # full test is on RAW weights (magnitude is a parameter), but direction consensus is
            # reported too -- it is real information (6-layer tanh it 15: all 200 L1 directions
            # agree at 6e-3 vs the black box while magnitudes differ 1.0-1.8x across members).
            row_dir_ok = cons_mask(Rdir)
            row_ok[k] = row_dir_ok if scale_free else cons_mask(Rraw)
            if not scale_free:
                row_dir[k] = row_dir_ok
                # MAGNITUDE consensus (tanh/sigmoid: the row length is a parameter): per unit, the
                # members' row norms within 2% (relative) of a seed's, same quorum / seed order.
                logn = Rraw[:, :, :-1].norm(dim=2).clamp_min(1e-12).log()         # (P, n_units)
                mag = torch.zeros(Rraw.shape[1], dtype=torch.bool, device=self.device)
                for j in range(Rraw.shape[1]):
                    for r in order:
                        if int(((logn[:, j] - logn[r, j]).abs() <= 0.02).sum()) >= quorum:
                            mag[j] = True; break
                row_mag[k] = mag & row_dir_ok                      # a row norm only means something once the direction agrees
        # pass 2: cols of every layer; a next-layer unit that failed ITS row consensus is
        # excluded (a bad row spoils one row but one entry of EVERY column)
        for k in range(nh):
            n_units = self.dims[k + 1]
            in_mask = (torch.ones(self.dims[k], dtype=torch.bool, device=self.device) if k == 0
                       else ident[k - 1])
            rows = int(row_ok[k].sum())
            # outgoing columns: columns of layer k+1 for this layer's units; the next
            # layer's units (rows of W_{k+1}) paired by activation when hidden, fixed
            # output basis when k+1 is the head
            Craw = []
            # output rows of the column test = next-layer units that are identified AND passed their row
            # test; when too few are identified (mid-transition: deeper activations do not match across
            # members while layer-1 magnitudes differ) fall back to ALL next-layer units whose rows passed
            # -- same fallback the row test has; a column over a handful of rows is noise, n/a is worse.
            fallback_out = False
            if k + 1 < nh:
                out_mask = ident[k + 1] & row_ok[k + 1]
                if int(out_mask.sum()) < self.min_ident * self.dims[k + 2]:
                    out_mask = row_ok[k + 1].clone(); fallback_out = True
            else:
                out_mask = None
            for mi, mc in enumerate(canon):
                Wn = mc.layers[k + 1].weight
                pr = perms[mi][k]
                Ck = Wn[:, pr]
                if k + 1 < nh:
                    Ck = Ck[perms[mi][k + 1]]
                    Ck = Ck[out_mask]                              # next-layer units: paired AND in row consensus
                Craw.append(Ck.T)   # (n_units, n_out_id)
            Craw = torch.stack(Craw)
            Cdir = Craw / Craw.norm(dim=2, keepdim=True).clamp_min(1e-12)
            n_out_id = int(out_mask.sum()) if k + 1 < nh else self.dims[-1]
            enough = n_out_id >= self.min_ident * (self.dims[k + 2] if k + 1 < nh else self.dims[-1])
            ident_out_true = int((ident[k + 1] & row_ok[k + 1]).sum()) if k + 1 < nh else self.dims[-1]
            cols_dir = n_cons(Cdir) if enough else 0
            cols = cols_dir if scale_free else (n_cons(Craw) if enough else 0)
            out[k] = dict(rows=rows, cols=cols, n=n_units, ident_in=int(in_mask.sum()), n_in=self.dims[k],
                          ident_out=n_out_id, ident_out_true=ident_out_true, fallback_out=fallback_out, is_head=(k + 1 == nh),
                          n_out=(self.dims[k + 2] if k + 1 < nh else self.dims[-1]),
                          ident_self=int(ident[k].sum()))
            if not scale_free:
                out[k]["rows_dir"] = int(row_dir[k].sum()); out[k]["cols_dir"] = cols_dir; out[k]["rows_mag"] = int(row_mag[k].sum())
        return out

    # ---------------------------------------------------------- logging --
    @torch.no_grad()
    def _solved_eps(self, ref, teacher, probe, l, solved):
        T = teacher.clone().to(ref.layers[0].weight.device).float(); scale_normalize_(T)
        ref = ref.clone(); scale_normalize_(ref)
        Zt = _preacts(T, probe); Zr = _preacts(ref, probe)
        if l == self.nW - 1:
            pt, _ = _act_perm(Zt[-1], Zr[-1])
            Ht = T.layers[-1].weight; Hr = ref.layers[-1].weight[:, pt]
            E = (Ht / Ht.norm(dim=0, keepdim=True) - Hr / Hr.norm(dim=0, keepdim=True)).abs().amax(0)
        else:
            pt, _ = _act_perm(Zt[l], Zr[l])
            Wt = torch.cat([T.layers[l].weight, T.layers[l].bias[:, None]], 1)
            Wr = ref.layers[l].weight[pt]; br = ref.layers[l].bias[pt]
            if l > 0:
                pin, _ = _act_perm(Zt[l - 1], Zr[l - 1]); Wr = Wr[:, pin]
            Wr = torch.cat([Wr, br[:, None]], 1)
            E = (Wt / Wt.norm(dim=1, keepdim=True) - Wr / Wr.norm(dim=1, keepdim=True)).abs().amax(1)
        kr = solved[pt]
        out = {}
        for name, mask in (("solved", kr), ("unsolved", ~kr)):
            out[f"eps_{name}_max"] = float(E[mask].max()) if bool(mask.any()) else None
            out[f"eps_{name}_mean"] = float(E[mask].mean()) if bool(mask.any()) else None
        return out
