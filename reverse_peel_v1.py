"""--reverse-peel: top-down committee peel.

Train the committee as usual, but track consensus on the LAST layer first, in the
head's gauge-correct frame (columns per last-hidden unit, members' units paired by
activation matching, unit-normalized columns, the code's cluster/quorum rule), peel
it, then recurse backwards one layer at a time.

  head  : consensus columns -> EXACT columns by the kink-jump refiner
          (head_refine.refine_head, black-box, no prefix needed) -> pinned in every
          member at that member's own column magnitude (scale = gauge) with the
          gradient masked. Peeled when >= thresh of the columns are pinned.
  L_k   : rows compared in the frame the pinned layer above induces (row index =
          the unit's identity under the head/above-layer pairing, input columns =
          activation pairing of the layer below), unit-normalized rows, same
          cluster/quorum rule -> consensus rows pinned at each member's own row
          magnitude (direction only). NOTE: there is no black-box refiner for a
          row given only the SUFFIX, so below the head the pins are at consensus
          precision (~1e-2); they are a training scaffold, logged against the
          teacher via the act-matched per-layer eps so their effect is visible.
Pinned entries: value written, gradient hook-masked, Adam state zeroed, and
re-written at every check (Adam momentum drift).
"""
import copy
import math
import torch
from align import scale_normalize_, _preacts
from head_refine import head_consensus_columns, refine_head, _act_perm


class ReversePeel:
    def __init__(self, dims, cfg, device):
        self.dims = list(dims)
        self.nW = len(dims) - 1                 # number of weight layers; head = nW-1
        self.frontier = self.nW - 1
        self.eps = cfg.reverse_eps
        self.thresh = cfg.reverse_thresh
        self.quorum_ratio = cfg.cluster_quorum
        self.pin_cos = 0.9                      # member's own unit must agree with the pin
        self.device = device
        self.pins = {}                          # (member idx, layer) -> dict(keep, hooked, vals)
        self.exact = {}                         # layer -> {ref unit j: unit vector (ref frame)}
        self.ref_id = None
        self.n_forced = 0                       # pins injected into a member whose unit disagreed
        self.teacher64 = None
        self.queries = 0
        self.done = False
        self.history = []

    # ------------------------------------------------------------- pins --
    def _pin_state(self, mi, m, l, n_units):
        key = (mi, l)
        st = self.pins.get(key)
        if st is None or st["hooked"] != id(m):
            # new member object (restart / combine / replace): its units are reindexed,
            # so old pins are meaningless -> start clean; consensus re-forms and re-pins
            keep = torch.zeros(n_units, dtype=torch.bool, device=self.device)
            st = {"keep": keep, "hooked": id(m), "vals": {}, "applied": set()}
            W, b = m.layers[l].weight, m.layers[l].bias
            if l == self.nW - 1:                # head: mask COLUMNS
                W.register_hook(lambda g, k=keep: g * (~k).to(g.dtype).unsqueeze(0))
            else:                               # hidden: mask ROWS (+ bias)
                W.register_hook(lambda g, k=keep: g * (~k).to(g.dtype).unsqueeze(1))
                b.register_hook(lambda g, k=keep: g * (~k).to(g.dtype))
            self.pins[key] = st
        return st

    @torch.no_grad()
    def _rewrite_pins(self, pop):
        for (mi, l), st in self.pins.items():
            if mi >= len(pop):
                continue
            m = pop[mi]
            for j, (w, b) in st["vals"].items():
                if l == self.nW - 1:
                    m.layers[l].weight[:, j] = w.to(m.layers[l].weight.dtype)
                else:
                    m.layers[l].weight[j] = w.to(m.layers[l].weight.dtype)
                    m.layers[l].bias[j] = b.to(m.layers[l].bias.dtype)

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

    # ------------------------------------------------------------- step --
    def step(self, pop, opts, teacher, X, ref_idx, t, forward_masks=None):
        """One consensus check: EVERY layer from the head down to the current
        frontier refines/pins its still-unsolved units (stragglers are retried at
        every check); the frontier moves down only when a layer is complete
        (pinned >= reverse_thresh * units, default 1.0 = all). Returns per-layer info."""
        if self.done:
            return None
        dev = self.device
        if self.ref_id != id(pop[ref_idx]):     # reference member replaced: its units are unlabeled,
            if self.ref_id is not None:         # so define the frame BY the cache (column j -> unit j)
                self._inject_identity(pop[ref_idx], opts[ref_idx], ref_idx)
            self.ref_id = id(pop[ref_idx])
        self._rewrite_pins(pop)
        n_probe = min(4096, len(X))
        probe = X[torch.randperm(len(X))[:n_probe]].to(dev, torch.float32)
        Z = [_preacts(m, probe) for m in pop]
        nh = self.nW - 1
        perms = [[_act_perm(Z[ref_idx][l], Z[i][l])[0] for l in range(nh)] for i in range(len(pop))]
        info = {"iter": t + 1, "frontier": self.frontier, "layers": {}}
        self.forward_masks = forward_masks or {}
        for l in range(self.nW - 1, self.frontier - 1, -1):
            n_units = self.dims[l] if l == self.nW - 1 else self.dims[l + 1]
            st_ref = self._pin_state(ref_idx, pop[ref_idx], l, n_units)
            fwd = self.forward_masks.get(l)
            if fwd is not None and l != self.nW - 1:
                st_ref["keep"] |= fwd.to(st_ref["keep"].device)   # forward-pinned rows are exact: solved
            if int(st_ref["keep"].sum()) >= n_units:
                info["layers"][l] = {"consensus": n_units, "n_units": n_units, "refined_now": 0,
                                     "queries_now": 0, "pinned_ref": n_units, "complete": True}
                continue
            if l == self.nW - 1:
                li = self._step_head(pop, opts, teacher, X, probe, ref_idx, perms)
                # stragglers: unsolved units + how often the reference member activates them
                keep = self.pins[(ref_idx, l)]["keep"]
                uns = (~keep).nonzero().flatten()
                act = (Z[ref_idx][-1][:, uns] > 0).float().mean(0)
                li["unsolved"] = uns.tolist()
                li["unsolved_act_frac"] = [round(float(a), 3) for a in act]
            else:
                li = self._step_rows(pop, opts, l, ref_idx, perms)
            li["pinned_ref"] = int(self.pins[(ref_idx, l)]["keep"].sum())
            li["complete"] = li["pinned_ref"] >= self.thresh * n_units
            li.update(self._solved_eps(pop[ref_idx], teacher, probe, l, self.pins[(ref_idx, l)]["keep"], perms, ref_idx))
            info["layers"][l] = li
        # frontier advance: only through COMPLETE layers
        while self.frontier >= 0 and info["layers"].get(self.frontier, {}).get("complete", False):
            self.frontier -= 1
            info["advanced_to"] = self.frontier
        if self.frontier < 0:
            self.done = True
        self.history.append(info)
        return info

    def _step_head(self, pop, opts, teacher, X, probe, ref_idx, perms):
        dev = self.device; L = self.nW - 1
        cons, _, sizes = head_consensus_columns(pop, probe, ref_idx=ref_idx,
                                                eps=self.eps, quorum_ratio=self.quorum_ratio)
        n3 = cons.shape[1]
        st_ref = self._pin_state(ref_idx, pop[ref_idx], L, n3)
        todo = ((sizes > 0) & ~st_ref["keep"]).nonzero().flatten()
        out = {"consensus": int((sizes > 0).sum()), "n_units": n3, "refined_now": 0, "queries_now": 0}
        if todo.numel() == 0:
            self._apply_head(pop, opts, perms, L, n3)
            return out
        if self.teacher64 is None:
            self.teacher64 = copy.deepcopy(teacher).double().to(dev).eval()
            self.teacher64.requires_grad_(False)
        oracle = lambda Xq: self.teacher64(Xq.to(torch.float64))      # noqa: E731
        ref64 = pop[ref_idx].clone().double().eval()
        n_pool = min(20000, len(X))
        X_pool = X[torch.randperm(len(X))[:n_pool]].to(dev, torch.float64)
        res = refine_head(oracle, ref64, cons, X_pool, units=todo, seed=int(torch.randint(0, 10**6, (1,))))
        self.queries += res["queries"]; out["queries_now"] = res["queries"]
        solved = res["ok"].nonzero().flatten()
        cache = self.exact.setdefault(L, {})
        for j in solved.tolist():
            cache[j] = res["cols"][:, j].detach().clone()
        out["refined_now"] = int(solved.numel())
        self._apply_head(pop, opts, perms, L, n3)
        return out

    @torch.no_grad()
    def _apply_head(self, pop, opts, perms, L, n3):
        """Pin every cached exact column into every member that lacks it (at the
        member's own column magnitude; only if the member's column already agrees)."""
        cache = self.exact.get(L, {})
        with torch.no_grad():
            for mi, m in enumerate(pop):
                st = self._pin_state(mi, m, L, n3)
                W = m.layers[L].weight
                for j, u in cache.items():
                    if j in st["applied"]:              # this column already sits in one of my units
                        continue
                    jm = int(perms[mi][L - 1][j])
                    if bool(st["keep"][jm]):            # unit already holds another pinned column
                        continue
                    st["applied"].add(j)
                    u = u.to(W.dtype)
                    col = W[:, jm]
                    # FORCED pin (peel semantics): the solved column is injected whether or
                    # not the member's unit already agreed; a disagreeing member's unit is
                    # thereby relabeled and its tail retrains under the fixed head. Count it.
                    if float((col * u).sum()) / float(col.norm().clamp_min(1e-12)) < self.pin_cos:
                        self.n_forced += 1
                    newcol = u * col.norm().clamp_min(1e-6)
                    W[:, jm] = newcol
                    st["keep"][jm] = True
                    st["vals"][jm] = (newcol.detach().clone(), None)
                    self._zero_adam(opts[mi], W, jm, column=True)

    @torch.no_grad()
    def _step_rows(self, pop, opts, l, ref_idx, perms):
        """Hidden layer l (0-based weight index): rows in the frame induced from above."""
        dev = self.device
        n_out, n_in = self.dims[l + 1], self.dims[l]
        P = len(pop); quorum = math.ceil(self.quorum_ratio * P)
        rows = []
        for mi, m in enumerate(pop):
            W = m.layers[l].weight; b = m.layers[l].bias
            pr = perms[mi][l]                                   # ref unit i -> member unit pr[i]
            R = W[pr]
            if l > 0:
                R = R[:, perms[mi][l - 1]]                      # input columns in ref's order
            R = torch.cat([R, b[pr][:, None]], 1)
            rows.append(R / R.norm(dim=1, keepdim=True).clamp_min(1e-12))
        U = torch.stack(rows)                                   # (P, n_out, n_in+1)
        st_ref = self._pin_state(ref_idx, pop[ref_idx], l, n_out)
        n_cons = 0; pinned_now = 0
        # re-apply cached consensus rows to members that lack them (e.g. replaced members)
        for j, c in self.exact.get(l, {}).items():
            for mi, m in enumerate(pop):
                st = self._pin_state(mi, m, l, n_out)
                if j in st["applied"]:
                    continue
                jm = int(perms[mi][l][j])
                if bool(st["keep"][jm]):
                    continue
                st["applied"].add(j)
                W = m.layers[l].weight; b = m.layers[l].bias
                v = torch.empty(n_in + 1, device=dev, dtype=W.dtype)
                if l > 0:
                    v[perms[mi][l - 1]] = c[:n_in].to(W.dtype)
                else:
                    v[:n_in] = c[:n_in].to(W.dtype)
                v[n_in] = c[n_in].to(W.dtype)
                cur = torch.cat([W[jm], b[jm:jm + 1]])
                if float((cur * v).sum()) / float(cur.norm().clamp_min(1e-12)) < self.pin_cos:
                    self.n_forced += 1
                mag = cur.norm().clamp_min(1e-6); W[jm] = v[:n_in] * mag; b[jm] = v[n_in] * mag
                st["keep"][jm] = True; st["vals"][jm] = (W[jm].detach().clone(), b[jm].detach().clone())
                self._zero_adam(opts[mi], W, jm, column=False); self._zero_adam(opts[mi], b, jm, column=False)
        for j in range(n_out):
            best = None
            for r in range(P):
                mem = (U[:, j] - U[r, j]).abs().amax(-1) <= self.eps
                k = int(mem.sum())
                if k >= quorum and (best is None or k > best[0]):
                    best = (k, mem)
            if best is None:
                continue
            n_cons += 1
            if bool(st_ref["keep"][j]):          # already pinned (by us or by the forward peel)
                continue
            c = U[best[1], j].mean(0); c = c / c.norm()
            self.exact.setdefault(l, {})[j] = c.detach().clone()
            for mi, m in enumerate(pop):
                st = self._pin_state(mi, m, l, n_out)
                if j in st["applied"]:
                    continue
                jm = int(perms[mi][l][j])
                if bool(st["keep"][jm]):
                    continue
                st["applied"].add(j)
                W = m.layers[l].weight; b = m.layers[l].bias
                v = torch.empty(n_in + 1, device=dev, dtype=W.dtype)
                if l > 0:
                    v[perms[mi][l - 1]] = c[:n_in].to(W.dtype)   # back to member's input order
                else:
                    v[:n_in] = c[:n_in].to(W.dtype)
                v[n_in] = c[n_in].to(W.dtype)
                cur = torch.cat([W[jm], b[jm:jm + 1]])
                if float((cur * v).sum()) / float(cur.norm().clamp_min(1e-12)) < self.pin_cos:
                    self.n_forced += 1
                mag = cur.norm().clamp_min(1e-6)
                W[jm] = v[:n_in] * mag; b[jm] = v[n_in] * mag
                st["keep"][jm] = True
                st["vals"][jm] = (W[jm].detach().clone(), b[jm].detach().clone())
                self._zero_adam(opts[mi], W, jm, column=False)
                self._zero_adam(opts[mi], b, jm, column=False)
            pinned_now += 1
        return {"consensus": n_cons, "n_units": n_out, "refined_now": pinned_now, "queries_now": 0}

    @torch.no_grad()
    def _solved_eps(self, ref, teacher, probe, l, keep, perms, ref_idx):
        """Direction error of the reference member's units of layer l vs the TEACHER
        (unit vectors, gauge-correct frame: head = columns paired by last-hidden
        activation matching; hidden = rows with input columns paired by the layer
        below's activation matching). Reported separately for solved (pinned) and
        unsolved units. Logging only -- the teacher is never used for decisions."""
        T = teacher.clone().to(ref.layers[0].weight.device).float()
        Zt = _preacts(T, probe); Zr = _preacts(ref, probe)
        if l == self.nW - 1:
            pt, _ = _act_perm(Zt[-1], Zr[-1])                 # teacher unit i <-> ref unit pt[i]
            Ht = T.layers[-1].weight; Hr = ref.layers[-1].weight[:, pt]
            E = (Ht / Ht.norm(dim=0, keepdim=True) - Hr / Hr.norm(dim=0, keepdim=True)).abs().amax(0)
            kr = keep[pt]
        else:
            pt, _ = _act_perm(Zt[l], Zr[l])
            Wt = torch.cat([T.layers[l].weight, T.layers[l].bias[:, None]], 1)
            Wr = ref.layers[l].weight[pt]; br = ref.layers[l].bias[pt]
            if l > 0:
                pin, _ = _act_perm(Zt[l - 1], Zr[l - 1])
                Wr = Wr[:, pin]
            Wr = torch.cat([Wr, br[:, None]], 1)
            E = (Wt / Wt.norm(dim=1, keepdim=True) - Wr / Wr.norm(dim=1, keepdim=True)).abs().amax(1)
            kr = keep[pt]
        out = {}
        for name, mask in (("solved", kr), ("unsolved", ~kr)):
            if bool(mask.any()):
                out[f"eps_{name}_max"] = float(E[mask].max()); out[f"eps_{name}_mean"] = float(E[mask].mean())
            else:
                out[f"eps_{name}_max"] = None; out[f"eps_{name}_mean"] = None
        return out

    @torch.no_grad()
    def _inject_identity(self, m, opt, mi):
        """A fresh/replaced REFERENCE member has unlabeled units: inject every cached
        value under the identity assignment (cached unit j -> its unit j), which makes
        the member carry the frame the cache was expressed in. Grad-masked as usual."""
        L = self.nW - 1
        for l, cache in self.exact.items():
            n_units = self.dims[l] if l == L else self.dims[l + 1]
            st = self._pin_state(mi, m, l, n_units)
            W = m.layers[l].weight; b = m.layers[l].bias
            for j, v in cache.items():
                st["applied"].add(j)
                if l == L:
                    newcol = v.to(W.dtype) * W[:, j].norm().clamp_min(1e-6)
                    W[:, j] = newcol; st["keep"][j] = True; st["vals"][j] = (newcol.clone(), None)
                    self._zero_adam(opt, W, j, column=True)
                else:
                    n_in = self.dims[l]
                    mag = torch.cat([W[j], b[j:j + 1]]).norm().clamp_min(1e-6)
                    W[j] = v[:n_in].to(W.dtype) * mag; b[j] = v[n_in].to(b.dtype) * mag
                    st["keep"][j] = True; st["vals"][j] = (W[j].clone(), b[j].clone())
                    self._zero_adam(opt, W, j, column=False); self._zero_adam(opt, b, j, column=False)
