"""Target the unsolved units: hold the certified weights and aim the queries at inputs
where the uncertified units are ACTIVE and the members DISAGREE about them.

  map     : teacher-free certification map (consensus_eps.entry_coverage with
            teacher=None): per matrix, certified entries and the consensus values in the
            reference member's frame (canonical gauge); refreshed at every log point.
  targets : hidden units whose incoming row is NOT certified (uncertified from below)
            or whose outgoing column is NOT certified (uncertified from above).
  hold    : 'pin' = direction-only snap of every certified row/column after each optimizer
            step (reverse_peel.ReversePeel machinery, driven by the map instead of a frontier);
            'l1'  = lambda * sum |normalize(certified part of the row) - guess direction|_1
            added to the fit loss (zero at the guess; the data can still move an entry).
  queries : a fraction of each iteration's queries is produced by maximizing, over the
            input, the variance across members of the targets' pre-activations where the
            targets are active (soft gate), from random starts plus buffer points where
            the reference member's targets fire. Hit rate (targets active) is logged.
"""
import math
import torch
from align import _preacts
from reverse_peel import ReversePeel
from consensus_eps import entry_coverage


def _preacts_grad(net, X):
    """Differentiable hidden pre-activations (align._preacts is no_grad)."""
    zs = []; h = X
    for layer in net.layers[:-1]:
        z = layer(h); zs.append(z); h = net.act(z)
    return zs


class Targeter:
    def __init__(self, pop, cfg, device, mode="pin", lam=1.0, frac=0.5, losses=None):
        self.dims = list(pop[0].dims); self.nW = len(self.dims) - 1; self.nh = self.nW - 1
        self.cfg = cfg; self.device = device; self.mode = mode; self.lam = lam; self.frac = frac
        self.rp = ReversePeel(self.dims, cfg, device)
        self.by_id = {}
        self.prev_cert = None
        self.last_hit = None
        self.masks = {}; self.hooked = set(); self.pin_cols = {}
        self.l1 = {}; self.nets = []                      # proximal-L1 state per (member, layer) / head
        self.forced = []                                  # (layer, unit) forced to be targets at the first refresh
        spec = getattr(cfg, "target_units", "") or ""
        for part in spec.split(";"):
            if ":" in part:
                l, us = part.split(":"); self.forced += [(int(l) - 1, int(u)) for u in us.split(",") if u.strip()]
        self.refresh(pop, losses=losses, opts=None)       # losses pick the reference member (best); None -> m0

    # ------------------------------------------------------------ map --
    @torch.no_grad()
    def refresh(self, pop, losses, opts=None):
        dev = self.device
        losses = list(losses) if losses is not None else [0.0] * len(pop)
        torch.manual_seed(int(torch.randint(0, 10 ** 6, (1,))))
        cov = entry_coverage(pop, None, losses, self.cfg, dev, ref=getattr(self, "fixed_ref", None),
                             cert_gauge=True, sticky=self.prev_cert if self.mode == "pin" else None,
                             value=getattr(self.cfg, "target_value", "seed"))   # 'seed' (reference's entry) or quorum 'mean'
        self.cov = cov; self.perms = cov[0]["perms"]; self.ref = cov[0]["ref"]
        self.fixed_ref = self.ref                        # keep the same reference member for the whole run
        self.by_id = {id(m): mi for mi, m in enumerate(pop)}
        # targets: hidden units whose OWN incoming row is uncertified; for the last hidden
        # layer also units whose head column is uncertified. (A unit's outgoing column into
        # an unsolved next layer is that layer's problem -- counting it here pulled every
        # layer-3 unit into the target set when W4 was unsolved.)
        self.targets = []
        cert = {}
        for l in range(self.nh):
            ok = cov[l]["row_cov"].any(1)
            if l == self.nh - 1:
                ok = ok & cov[l + 1]["col_cov"].any(0)
            if self.mode == "pin" and self.prev_cert is not None:
                ok = ok | self.prev_cert[l]              # STICKY: a pinned unit is frozen, its certificate stands; never released
            if self.prev_cert is None and self.forced:   # first refresh: forced targets are NOT certified (and get re-drawn)
                for (fl, fj) in self.forced:
                    if fl == l:
                        ok[fj] = False
            cert[l] = ok
            for j in (~ok).nonzero().flatten().tolist():
                self.targets.append((l, j))
        newly = (sum(int((cert[l] & ~self.prev_cert[l]).sum()) for l in range(self.nh))
                 if self.prev_cert is not None else 0)
        lost = (sum(int((~cert[l] & self.prev_cert[l]).sum()) for l in range(self.nh))
                if self.prev_cert is not None else 0)
        self.prev_cert = cert
        # hold bookkeeping, VECTORIZED per (member, layer): certified rows u, a 2D mask over
        # the member's input columns, the canonical consensus values in the member's column
        # order (+bias); head: certified columns and their unit directions. The member's
        # live input scales convert canonical -> member gauge at hold time (one chain of
        # matmuls per layer, not per row).
        self.hold_rows = {}; self.hold_head = {}
        if self.forced and not getattr(self, "_forced_applied", False):      # forced targets: not held, not pinned
            for (fl, fj) in self.forced:
                cov[fl]["row_cov"][fj] = False
                if fl == self.nh - 1:
                    cov[fl + 1]["col_cov"][:, fj] = False
                if fl + 1 < self.nW:
                    cov[fl + 1]["col_cov"][:, fj] = False if fl + 1 == self.nW - 1 else cov[fl + 1]["col_cov"][:, fj]
            self._forced_applied = True
        for mi, m in enumerate(pop):
            for k in range(self.nW):
                is_head = (k == self.nW - 1)
                if not is_head:
                    rc, V, bv = cov[k]["row_cov"], cov[k]["V"], cov[k]["b"]
                    rows_ref = rc.any(1).nonzero().flatten()
                    if rows_ref.numel() == 0:
                        continue
                    u = self.perms[mi][k][rows_ref]                             # member units
                    n_in = self.dims[k]
                    mask_ref = rc[rows_ref]                                     # (R, n_in) ref col order
                    if k > 0:
                        inv = torch.empty(n_in, dtype=torch.long, device=dev); inv[self.perms[mi][k - 1]] = torch.arange(n_in, device=dev)
                        # member column c holds ref column inv[c]
                        mask_m = mask_ref[:, inv]; Vm = V[rows_ref][:, inv]
                    else:
                        mask_m = mask_ref; Vm = V[rows_ref]
                    self.hold_rows[(mi, k)] = dict(u=u, mask=mask_m, V=Vm, b=bv[rows_ref])
                else:
                    cc, V = cov[k]["col_cov"].any(0), cov[k]["V"]
                    cols_ref = cc.nonzero().flatten()
                    if cols_ref.numel() == 0:
                        continue
                    u = self.perms[mi][k - 1][cols_ref]
                    D = V[:, cols_ref]; D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-12)
                    self.hold_head[mi] = dict(u=u, D=D)
        # PIN = freeze: the member's OWN certified entries get a zero gradient (hook) and their
        # Adam state is zeroed once, so they never move; zero per-step cost. Masks are updated
        # in place at every refresh (newly certified entries frozen, lost ones released).
        if self.mode == "pin":
            for mi, m in enumerate(pop):
                for k in range(self.nW):
                    W, b = m.layers[k].weight, m.layers[k].bias
                    key = (mi, k)
                    if key not in self.masks or self.masks[key][0].shape != W.shape:
                        self.masks[key] = (torch.zeros_like(W, dtype=torch.bool), torch.zeros_like(b, dtype=torch.bool))
                    kw, kb = self.masks[key]
                    new_w = torch.zeros_like(kw); new_b = torch.zeros_like(kb)
                    if k < self.nW - 1:
                        h = self.hold_rows.get((mi, k))
                        if h is not None:
                            # entry (u, c) of a certified row u is frozen iff input c was already certified
                            # when row u was FIRST pinned, or the entry itself is agreed now (row_cov).
                            # An input certified later (e.g. after a reinit) has entries above it that
                            # were never learned; freezing them whole-row would lock garbage (it did:
                            # column 47, error 0.23 vs 0.005, 163 -> 4 W4 rows within tolerance).
                            new_b[h["u"]] = True
                            if k == 0:
                                new_w[h["u"]] = True
                            else:
                                cin = cov[k - 1]["row_cov"].any(1)           # (ref order) certified units of layer k-1
                                cin_m = torch.zeros(W.shape[1], dtype=torch.bool, device=dev); cin_m[self.perms[mi][k - 1]] = cin
                                pc = self.pin_cols.get(key)
                                if pc is None or pc.shape != W.shape:
                                    pc = torch.zeros_like(W, dtype=torch.bool); self.pin_cols[key] = pc
                                first = h["u"][~kw[h["u"]].any(1)]           # rows pinned for the first time now
                                pc[first] = cin_m[None, :]
                                agreed = torch.zeros_like(W, dtype=torch.bool); agreed[h["u"]] = h["mask"]
                                new_w[h["u"]] = pc[h["u"]] | agreed[h["u"]]
                    else:
                        hh = self.hold_head.get(mi)
                        if hh is not None:
                            new_w[:, hh["u"]] = True
                    new_w |= kw; new_b |= kb                            # never release a pin
                    newly_mask = new_w & ~kw
                    kw.copy_(new_w); kb.copy_(new_b)
                    if (id(m), k) not in self.hooked:                   # one hook per parameter, reads the live mask
                        W.register_hook(lambda g, k_=kw: g * (~k_).to(g.dtype))
                        b.register_hook(lambda g, k_=kb: g * (~k_).to(g.dtype))
                        self.hooked.add((id(m), k))
                    if opts is not None:
                        st = opts[mi].state.get(W)
                        if st:
                            for nm in ("exp_avg", "exp_avg_sq"):
                                if nm in st: st[nm][newly_mask] = 0
                        stb = opts[mi].state.get(b)
                        if stb and k < self.nW - 1:
                            for nm in ("exp_avg", "exp_avg_sq"):
                                if nm in stb: stb[nm][new_b & ~kb] = 0
        # L1 = PROXIMAL hold (exact-penalty semantics). A gradient penalty lam*|dir - anchor|_1 cannot
        # work under Adam: the subgradient flips sign at the anchor, Adam normalises its size away and
        # the momentum limit cycle leaves every entry chattering ~10*lr (0.01-0.05) from the anchor,
        # above the 0.02 certification tolerance, for ANY lam (measured: lam=5 lost 21 rows at it 1).
        # Proximal rule, per certified entry: AT the anchor the L1 subgradient absorbs any data
        # gradient |g| <= lam (gradient zeroed, entry re-snapped after the step); |g| > lam releases
        # the entry with the shrunk gradient sign(g)(|g|-lam); an entry off the anchor gets the pull
        # lam*sign(w - a) and is snapped the moment it crosses. lam is in data-gradient units
        # (mean 1e-3..9e-3, max 0.2..2 on certified entries of the 6-layer run).
        if self.mode == "l1":
            self.nets = list(pop)
            for mi, m in enumerate(pop):
                for k in range(self.nW):
                    key = (mi, k)
                    W, b = m.layers[k].weight, m.layers[k].bias
                    h = self.hold_rows.get(key) if k < self.nW - 1 else self.hold_head.get(mi)
                    if h is None:
                        self.l1.pop(key, None); continue
                    if k < self.nW - 1:
                        u, mask, d = self._row_targets(m, mi, k)
                        st = self.l1.get(key)
                        R = len(u)
                        if st is None or st["u"].shape != u.shape or not bool((st["u"] == u).all()):
                            st = dict(u=u, mask=mask, d=d, at_w=torch.zeros(R, self.dims[k], dtype=torch.bool, device=self.device),
                                      at_b=torch.zeros(R, dtype=torch.bool, device=self.device), sgn_w=None, sgn_b=None)
                        else:
                            st.update(u=u, mask=mask, d=d)
                    else:
                        u, D = h["u"], h["D"]
                        st = self.l1.get(key)
                        if st is None or st["u"].shape != u.shape or not bool((st["u"] == u).all()):
                            st = dict(u=u, D=D, at=torch.zeros(self.dims[-1], len(u), dtype=torch.bool, device=self.device), sgn=None)
                        else:
                            st.update(u=u, D=D)
                    self.l1[key] = st
                    if (id(m), k) not in self.hooked:
                        W.register_hook(lambda g, key=key: self._l1_hook(key, g, bias=False))
                        b.register_hook(lambda g, key=key: self._l1_hook(key, g, bias=True))
                        self.hooked.add((id(m), k))
        self.stats = dict(n_targets=len(self.targets), per_layer=[int((~cert[l]).sum()) for l in range(self.nh)],
                          newly_certified=newly, lost=lost, ref=self.ref)
        return self.stats

    # ----------------------------------------------------------- hold --
    def _row_targets(self, net, mi, k):
        """(u, mask, d) for member mi, layer k: certified units, 2D mask (member col order)
        and the guess DIRECTION (identified entries + bias, unit norm) in the live gauge."""
        h = self.hold_rows.get((mi, k))
        if h is None:
            return None
        n = self.cov[k]["in_scales"][mi]                                   # the gauge V was certified in (None for k==0)
        Dw = h["V"] if n is None else h["V"] / n[None, :]
        Dw = Dw * h["mask"]
        d = torch.cat([Dw, h["b"][:, None]], 1)
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return h["u"], h["mask"], d

    @torch.no_grad()
    def _anchor(self, key):
        """Raw-unit anchors for (member, layer) key: rows -> (a_w (R,n_in), a_b (R,)); head -> (n_out, C)."""
        mi, k = key; m = self.nets[mi]; st = self.l1[key]
        W, b = m.layers[k].weight, m.layers[k].bias
        if k < self.nW - 1:
            u, mask, d = st["u"], st["mask"], st["d"]
            cur = torch.cat([W[u] * mask, b[u][:, None]], 1)
            a = d * cur.norm(dim=1, keepdim=True)
            return a[:, :-1], a[:, -1]
        u, D = st["u"], st["D"]
        return D * W[:, u].norm(dim=0, keepdim=True), None

    def _l1_hook(self, key, g, bias):
        st = self.l1.get(key)
        if st is None:
            return g
        mi, k = key; lam = self.lam
        with torch.no_grad():
            if k < self.nW - 1:
                a_w, a_b = self._anchor(key); u, mask = st["u"], st["mask"]; W, b = self.nets[mi].layers[k].weight, self.nets[mi].layers[k].bias
                if not bias:
                    gu = g[u]; delta = W[u] - a_w; at = st["at_w"]
                    gn = torch.where(at, gu.sign() * (gu.abs() - lam).clamp_min(0), gu + lam * delta.sign())
                    gn = torch.where(mask, gn, gu)
                    st["rel_w"] = at & (gu.abs() > lam) & mask; st["sgn_w"] = delta.sign()
                    g = g.clone(); g[u] = gn
                else:
                    gu = g[u]; delta = b[u] - a_b; at = st["at_b"]
                    gn = torch.where(at, gu.sign() * (gu.abs() - lam).clamp_min(0), gu + lam * delta.sign())
                    st["rel_b"] = at & (gu.abs() > lam); st["sgn_b"] = delta.sign()
                    g = g.clone(); g[u] = gn
            else:
                if bias:
                    return g
                a, _ = self._anchor(key); u = st["u"]; W = self.nets[mi].layers[k].weight
                gu = g[:, u]; delta = W[:, u] - a; at = st["at"]
                gn = torch.where(at, gu.sign() * (gu.abs() - lam).clamp_min(0), gu + lam * delta.sign())
                st["rel"] = at & (gu.abs() > lam); st["sgn"] = delta.sign()
                g = g.clone(); g[:, u] = gn
        return g

    @torch.no_grad()
    def hold(self, net):
        """pin: nothing (frozen via gradient masks). l1: proximal snap after the optimizer step --
        entries that stayed at the anchor (|g| <= lam) are reset there (Adam momentum would drift
        them); entries that crossed their anchor since the hook arrive at it."""
        if self.mode != "l1":
            return
        mi = self.by_id.get(id(net))
        if mi is None:
            return
        for k in range(self.nW):
            key = (mi, k); st = self.l1.get(key)
            if st is None or st.get("sgn_w" if k < self.nW - 1 else "sgn") is None:
                continue
            if k < self.nW - 1:
                a_w, a_b = self._anchor(key); u, mask = st["u"], st["mask"]
                W, b = net.layers[k].weight, net.layers[k].bias
                dw = W[u] - a_w; crossed = (dw.sign() != st["sgn_w"]) & (st["sgn_w"] != 0)
                at = ((st["at_w"] & ~st["rel_w"]) | crossed) & mask
                W[u] = torch.where(at, a_w, W[u]); st["at_w"] = at
                db = b[u] - a_b; crossed_b = (db.sign() != st["sgn_b"]) & (st["sgn_b"] != 0)
                at_b = (st["at_b"] & ~st["rel_b"]) | crossed_b
                b[u] = torch.where(at_b, a_b, b[u]); st["at_b"] = at_b
            else:
                a, _ = self._anchor(key); u = st["u"]; W = net.layers[k].weight
                d = W[:, u] - a; crossed = (d.sign() != st["sgn"]) & (st["sgn"] != 0)
                at = (st["at"] & ~st["rel"]) | crossed
                W[:, u] = torch.where(at, a, W[:, u]); st["at"] = at

    @torch.no_grad()
    def held_fraction(self):
        """fraction of certified entries currently sitting exactly at their anchor (l1 mode)."""
        n = 0; tot = 0
        for key, st in self.l1.items():
            if "at_w" in st:
                n += int((st["at_w"] & st["mask"]).sum()); tot += int(st["mask"].sum())
            else:
                n += int(st["at"].sum()); tot += st["at"].numel()
        return n / max(tot, 1)

    @torch.no_grad()
    def _hold_snap_unused(self, net):
        mi = self.by_id.get(id(net))
        if mi is None:
            return
        for k in range(self.nW - 1):
            rt = self._row_targets(net, mi, k)
            if rt is None:
                continue
            u, mask, d = rt
            W, b = net.layers[k].weight, net.layers[k].bias
            cur = torch.cat([W[u] * mask, b[u][:, None]], 1)
            r = torch.maximum((cur * d).sum(1), 0.1 * cur.norm(dim=1))
            Wn = d[:, :-1] * r[:, None]
            W[u] = torch.where(mask, Wn, W[u]); b[u] = d[:, -1] * r
        hh = self.hold_head.get(mi)
        if hh is not None:
            W = net.layers[-1].weight; u, D = hh["u"], hh["D"]
            col = W[:, u]
            r = torch.maximum((col * D).sum(0), 0.1 * col.norm(dim=0))
            W[:, u] = D * r[None, :]

    def penalty(self, net):
        """L1 anchor on the certified parts of the rows (and head columns), direction only."""
        if self.mode != "l1" or True:                   # proximal hold replaces the gradient penalty (see refresh)
            return None
        mi = self.by_id.get(id(net))
        if mi is None:
            return None
        tot = 0.0; n = 0
        for k in range(self.nW - 1):
            rt = self._row_targets(net, mi, k)
            if rt is None:
                continue
            u, mask, d = rt
            W, b = net.layers[k].weight, net.layers[k].bias
            cur = torch.cat([W[u] * mask, b[u][:, None]], 1)
            cur = cur / cur.norm(dim=1, keepdim=True).clamp_min(1e-12)
            tot = tot + (cur - d).abs().sum(); n += len(u)
        hh = self.hold_head.get(mi)
        if hh is not None:
            W = net.layers[-1].weight; u, D = hh["u"], hh["D"]
            col = W[:, u]; col = col / col.norm(dim=0, keepdim=True).clamp_min(1e-12)
            tot = tot + (col - D).abs().sum(); n += len(u)
        if n == 0:
            return None
        # exact-penalty rule: the L1 subgradient per entry is lam (PER UNIT, not divided by the
        # number of held units); it must exceed the data gradient on certified entries (measured
        # mean 1e-3..9e-3, max 0.2..2 on the 6-layer run) for the minimiser to sit AT the anchor.
        return self.lam * tot

    # ---------------------------------------------------- features --
    def feats(self, pop):
        """Per-member callables I -> (q, n_targets): the members' ACTIVATIONS of the target
        units (paired to the reference), concatenated over layers. Fed to gen_queries as the
        vectors whose disagreement the regular sampler maximizes."""
        by_layer = {}
        for (l, j) in self.targets:
            by_layer.setdefault(l, []).append(j)
        layers = sorted(by_layer)
        js = {l: torch.tensor(by_layer[l], device=self.device) for l in layers}
        fns = []
        for mi, m in enumerate(pop):
            cols = {l: self.perms[mi][l][js[l]] for l in layers}
            def f(I, m=m, cols=cols):
                zs = _preacts_grad(m, I)
                return torch.cat([m.act(zs[l])[:, cols[l]] for l in layers], 1)
            fns.append(f)
        return fns

    @torch.no_grad()
    def snap_to_consensus(self, pop):
        """Put every member on the CONSENSUS values: canonicalise the member in the certified gauge
        (function-preserving), then overwrite every certified entry (row or column view) with the
        quorum-mean canonical value and every certified row's bias with the reference's canonical
        bias (biases scatter ~40% across members inside the 0.02 certificate; averaging them breaks
        the function, the reference's are self-consistent). Members then differ only in their
        uncertified units. Call before the optimizers are created (parametrisation changes)."""
        cov = self.cov; dev = self.device
        for mi, m in enumerate(pop):
            st = None
            for k in range(self.nW):
                W, b = m.layers[k].weight, m.layers[k].bias
                is_head = (k == self.nW - 1)
                We = W if st is None else W * st[None, :]
                if not is_head:
                    if k == 0:
                        cm = torch.ones(W.shape[1], dtype=torch.bool, device=dev)
                    else:
                        cm = torch.zeros(W.shape[1], dtype=torch.bool, device=dev)
                        cm[self.perms[mi][k - 1]] = self.prev_cert[k - 1] if k - 1 == self.nh - 1 else cov[k - 1]["row_cov"].any(1)
                    nr = (We * cm[None, :]).norm(dim=1).clamp_min(1e-12)
                    W.copy_(We / nr[:, None]); b.copy_(b / nr); st = nr
                else:
                    W.copy_(We)
                # certified entries -> consensus values (ref frame -> member frame)
                e = cov[k]; mask = e["row_cov"] | e["col_cov"]
                rows = torch.arange(W.shape[0], device=dev) if is_head else self.perms[mi][k]
                cols = torch.arange(W.shape[1], device=dev) if k == 0 else self.perms[mi][k - 1]
                Wm = W[rows][:, cols]
                # hidden layers: quorum MEAN entries. Head: the REFERENCE's canonical columns -- the head
                # certificate is direction-only, column SCALES differ across members (each compensates
                # its own L4 units' uncertified-input entries), so the mean head is inconsistent
                # (members' MAE 0.07 -> 0.34..7.7 with the mean head; the ref's head is self-consistent).
                Wm = torch.where(mask, e["M_ref"] if is_head else e["V"], Wm)
                W[rows[:, None], cols[None, :]] = Wm
                if not is_head:
                    rc = e["row_cov"].any(1)
                    b[rows[rc]] = e["b_ref"][rc]

    @torch.no_grad()
    def reinit_unsolved(self, pop):
        """Re-draw every unsolved unit in every member: its incoming row (+bias) and its outgoing
        column, uniform at the member's own scale (median certified row norm of that layer, median
        column norm of the next), so the unit restarts from a fresh point instead of the stationary
        one the members converged to. Adam state is left to the optimizer (zeroed at first refresh)."""
        g = torch.Generator(device=self.device).manual_seed(12345)
        for mi, m in enumerate(pop):
            for (l, j) in self.targets:
                u = int(self.perms[mi][l][j])
                W, b = m.layers[l].weight, m.layers[l].bias
                cert = self.cov[l]["row_cov"].any(1)
                rows_c = self.perms[mi][l][cert] if bool(cert.any()) else torch.arange(W.shape[0], device=self.device)
                scale = W[rows_c].norm(dim=1).median()
                r = torch.rand(W.shape[1], generator=g, device=self.device) * 2 - 1
                W[u] = r / r.norm() * scale
                b[u] = (torch.rand(1, generator=g, device=self.device).item() * 2 - 1) * b[rows_c].abs().median()
                Wn = m.layers[l + 1].weight
                cscale = Wn.norm(dim=0).median()
                c = torch.rand(Wn.shape[0], generator=g, device=self.device) * 2 - 1
                Wn[:, u] = c / c.norm() * cscale

    def _assignment(self):
        """round-robin: query i -> target i % n_targets; per layer the member columns of the targets."""
        per_layer = {}
        for k, (l, j) in enumerate(self.targets):
            per_layer.setdefault(l, []).append(k)
        tgt_idx = {l: torch.tensor(ks, device=self.device) for l, ks in per_layer.items()}
        return per_layer, tgt_idx

    def act_term(self, pop):
        """Loss term for gen_queries (added to the OUTPUT-disagreement loss): every query pushes
        ALL unsolved units at once. Per member and target unit the PREACTIVATION z (full gradient
        while the unit is inactive; the leaky output would push back with slope 0.01) with a
        saturating margin: min(z, tau)/tau, tau = the unit's mean positive preactivation on
        uniform points, so a unit that is already active stops pulling and the gradient goes to
        the ones that are not. Averaged over members, targets and queries; weight cfg.target_act_w."""
        w = float(getattr(self.cfg, "target_act_w", 1.0))
        per_layer, _ = self._assignment()
        cols = {mi: {l: torch.tensor([int(self.perms[mi][l][self.targets[k][1]]) for k in ks], device=self.device)
                     for l, ks in per_layer.items()} for mi in range(len(pop))}
        with torch.no_grad():
            U = torch.rand(2048, self.dims[0], device=self.device) * 2 - 1
            tau = {}
            for mi, m in enumerate(pop):
                zs = _preacts(m, U)
                for l in per_layer:
                    z = zs[l][:, cols[mi][l]]
                    pos = (z > 0).float()
                    tau[(mi, l)] = ((z * pos).sum(0) / pos.sum(0).clamp_min(1)).clamp_min(1e-3)
        def f(I, sl):
            tot = 0.0
            for mi, m in enumerate(pop):
                zs = _preacts_grad(m, I)
                for l in per_layer:
                    z = zs[l][:, cols[mi][l]]
                    tot = tot + (torch.minimum(z, tau[(mi, l)]) / tau[(mi, l)]).mean()
            return -w * tot / len(pop)
        return f

    @torch.no_grad()
    def record_hit(self, pop, I):
        """teacher-free: fraction of each target's OWN queries on which the reference's unit is active."""
        ref = pop[self.ref]; Z = _preacts(ref, I)
        A = torch.stack([(Z[l][:, j] > 0).float() for (l, j) in self.targets], 1)       # (q, nT)
        own = A.mean(0).cpu()
        self.last_hit = float(own.mean()); self.last_hit_min = float(own.min()); self.last_hit_low = int((own < 0.5).sum())
        self.last_hit_n = float(A.sum(1).mean())

    # -------------------------------------------------------- queries --
    def queries(self, pop, n, X_buf, gen, steps=30, lr=0.1, tau=0.1):
        """n inputs maximizing disagreement on the targets where they are active."""
        dev = self.device; d0 = self.dims[0]
        if not self.targets or n <= 0:
            return torch.empty(0, d0, device=dev)
        cfg = self.cfg
        # starts: buffer points where the reference's targets fire (if any), else random
        starts = []
        ref = pop[self.ref]
        gcpu = torch.Generator(device="cpu").manual_seed(int(torch.randint(0, 10 ** 6, (1,))))
        if X_buf is not None and len(X_buf) > 0:
            Xb = X_buf[torch.randperm(len(X_buf), generator=gcpu)[:min(20000, len(X_buf))]].to(dev)
            with torch.no_grad():
                Z = _preacts(ref, Xb)
                act = torch.zeros(len(Xb), device=dev)
                for (l, j) in self.targets:
                    act += (Z[l][:, j] > 0).float()
            idx = act.nonzero().flatten()
            if idx.numel() > 0:
                sel = idx[torch.randperm(idx.numel(), generator=gcpu).to(dev)[:n // 2]]
                starts.append(Xb[sel] + 0.05 * torch.randn(len(sel), d0, device=dev))
        n_rand = n - sum(len(s) for s in starts)
        if cfg.qg_init == "uniform":
            starts.append((torch.rand(n_rand, d0, device=dev) * 2 - 1) * cfg.qg_range)
        else:
            starts.append(torch.randn(n_rand, d0, device=dev) * cfg.qg_init_std)
        x = torch.cat(starts).detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        by_layer = {}
        for (l, j) in self.targets:
            by_layer.setdefault(l, []).append(j)
        for step in range(steps):
            opt.zero_grad()
            total = 0.0
            Zall = [_preacts_grad(m, x) for m in pop]                   # once per member per step
            for l, js in by_layer.items():
                js_t = torch.tensor(js, device=dev)
                zs = []
                for mi, m in enumerate(pop):
                    z = Zall[mi][l]
                    zs.append(z[:, self.perms[mi][l][js_t]])          # (n, n_targets_l) in ref unit order
                Zm = torch.stack(zs)                                   # (P, n, T)
                gate = torch.sigmoid(Zm / tau).mean(0)                 # soft "active in the members"
                var = Zm.var(0)                                        # disagreement across members
                total = total + (var * gate).sum(1).mean()
            (-total).backward()
            opt.step()
            if step in {int(0.5 * steps), int(0.8 * steps)}:
                for g in opt.param_groups:
                    g["lr"] /= 10
        x = x.detach()
        with torch.no_grad():                                          # hit rate: a target active in the reference
            Z = _preacts(ref, x); hit = torch.zeros(len(x), dtype=torch.bool, device=dev)
            for (l, j) in self.targets:
                hit |= Z[l][:, j] > 0
            self.last_hit = float(hit.float().mean())
        return x
