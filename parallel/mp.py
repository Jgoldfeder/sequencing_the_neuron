"""Multi-GPU committee: the two hot phases of method.reconstruct, parallelised without touching method.py.

1. Query generation (gen_queries): every GPU holds a replica of ALL members and generates its share of
   the q queries (the disagreement search is independent per query); shares are concatenated on the main
   device. Installed by monkey-patching method.gen_queries (reconstruct looks the name up at call time).
2. Training (StackedTrainer): the P members are split across GPUs; each GPU trains its subset as one
   batched matmul (stacked.StackedTrainer math, same minibatch sequence); per-member losses are gathered
   on the main device. Installed by replacing stacked.StackedTrainer (reconstruct imports it at call time).

Plain mode only (no --balance*, no peel/stages); those paths raise. Everything else (teacher labelling,
ring buffer, consensus, eval) stays on the main device exactly as before.
"""
import copy, threading, concurrent.futures as cf
import torch

_tls = threading.local()
_orig_gen_queries = None
DEVICES = None
MAIN = None


def set_devices(devices):
    global DEVICES, MAIN
    DEVICES = [torch.device(d) for d in devices]
    MAIN = DEVICES[0]


def _replica(m, dev):
    r = m.__class__(m.dims, act=m.act_name).to(dev)
    r.load_state_dict({k: v.to(dev, non_blocking=True) for k, v in m.state_dict().items()})
    r.eval()
    for p in r.parameters():
        p.requires_grad_(False)
    return r


# ----------------------------------------------------------------------------- queries --
def gen_queries_mp(pop, cfg, input_dim, device, gen, losses=None, init_override=None, ref_last=False, feats=None, extra=None):
    """Split cfg.q across DEVICES; each device runs the ORIGINAL gen_queries on a replica committee with
    its own generator (seeded from `gen`, so the run stays seed-determined); concat on `device`."""
    if getattr(_tls, "inside", False) or DEVICES is None or len(DEVICES) == 1 or init_override is not None \
            or feats is not None or extra is not None or ref_last:
        return _orig_gen_queries(pop, cfg, input_dim, device, gen, losses=losses, init_override=init_override,
                                 ref_last=ref_last, feats=feats, extra=extra)
    G = len(DEVICES); q = cfg.q
    shares = [q // G + (1 if i < q % G else 0) for i in range(G)]
    seeds = [int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen, device=gen.device)) for _ in range(G)]
    import dataclasses

    def work(i):
        dev = DEVICES[i]
        _tls.inside = True
        try:
            with torch.cuda.device(dev):
                reps = [_replica(m, dev) for m in pop]
                g = torch.Generator(device=dev); g.manual_seed(seeds[i])
                sub = dataclasses.replace(cfg, q=shares[i])
                out = _orig_gen_queries(reps, sub, input_dim, dev, g, losses=losses)
                return out.to(device, non_blocking=True)
        finally:
            _tls.inside = False
    with cf.ThreadPoolExecutor(max_workers=G) as ex:
        outs = list(ex.map(work, range(G)))
    torch.cuda.synchronize(device)
    return torch.cat(outs, 0)


def install_query_patch(method_module):
    global _orig_gen_queries
    if _orig_gen_queries is None:
        _orig_gen_queries = method_module.gen_queries
    method_module.gen_queries = gen_queries_mp


# ---------------------------------------------------------------------------- training --
class _DeviceStacked:
    """stacked.StackedTrainer for a SUBSET of members, living on `dev` (members themselves stay on MAIN)."""

    def __init__(self, members, opts, dev):
        self.dev = dev; self.P = len(members); self.L = len(members[0].layers); self.act = members[0].act
        self.W = [torch.stack([m.layers[l].weight.detach() for m in members]).to(dev).requires_grad_(True) for l in range(self.L)]
        self.b = [torch.stack([m.layers[l].bias.detach() for m in members]).to(dev).requires_grad_(True) for l in range(self.L)]
        d = opts[0].defaults
        self.opt = torch.optim.Adam(self.W + self.b, lr=d["lr"], betas=d["betas"], eps=d["eps"],
                                    weight_decay=d.get("weight_decay", 0.0), amsgrad=d.get("amsgrad", False))
        with torch.no_grad():
            for l in range(self.L):
                for kind, P_st in (("weight", self.W[l]), ("bias", self.b[l])):
                    sts = [opts[mi].state.get(getattr(m.layers[l], kind)) for mi, m in enumerate(members)]
                    if not all(st and "exp_avg" in st for st in sts):
                        continue
                    st = self.opt.state[P_st]
                    s0 = sts[0]["step"]
                    st["step"] = s0.clone() if torch.is_tensor(s0) else torch.tensor(float(s0))
                    st["exp_avg"] = torch.stack([s["exp_avg"] for s in sts]).to(dev)
                    st["exp_avg_sq"] = torch.stack([s["exp_avg_sq"] for s in sts]).to(dev)

    @torch.no_grad()
    def pull(self, members):
        for l in range(self.L):
            self.W[l].copy_(torch.stack([m.layers[l].weight for m in members]).to(self.dev, non_blocking=True))
            self.b[l].copy_(torch.stack([m.layers[l].bias for m in members]).to(self.dev, non_blocking=True))

    @torch.no_grad()
    def push(self, members, opts=None):
        for mi, m in enumerate(members):
            for l in range(self.L):
                m.layers[l].weight.copy_(self.W[l][mi].to(m.layers[l].weight.device, non_blocking=True))
                m.layers[l].bias.copy_(self.b[l][mi].to(m.layers[l].bias.device, non_blocking=True))
        if opts is not None:
            for l in range(self.L):
                for kind, P_st in (("weight", self.W[l]), ("bias", self.b[l])):
                    st = self.opt.state.get(P_st)
                    if not st or "exp_avg" not in st:
                        continue
                    for mi, m in enumerate(members):
                        p = getattr(m.layers[l], kind); ms = opts[mi].state[p]; md = p.device
                        ms["step"] = st["step"].clone(); ms["exp_avg"] = st["exp_avg"][mi].to(md); ms["exp_avg_sq"] = st["exp_avg_sq"][mi].to(md)

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def forward(self, xb):
        h = xb.unsqueeze(0).expand(self.P, -1, -1)
        for l in range(self.L):
            h = torch.baddbmm(self.b[l].unsqueeze(1), h, self.W[l].transpose(1, 2))
            if l < self.L - 1:
                h = self.act(h)
        return h

    def step(self, xb, yb, fit_loss="l1"):
        self.opt.zero_grad(set_to_none=True)
        out = self.forward(xb)
        per = ((out - yb.unsqueeze(0)) ** 2).mean(dim=(1, 2)) if fit_loss == "mse" else (out - yb.unsqueeze(0)).abs().mean(dim=(1, 2))
        per.sum().backward()
        self.opt.step()
        return per.detach()


class MultiStackedTrainer:
    """Drop-in for stacked.StackedTrainer: members split across DEVICES (the main device gets the smallest
    share, it also runs the sampler's share and the consensus). Same minibatch to every device; per-member
    losses gathered on the main device in member order."""
    row_targets = None; actnorm_target = None; actnorm_ref = None; last_lrs = None; dirlr = None

    def __init__(self, pop, opts):
        self.ids = [id(m) for m in pop]; self.P = len(pop); self.L = len(pop[0].layers); self.act = pop[0].act
        G = len(DEVICES); base, rem = divmod(self.P, G)
        sizes = [base + (1 if i >= G - rem else 0) for i in range(G)]          # extras go to the NON-main devices
        self.groups = []; s = 0
        for i, n in enumerate(sizes):
            idx = list(range(s, s + n)); s += n
            if n:
                self.groups.append((DEVICES[i], idx, _DeviceStacked([pop[j] for j in idx], [opts[j] for j in idx], DEVICES[i])))
        self.main = pop[0].layers[0].weight.device
        self.pool = cf.ThreadPoolExecutor(max_workers=len(self.groups))
        print(f"  [mp] members per device: " + ", ".join(f"{str(dev)}:{len(idx)}" for dev, idx, _ in self.groups), flush=True)

    def matches(self, pop):
        return [id(m) for m in pop] == self.ids

    def pull(self, pop):
        for dev, idx, st in self.groups:
            st.pull([pop[j] for j in idx])

    def push(self, pop, opts=None):
        for dev, idx, st in self.groups:
            st.push([pop[j] for j in idx], [opts[j] for j in idx] if opts is not None else None)
        torch.cuda.synchronize(self.main)

    def set_lr(self, lr):
        for _, _, st in self.groups:
            st.set_lr(lr)

    def step(self, xb, yb, fit_loss="l1"):
        def work(g):
            dev, idx, st = g
            with torch.cuda.device(dev):
                xd = xb.to(dev, non_blocking=True); yd = yb.to(dev, non_blocking=True)
                return st.step(xd, yd, fit_loss).to(self.main, non_blocking=True)
        pers = list(self.pool.map(work, self.groups))
        return torch.cat(pers, 0)

    # balance modes are not supported in the parallel trainer
    def balance(self, *a, **k): raise NotImplementedError("--balance is not supported by run_parallel.py (plain mode only)")
    balance_layers = rownorm_init = project_rows = rownorm_refresh = apply_dirlr = balance

    @property
    def W(self):   # the balance logging reads _stacked.W; give it the main device's view
        return [torch.cat([st.W[l].to(self.main) for _, _, st in self.groups]) for l in range(self.L)]


def install_trainer_patch(stacked_module):
    stacked_module.StackedTrainer = MultiStackedTrainer
