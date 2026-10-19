"""Stacked committee training: all P members as ONE batched matmul per layer.

Plain-mode fast path for method.reconstruct. Mathematically identical to training
the members one by one: the loss is the SUM of the per-member means, so each
member's gradient is exactly its own, and Adam is elementwise, so one Adam over
the stacked parameters equals P separate Adams with the same hyper-parameters
and state. Only the summation order inside cuBLAS differs (bmm vs mm), i.e.
fp32 rounding. The members' nn.Modules stay the public objects: `pull` copies
them into the stack before an iteration's training, `push` copies the trained
weights back so the sampler / consensus / snapshots see them unchanged.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets import Square, Squareplus, Abs, Power, Sine


def _adam_kernel(p, g, m, v, lr, step, b1, b2, eps):
    """torch.optim.Adam's update (no weight decay / amsgrad) as one fused pass; lr and step are 0-d tensors
    so torch.compile does not recompile when the lr schedule or dirlr changes them."""
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    bc1 = 1 - b1 ** step
    bc2 = 1 - b2 ** step
    denom = v.sqrt() / bc2.sqrt() + eps
    p.add_((lr / bc1) * (m / denom), alpha=-1.0)


_ADAM_COMPILED = None


def _adam_update(p, g, m, v, lr, step, b1, b2, eps):
    global _ADAM_COMPILED
    if _ADAM_COMPILED is None:
        try:
            _ADAM_COMPILED = torch.compile(_adam_kernel)
        except Exception:
            _ADAM_COMPILED = _adam_kernel
    try:
        _ADAM_COMPILED(p, g, m, v, lr, step, b1, b2, eps)
    except Exception:                                  # compile backend failure: eager fallback
        _ADAM_COMPILED = _adam_kernel
        _adam_kernel(p, g, m, v, lr, step, b1, b2, eps)


class FlatAdam:
    """Adam for the stacked trainer over ONE flat fp32 buffer holding every member's every parameter.
    The per-layer tensors (W[l], b[l]) are LEAF tensors whose storage aliases the buffer (detached views),
    so the forward/backward are unchanged and each leaf gets its own .grad; step() copies the 22 grads
    into one flat grad buffer and runs the update as a single fused pass (0.26 ms for 7.7M params vs
    1.30 ms for torch's foreach Adam). [Making the leaves plain autograd views of one leaf buffer is a
    trap: every view's backward scatters into a full-size tensor, 22 x 30 MB per step, slower than
    foreach.] Mimics the torch.optim.Adam surface the trainer uses: param_groups (one per layer, each
    {"params": [W, b], "lr", "_base_lr"}), state[param] = {"step", "exp_avg", "exp_avg_sq"} (views of
    the flat moment buffers, so the balance / rownorm code can rescale them in place), zero_grad(),
    step(). Same Adam formula and fp32 math; per-group lrs (dirlr) take a per-group pass."""

    def __init__(self, shapes, lr, betas, eps, device, dtype, weight_decay=0.0, amsgrad=False):
        assert weight_decay == 0.0 and not amsgrad, "FlatAdam: weight_decay / amsgrad not supported"
        self.b1, self.b2, self.eps = float(betas[0]), float(betas[1]), float(eps)
        n = sum(int(torch.Size(sh).numel()) for grp in shapes for sh in grp)
        self.flat = torch.zeros(n, device=device, dtype=dtype)      # parameter storage (no grad itself)
        self.g = torch.zeros(n, device=device, dtype=dtype)         # flat gradient buffer
        self.m = torch.zeros(n, device=device, dtype=dtype)
        self.v = torch.zeros(n, device=device, dtype=dtype)
        self.step_t = torch.zeros((), device=device, dtype=dtype)
        self.lr_t = torch.full((), float(lr), device=device, dtype=dtype)
        self.param_groups, self.state, self.spans = [], {}, []
        a = 0
        for grp in shapes:
            params, span = [], []
            for sh in grp:
                k = int(torch.Size(sh).numel())
                p = self.flat[a:a + k].view(*sh).detach().requires_grad_(True)   # leaf aliasing the buffer
                params.append(p)
                self.state[p] = {"step": self.step_t, "exp_avg": self.m[a:a + k].view(*sh),
                                 "exp_avg_sq": self.v[a:a + k].view(*sh)}
                span.append((a, a + k)); a += k
            self.param_groups.append({"params": params, "lr": float(lr), "_base_lr": float(lr)})
            self.spans.append((span[0][0], span[-1][1]))     # a group's params are contiguous in the buffer
            self._pspans = getattr(self, "_pspans", []) + [span]
        self.defaults = {"lr": float(lr), "betas": (self.b1, self.b2), "eps": self.eps}

    def zero_grad(self, set_to_none=True):
        for grp in self.param_groups:
            for p in grp["params"]:
                p.grad = None

    @torch.no_grad()
    def step(self):
        grads = [p.grad for grp in self.param_groups for p in grp["params"]]
        if any(gr is None for gr in grads):
            return
        gviews = [self.g[a:b] for grp_spans in self._pspans for (a, b) in grp_spans]
        torch._foreach_copy_(gviews, [gr.reshape(-1) for gr in grads])
        g = self.g
        self.step_t += 1
        lrs = [grp["lr"] for grp in self.param_groups]
        if all(x == lrs[0] for x in lrs):
            self.lr_t.fill_(lrs[0])
            _adam_update(self.flat, g, self.m, self.v, self.lr_t, self.step_t, self.b1, self.b2, self.eps)
        else:                                               # per-layer lrs (dirlr): one pass per group
            for (a, b), lr in zip(self.spans, lrs):
                self.lr_t.fill_(lr)
                _adam_update(self.flat[a:b], g[a:b], self.m[a:b], self.v[a:b], self.lr_t, self.step_t,
                             self.b1, self.b2, self.eps)


class StackedTrainer:
    def __init__(self, pop, opts):
        self.P = len(pop); self.L = len(pop[0].layers); self.act = pop[0].act
        self.ids = [id(m) for m in pop]
        d = opts[0].defaults
        self.base_lr0 = d["lr"]; self.dirlr = None            # --balance-dirlr: per-layer lr = eta * |row| / sqrt(fan_in)
        p0 = pop[0].layers[0].weight
        shapes = [[(self.P,) + tuple(m.layers[l].weight.shape), (self.P,) + tuple(m.layers[l].bias.shape)]
                  for l, m in ((l, pop[0]) for l in range(self.L))]
        self.opt = FlatAdam(shapes, lr=d["lr"], betas=d["betas"], eps=d["eps"], device=p0.device, dtype=p0.dtype,
                            weight_decay=d.get("weight_decay", 0.0), amsgrad=d.get("amsgrad", False))
        self.W = [self.opt.param_groups[l]["params"][0] for l in range(self.L)]   # views of the flat buffer
        self.b = [self.opt.param_groups[l]["params"][1] for l in range(self.L)]
        with torch.no_grad():
            for l in range(self.L):
                self.W[l].copy_(torch.stack([m.layers[l].weight.detach() for m in pop]))
                self.b[l].copy_(torch.stack([m.layers[l].bias.detach() for m in pop]))
        self._import_state(pop, opts)

    def matches(self, pop):
        return [id(m) for m in pop] == self.ids

    @torch.no_grad()
    def _import_state(self, pop, opts):
        """Carry over the members' Adam state (if any) so switching paths mid-run is exact."""
        for l in range(self.L):
            for kind, P_st in (("weight", self.W[l]), ("bias", self.b[l])):
                sts = [opts[mi].state.get(getattr(m.layers[l], kind)) for mi, m in enumerate(pop)]
                if not all(st and "exp_avg" in st for st in sts):
                    continue
                st = self.opt.state[P_st]                     # entries are VIEWS of the flat moment buffers: copy into them
                st["step"].fill_(float(sts[0]["step"]))
                st["exp_avg"].copy_(torch.stack([s["exp_avg"] for s in sts]))
                st["exp_avg_sq"].copy_(torch.stack([s["exp_avg_sq"] for s in sts]))

    @torch.no_grad()
    def pull(self, pop):
        for l in range(self.L):
            self.W[l].copy_(torch.stack([m.layers[l].weight for m in pop]))
            self.b[l].copy_(torch.stack([m.layers[l].bias for m in pop]))

    @torch.no_grad()
    def push(self, pop, opts=None):
        for mi, m in enumerate(pop):
            for l in range(self.L):
                m.layers[l].weight.copy_(self.W[l][mi]); m.layers[l].bias.copy_(self.b[l][mi])
        if opts is not None:                    # keep the per-member optimizers' state coherent too
            for l in range(self.L):
                for kind, P_st in (("weight", self.W[l]), ("bias", self.b[l])):
                    st = self.opt.state.get(P_st)
                    if not st or "exp_avg" not in st:
                        continue
                    for mi, m in enumerate(pop):
                        p = getattr(m.layers[l], kind); ms = opts[mi].state[p]
                        ms["step"] = st["step"].clone(); ms["exp_avg"] = st["exp_avg"][mi].clone(); ms["exp_avg_sq"] = st["exp_avg_sq"][mi].clone()

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def forward(self, xb):
        h = xb.unsqueeze(0).expand(self.P, -1, -1)
        for l in range(self.L):
            h = torch.baddbmm(self.b[l].unsqueeze(1), h, self.W[l].transpose(1, 2))
            if l < self.L - 1:
                h = self.act(h)
        return h                                              # (P, B, out)

    def _act_prime(self, z):
        """Elementwise derivative of the hidden activation at pre-activation z (same for every member)."""
        a = self.act
        if isinstance(a, nn.LeakyReLU):
            return torch.where(z > 0, torch.ones((), dtype=z.dtype, device=z.device),
                               torch.full((), a.negative_slope, dtype=z.dtype, device=z.device))
        if isinstance(a, nn.ReLU):
            return (z > 0).to(z.dtype)
        if isinstance(a, nn.PReLU):
            return torch.where(z > 0, torch.ones((), dtype=z.dtype, device=z.device), a.weight.reshape(()).to(z.dtype))
        if isinstance(a, nn.Tanh):
            return 1 - torch.tanh(z) ** 2
        if isinstance(a, nn.Sigmoid):
            s = torch.sigmoid(z); return s * (1 - s)
        if isinstance(a, Square):
            return 2 * z
        if isinstance(a, nn.ELU):
            return torch.where(z > 0, torch.ones((), dtype=z.dtype, device=z.device), a.alpha * torch.exp(z))
        if isinstance(a, nn.GELU):
            Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2.0))); return Phi + z * torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        if isinstance(a, nn.SiLU):
            s = torch.sigmoid(z); return s * (1 + z * (1 - s))
        if isinstance(a, nn.Mish):
            tsp = torch.tanh(F.softplus(z)); return tsp + z * torch.sigmoid(z) * (1 - tsp * tsp)
        if isinstance(a, nn.Softplus):
            return torch.where(z * a.beta > a.threshold, torch.ones_like(z), torch.sigmoid(a.beta * z))
        if isinstance(a, nn.Softsign):
            return 1 / (1 + z.abs()) ** 2
        if isinstance(a, nn.CELU):
            return torch.where(z > 0, torch.ones_like(z), torch.exp(z / a.alpha))
        if isinstance(a, nn.SELU):
            return 1.0507009873554805 * torch.where(z > 0, torch.ones_like(z), 1.6732632423543772 * torch.exp(z))
        if isinstance(a, nn.Hardswish):
            return torch.where(z <= -3, torch.zeros_like(z), torch.where(z >= 3, torch.ones_like(z), (2 * z + 3) / 6))
        if isinstance(a, Squareplus):
            return 0.5 * (1 + z / torch.sqrt(z * z + a.b))
        if isinstance(a, Power):
            return a.k * z ** (a.k - 1)
        if isinstance(a, Abs):
            return torch.sign(z)
        if isinstance(a, Sine):
            return torch.cos(z)
        raise ValueError(f"stacked boundary loss: unsupported activation {type(a).__name__}")

    def boundary_loss(self, xb, yb, floor=1e-6):
        """Hard-label boundary loss for all members at once (method.boundary_loss, batched).
        yb: (B,) long labels. Returns the per-member mean squared distance violation (P,).
        The input gradient of f_y - f_j is a manual vector-Jacobian product through the
        batched forward (one bmm per layer times the activation derivative), so the whole
        loss is an ordinary graph: one backward, no create_graph double backward."""
        h = xb.unsqueeze(0).expand(self.P, -1, -1); pres = []
        for l in range(self.L):
            z = torch.baddbmm(self.b[l].unsqueeze(1), h, self.W[l].transpose(1, 2))
            if l < self.L - 1:
                pres.append(z); h = self.act(z)
            else:
                out = z                                            # (P, B, C)
        yi = yb.view(1, -1, 1).expand(self.P, -1, 1)
        fy = out.gather(2, yi)
        j = out.scatter(2, yi, float("-inf")).argmax(2, keepdim=True)   # runner-up class
        viol = (out.gather(2, j) - fy).clamp_min(0).squeeze(2)         # (P, B): f_j - f_y if misclassified
        g = torch.zeros_like(out).scatter_(2, yi, 1.0).scatter_(2, j, -1.0)   # d(f_y - f_j)/d out
        for l in reversed(range(self.L)):
            g = torch.bmm(g, self.W[l])                                # (P, B, in_l)
            if l > 0:
                g = g * self._act_prime(pres[l - 1])
        gn = g.norm(dim=2).clamp_min(floor)                            # ||grad_x (f_y - f_j)||
        return ((viol / gn) ** 2).mean(dim=1)

    def step(self, xb, yb, fit_loss="l1", floor=1e-6):
        """One optimizer step for every member on the same minibatch; returns the per-member losses (detached).
        fit_loss: 'l1' | 'mse' (yb float targets) | 'xent' | 'boundary' (yb long hard labels)."""
        self.opt.zero_grad(set_to_none=True)
        if fit_loss == "boundary":
            per = self.boundary_loss(xb, yb, floor)
        else:
            out = self.forward(xb)
            if fit_loss == "xent":
                per = F.cross_entropy(out.flatten(0, 1), yb.repeat(self.P), reduction="none").view(self.P, -1).mean(dim=1)
            elif fit_loss == "mse":
                per = ((out - yb.unsqueeze(0)) ** 2).mean(dim=(1, 2))
            else:
                per = (out - yb.unsqueeze(0)).abs().mean(dim=(1, 2))
        per.sum().backward()
        if self.dirlr is not None:
            self.apply_dirlr()
        self.opt.step()
        if self.row_targets is not None:
            self.project_rows()
        if self.actnorm_target is not None:
            self.project_actnorm(self.actnorm_target)
        return per.detach()


# ---------------------------------------------------------------------------
# gauge balancing (--balance): incoming norm == outgoing norm for every hidden unit
# ---------------------------------------------------------------------------
@torch.no_grad()
def balance_tensors_(W, b, states, sweeps=3, eps=1e-12):
    """ENorm-style balancing of a positively homogeneous MLP (leaky-ReLU / ReLU), in place.
    W: list of L weight tensors (..., out, in); b: list of L bias tensors (..., out); the leading
    dims (if any) index members. For each hidden unit i of layer l (l = 0..L-2) with incoming norm
    a_i = |(W_l[i,:], b_l[i])| and outgoing norm d_i = |W_{l+1}[:, i]|, scale the row+bias by
    r_i = sqrt(d_i/a_i) and the column by 1/r_i, so both become sqrt(a_i d_i). The function is
    unchanged (phi(r z) = r phi(z) for r > 0). Sweeps over the layers are repeated because layer
    l+1's balancing changes layer l's outgoing norms.
    states: list of L pairs (stW, stb) of Adam state dicts (or None) holding 'exp_avg' and
    'exp_avg_sq'; they are transformed by the gradient law (g_row -> g_row/r, g_col -> g_col*r,
    squares accordingly), so m/sqrt(v) is unchanged and Adam continues without a transient.
    Returns the per-layer median row norm BEFORE balancing (diagnostic)."""
    L = len(W)
    before = [float(W[l].norm(dim=-1).median()) for l in range(L)]
    for _ in range(sweeps):
        for l in range(L - 1):
            a = torch.sqrt((W[l] ** 2).sum(-1) + b[l] ** 2)                 # (..., out_l)
            d = torch.sqrt((W[l + 1] ** 2).sum(-2))                        # (..., out_l)
            ok = (a > eps) & (d > eps)
            r = torch.where(ok, torch.sqrt(d.clamp_min(eps) / a.clamp_min(eps)), torch.ones_like(a))
            rr, rc = r.unsqueeze(-1), r.unsqueeze(-2)                      # rows of W_l, columns of W_{l+1}
            W[l].mul_(rr); b[l].mul_(r); W[l + 1].div_(rc)
            stW, stb = states[l] if states[l] is not None else (None, None)
            if stW is not None and "exp_avg" in stW:
                stW["exp_avg"].div_(rr); stW["exp_avg_sq"].div_(rr * rr)
            if stb is not None and "exp_avg" in stb:
                stb["exp_avg"].div_(r); stb["exp_avg_sq"].div_(r * r)
            stW1 = states[l + 1][0] if states[l + 1] is not None else None
            if stW1 is not None and "exp_avg" in stW1:
                stW1["exp_avg"].mul_(rc); stW1["exp_avg_sq"].mul_(rc * rc)
    return before


@torch.no_grad()
def balance_module_(net, opt=None, sweeps=3):
    """Balance one nn.Module MLP (net.layers of nn.Linear) in place, transforming its Adam state."""
    W = [l.weight for l in net.layers]; b = [l.bias for l in net.layers]
    states = [((opt.state.get(l.weight) or None), (opt.state.get(l.bias) or None)) if opt is not None else None
              for l in net.layers]
    states = [None if s is None or s[0] is None or "exp_avg" not in s[0] else s for s in states]
    return balance_tensors_(W, b, states, sweeps)


def _balance_stacked(self, sweeps=3):
    """Balance every member of the stack in place (see balance_tensors_), Adam state included."""
    states = []
    for l in range(self.L):
        sw, sb = self.opt.state.get(self.W[l]), self.opt.state.get(self.b[l])
        states.append((sw, sb) if sw and "exp_avg" in sw else None)
    return balance_tensors_(self.W, self.b, states, sweeps)


StackedTrainer.balance = _balance_stacked


# ---------------------------------------------------------------------------
# LAYER-wise balancing: equal pre-activation RMS in every hidden layer (--balance, mode "layer")
# ---------------------------------------------------------------------------
@torch.no_grad()
def balance_layers_(W, b, states, X, act, target=None, eps=1e-12):
    """Equalise the pre-activation scale of every hidden layer, in place and function-preserving.
    W[l]: (..., out, in), b[l]: (..., out) (leading dims = members); X: (B, in) probe batch.
    For hidden layer l with pre-activation RMS s_l (per member, over the batch and units) choose
    c_l = target / s_l, scale W_l, b_l by c_l and W_{l+1} by 1/c_l: z_l becomes target-scaled while
    z_{l+1} and the output are unchanged (phi(c z) = c phi(z)). target=None -> the geometric mean of
    the s_l (per member), i.e. same total scale, just equalised across layers; the head absorbs the
    product. Adam moments are transformed by the same scalars (gradient law), so m/sqrt(v) is unchanged.
    Returns (rms_before, rms_after) per hidden layer, averaged over members."""
    L = len(W); lead = W[0].shape[:-2]
    h = X.reshape((1,) * len(lead) + X.shape).expand(*lead, *X.shape)          # (..., B, in)
    rms = []
    for l in range(L - 1):
        z = torch.matmul(h, W[l].transpose(-1, -2)) + b[l].unsqueeze(-2)         # (..., B, out)
        rms.append(z.pow(2).mean(dim=(-1, -2)).sqrt().clamp_min(eps))           # (...,)
        h = act(z)
    rms = torch.stack(rms, 0)                                                   # (L-1, ...)
    tgt = rms.log().mean(0).exp() if target is None else torch.full_like(rms[0], float(target))
    for l in range(L - 1):
        c = tgt / rms[l]                                                        # (...,)
        cw = c.reshape(*c.shape, 1, 1); cb = c.reshape(*c.shape, 1)
        W[l].mul_(cw); b[l].mul_(cb); W[l + 1].div_(cw)
        stW, stb = states[l] if states[l] is not None else (None, None)
        if stW is not None and "exp_avg" in stW:
            stW["exp_avg"].div_(cw); stW["exp_avg_sq"].div_(cw * cw)
        if stb is not None and "exp_avg" in stb:
            stb["exp_avg"].div_(cb); stb["exp_avg_sq"].div_(cb * cb)
        stW1 = states[l + 1][0] if states[l + 1] is not None else None
        if stW1 is not None and "exp_avg" in stW1:
            stW1["exp_avg"].mul_(cw); stW1["exp_avg_sq"].mul_(cw * cw)
    before = [float(r.mean()) for r in rms]
    after = [float(tgt.mean())] * (L - 1)
    return before, after


def _balance_layers_stacked(self, X, target=None):
    states = []
    for l in range(self.L):
        sw, sb = self.opt.state.get(self.W[l]), self.opt.state.get(self.b[l])
        states.append((sw, sb) if sw and "exp_avg" in sw else None)
    return balance_layers_(self.W, self.b, states, X, self.act, target)


StackedTrainer.balance_layers = _balance_layers_stacked


@torch.no_grad()
def balance_layers_module_(net, opt, X, target=None):
    W = [l.weight for l in net.layers]; b = [l.bias for l in net.layers]
    states = [((opt.state.get(l.weight) or None), (opt.state.get(l.bias) or None)) if opt is not None else None for l in net.layers]
    states = [None if s is None or s[0] is None or "exp_avg" not in s[0] else s for s in states]
    return balance_layers_(W, b, states, X, net.act, target)


# ---------------------------------------------------------------------------
# ROW-NORM projection (--balance-mode rownorm): hidden rows keep a FIXED norm, projected after every step
# ---------------------------------------------------------------------------
@torch.no_grad()
def rownorm_targets_(W, b, states, X, act, target_rms=1.0, fixed_norm=0.0):
    """One-time canonicalisation, then return each hidden unit's (row, bias) norm as the fixed norm it
    will be projected back to after every optimizer step. fixed_norm > 0: every hidden (row, bias) is
    set to that norm directly (the scale factors are folded into the next layer, head absorbs) --
    Adam's step is absolute, so the row norm IS the directional learning rate: relative directional
    step per Adam step ~ lr*sqrt(fan_in)/norm. fixed_norm == 0: equalise pre-activation RMS to
    `target_rms` instead and freeze whatever norms that gives."""
    if fixed_norm > 0:
        for l in range(len(W) - 1):
            n = torch.sqrt((W[l] ** 2).sum(-1) + b[l] ** 2).clamp_min(1e-12)
            f = fixed_norm / n                                                   # (..., out_l)
            W[l].mul_(f.unsqueeze(-1)); b[l].mul_(f); W[l + 1].div_(f.unsqueeze(-2))
            stW, stb = states[l] if states[l] is not None else (None, None)
            if stW is not None and "exp_avg" in stW:
                stW["exp_avg"].div_(f.unsqueeze(-1)); stW["exp_avg_sq"].div_(f.unsqueeze(-1) ** 2)
            if stb is not None and "exp_avg" in stb:
                stb["exp_avg"].div_(f); stb["exp_avg_sq"].div_(f * f)
            stW1 = states[l + 1][0] if states[l + 1] is not None else None
            if stW1 is not None and "exp_avg" in stW1:
                stW1["exp_avg"].mul_(f.unsqueeze(-2)); stW1["exp_avg_sq"].mul_(f.unsqueeze(-2) ** 2)
    else:
        balance_layers_(W, b, states, X, act, target=target_rms)
    return [torch.sqrt((W[l] ** 2).sum(-1) + b[l] ** 2) for l in range(len(W) - 1)]      # (..., out_l)


@torch.no_grad()
def project_rows_(W, b, targets, eps=1e-12):
    """Rescale every hidden unit's (row, bias) back to its fixed norm. Removes the radial component of
    the last update (Adam's norm random walk) without touching directions; the head is free."""
    for l, tgt in enumerate(targets):
        n = torch.sqrt((W[l] ** 2).sum(-1) + b[l] ** 2).clamp_min(eps)
        f = tgt / n
        W[l].mul_(f.unsqueeze(-1)); b[l].mul_(f)


def _stacked_rownorm_init(self, X, target_rms=1.0, fixed_norm=0.0):
    states = []
    for l in range(self.L):
        sw, sb = self.opt.state.get(self.W[l]), self.opt.state.get(self.b[l])
        states.append((sw, sb) if sw and "exp_avg" in sw else None)
    self.row_targets = rownorm_targets_(self.W, self.b, states, X, self.act, target_rms, fixed_norm)


def _stacked_project_rows(self):
    if getattr(self, "row_targets", None) is not None:
        project_rows_(self.W, self.b, self.row_targets)


StackedTrainer.rownorm_init = _stacked_rownorm_init
StackedTrainer.project_rows = _stacked_project_rows
StackedTrainer.row_targets = None


# ---------------------------------------------------------------------------
# ACT-NORM projection (--balance-mode actnorm): within each hidden layer all (row,bias) norms equal, and
# the common norm floats per step so the layer's pre-activation RMS on the batch equals the target
# ---------------------------------------------------------------------------
def _stacked_forward_capture(self, xb):
    """forward() that also records each hidden layer's pre-activation RMS per member (detached)."""
    h = xb.unsqueeze(0).expand(self.P, -1, -1); rms = []
    for l in range(self.L):
        z = torch.baddbmm(self.b[l].unsqueeze(1), h, self.W[l].transpose(1, 2))
        if l < self.L - 1:
            rms.append(z.detach().pow(2).mean(dim=(1, 2)).sqrt())           # (P,)
            h = self.act(z)
        else:
            h = z
    self._last_rms = rms
    return h


@torch.no_grad()
def _stacked_ref_rms(self):
    """Pre-activation RMS per hidden layer per member on the FIXED reference batch (weights-only function;
    the training minibatch cannot be used: once the buffer mixes warm-start and sampler points, consecutive
    minibatches differ in RMS by 2x+ and the projection would inject that as multiplicative noise)."""
    h = self.actnorm_ref.unsqueeze(0).expand(self.P, -1, -1); rms = []
    for l in range(self.L - 1):
        z = torch.baddbmm(self.b[l].unsqueeze(1), h, self.W[l].transpose(1, 2))
        rms.append(z.pow(2).mean(dim=(1, 2)).sqrt()); h = self.act(z)
    return rms


@torch.no_grad()
def _stacked_project_actnorm(self, target=1.0, eps=1e-12):
    """After an optimizer step: (1) equalise all hidden (row,bias) norms within each layer to their mean
    (kills within-layer row inflation = dying), (2) scale the layer so its pre-activation RMS on the fixed
    reference batch equals `target` (kills the depth-wise scale decay). Layers are done in order so each
    layer's RMS is measured with the layers below already projected. Head free."""
    h = self.actnorm_ref.unsqueeze(0).expand(self.P, -1, -1)
    for l in range(self.L - 1):
        n = torch.sqrt((self.W[l] ** 2).sum(-1) + self.b[l] ** 2).clamp_min(eps)        # (P, out)
        f = n.mean(dim=1, keepdim=True) / n                                               # equalise within layer
        self.W[l].mul_(f.unsqueeze(-1)); self.b[l].mul_(f)
        z = torch.baddbmm(self.b[l].unsqueeze(1), h, self.W[l].transpose(1, 2))
        c = (target / z.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(eps))                   # (P,)
        self.W[l].mul_(c.view(-1, 1, 1)); self.b[l].mul_(c.view(-1, 1))
        h = self.act(z * c.view(-1, 1, 1))


StackedTrainer.ref_rms = _stacked_ref_rms
StackedTrainer.forward_capture = _stacked_forward_capture
StackedTrainer.project_actnorm = _stacked_project_actnorm
StackedTrainer.actnorm_target = None
StackedTrainer.actnorm_ref = None


# ---------------------------------------------------------------------------
# DIRECTIONAL learning rate (--balance-dirlr eta): per-layer lr so that every layer's rows (head
# included) rotate by the same fraction eta per Adam step, whatever scale the gauge put them at
# ---------------------------------------------------------------------------
@torch.no_grad()
def _stacked_apply_dirlr(self):
    """Adam moves each weight by ~lr per step, so a row with fan_in entries and norm n rotates by
    ~lr*sqrt(fan_in)/n per step. Set lr_l = eta * median_n_l / sqrt(fan_in_l) (times the run's base-lr
    schedule factor) so the directional step is eta in every layer; the hidden norms are constant under
    rownorm, the head's norm is read off each step."""
    base = self.opt.param_groups[0].get("_base_lr", self.base_lr0); sched = base / self.base_lr0   # schedule factor (set_lr writes _base_lr)
    self.last_lrs = []
    for l, g in enumerate(self.opt.param_groups):
        if l == self.L - 1 and not getattr(self, "dirlr_head", False):
            g["lr"] = base; self.last_lrs.append(float(base)); continue      # head: no gauge -> its scale is the loss's business, keep the base lr
        n = torch.sqrt((self.W[l] ** 2).sum(-1) + self.b[l] ** 2).median()
        lr = float(self.dirlr * sched * n / (self.W[l].shape[-1] ** 0.5))
        g["lr"] = lr; self.last_lrs.append(lr)


def _stacked_set_lr(self, lr):
    for g in self.opt.param_groups:
        g["lr"] = lr; g["_base_lr"] = lr


StackedTrainer.apply_dirlr = _stacked_apply_dirlr
StackedTrainer.set_lr = _stacked_set_lr
StackedTrainer.last_lrs = None


def _stacked_rownorm_refresh(self, X, target_rms=1.0, fixed_norm=0.0):
    """--balance-refresh: re-canonicalise ONCE (function-preserving: layer rescale, next layer / head
    compensated, Adam moments transformed) and re-freeze the row norms. Undoes the depth-wise RMS decay
    that fixed norms accumulate as directions align, and brings the head back to O(1)."""
    self.rownorm_init(X, target_rms, fixed_norm)


StackedTrainer.rownorm_refresh = _stacked_rownorm_refresh
