"""Conv-aware port of the --balance machinery (rownorm + dirlr) for nets.ConvNet.

The MLP version lives in stacked.py and only runs inside the stacked fast path, which is
disabled as soon as any peel mask exists -- so no run has ever balanced during peeling, and
reconstruct_cnn never balanced at all (the flags were parsed and silently ignored).

Gauge facts used (leaky-ReLU / ReLU are positively homogeneous; avg-pool commutes with a
per-channel scale):
  * scaling channel c of hidden layer l (filter + bias) by f_c and dividing the next layer's
    incoming weights for channel c by f_c leaves the function unchanged;
  * after torch.flatten the Linear input index of (c, h, w) is c*H*W + h*W + w, so a conv ->
    FC transition compensates a contiguous block of H*W columns per channel.

Pinned (kink-exact) rows are never rescaled and never projected: they are exact at the
magnitude the scale-match injection chose, and the partial record would otherwise disagree
with the members. A layer whose NEXT layer has pinned rows is never rescaled either, since
the compensation would alter those exact rows.
"""
import math
import torch


def hidden_layers(net):
    return list(range(len(net.layers) - 1))


def row_norms(layer):
    """Per-output-channel norm of the augmented [w|b] row."""
    W = layer.weight
    k = W.shape[0]
    return torch.sqrt(W.reshape(k, -1).pow(2).sum(1) + layer.bias.pow(2))


def fan_in(layer):
    return layer.weight[0].numel()


@torch.no_grad()
def hidden_preacts(net, xb):
    """Pre-activations of every hidden layer on the batch (list, in layer order). Works for
    nets.ConvNet and nets.MLP (no conv stack)."""
    x = xb
    out = []
    if hasattr(net, "forward_preacts"):                   # ConvNet's own forward (LeNet-5 subsampling + C3 mask included)
        return net.forward_preacts(x)
    if not hasattr(net, "n_conv"):                        # MLP
        for layer in net.layers[:-1]:
            z = layer(x); out.append(z); x = net.act(z)
        return out
    if x.dim() == 2:
        x = x.view(x.shape[0], *net.input_shape)
    for i in range(net.n_conv):
        z = net.layers[i](x)
        out.append(z)
        x = net.act(z)
        if net.pools[i] > 0:
            x = torch.nn.functional.avg_pool2d(x, net.pools[i])
    x = torch.flatten(x, 1)
    for j in range(net.n_conv, len(net.layers) - 1):
        z = net.layers[j](x)
        out.append(z)
        x = net.act(z)
    return out


def _adam_state(opt, p):
    if opt is None:
        return None
    st = opt.state.get(p)
    return st if (st and "exp_avg" in st) else None


@torch.no_grad()
def scale_channels_(net, l, f, opt=None):
    """Function-preserving gauge move: rows (filters+bias) of hidden layer l *= f (per channel),
    next layer's incoming weights for those channels /= f. Adam moments are transformed by the
    gradient law so m/sqrt(v) is unchanged (no transient)."""
    lay, nxt = net.layers[l], net.layers[l + 1]
    f = f.to(lay.weight.dtype)
    shp_row = [-1] + [1] * (lay.weight.dim() - 1)
    lay.weight.mul_(f.view(shp_row)); lay.bias.mul_(f)
    st = _adam_state(opt, lay.weight)
    if st is not None:
        st["exp_avg"].div_(f.view(shp_row)); st["exp_avg_sq"].div_(f.view(shp_row) ** 2)
    st = _adam_state(opt, lay.bias)
    if st is not None:
        st["exp_avg"].div_(f); st["exp_avg_sq"].div_(f * f)
    # next layer: which input columns belong to channel c?
    if nxt.weight.dim() == 4:                              # conv -> conv
        fin = f.view(1, -1, 1, 1)
    elif lay.weight.dim() == 4:                            # conv -> flatten -> linear
        hw = nxt.weight.shape[1] // f.numel()
        fin = f.repeat_interleave(hw).view(1, -1)
    else:                                                  # linear -> linear
        fin = f.view(1, -1)
    nxt.weight.div_(fin)
    st = _adam_state(opt, nxt.weight)
    if st is not None:
        st["exp_avg"].mul_(fin); st["exp_avg_sq"].mul_(fin ** 2)


@torch.no_grad()
def canonicalize_(net, xb, target_rms=1.0, fixed_norm=0.0, opt=None, layers=None, equalize=True):
    """One-time canonicalisation of the given hidden layers (default all), in order, so each has
    pre-activation RMS == target_rms on xb (fixed_norm > 0: every row norm == fixed_norm instead).
    equalize: afterwards make every row of the layer the same norm (the layer median), compensated
    into the next layer -- still function-preserving, and a SCALAR target per layer survives the
    channel permutations the peel applies when aligning members. Returns per-layer targets
    (scalar tensors, or per-row tensors if equalize=False); None for layers not in `layers`."""
    layers = set(hidden_layers(net) if layers is None else layers)
    for l in hidden_layers(net):
        if l not in layers:
            continue
        if fixed_norm > 0:
            f = fixed_norm / row_norms(net.layers[l]).clamp_min(1e-12)
        else:
            z = hidden_preacts(net, xb)[l]                 # recomputed: earlier layers already rescaled
            rms = float(z.pow(2).mean().sqrt().clamp_min(1e-12))
            f = torch.full((net.layers[l].weight.shape[0],), target_rms / rms,
                           device=net.layers[l].weight.device)
        scale_channels_(net, l, f, opt)
        if equalize and fixed_norm <= 0:
            n = row_norms(net.layers[l]).clamp_min(1e-12)
            scale_channels_(net, l, n.median() / n, opt)
    out = []
    for l in hidden_layers(net):
        if l not in layers:
            out.append(None); continue
        n = row_norms(net.layers[l]).detach()
        out.append(n.median().clone() if equalize or fixed_norm > 0 else n.clone())
    return out


@torch.no_grad()
def project_rows_(net, targets, pinned=None, eps=1e-12):
    """Snap every UNPINNED hidden (row,bias) back to its fixed norm (removes Adam's radial random
    walk; directions untouched; head free). `pinned`: {l: bool[Cout]} rows to leave alone."""
    for l, tgt in enumerate(targets):
        if tgt is None:
            continue
        lay = net.layers[l]
        n = row_norms(lay).clamp_min(eps)
        f = tgt / n
        if pinned is not None and l in pinned and pinned[l] is not None:
            f = torch.where(pinned[l].to(f.device), torch.ones_like(f), f)
        shp = [-1] + [1] * (lay.weight.dim() - 1)
        lay.weight.mul_(f.view(shp).to(lay.weight.dtype)); lay.bias.mul_(f.to(lay.bias.dtype))


def split_param_groups(opt, net, lr):
    """Rebuild a single-group Adam as one param group per layer (weight+bias), keeping the Adam
    state. Returns the (possibly new) optimizer. No-op if already split."""
    if len(opt.param_groups) == len(net.layers):
        return opt
    groups = []
    for lay in net.layers:
        ps = [p for p in (lay.weight, lay.bias) if p.requires_grad]
        if ps:
            groups.append({"params": ps, "lr": lr})
    if not groups:
        return opt
    new = torch.optim.Adam(groups, lr=lr)
    for p, st in opt.state.items():
        new.state[p] = st
    return new


def apply_dirlr(net, opt, eta, base_lr, head=False):
    """Per-layer lr = eta * median(row,bias)-norm / sqrt(fan_in): every hidden layer's rows rotate by
    ~eta per Adam step regardless of their pinned scale. The head keeps base_lr unless head=True.
    Requires split param groups (one per layer, in layer order, skipping frozen layers)."""
    lrs = []
    gi = 0
    for l, lay in enumerate(net.layers):
        if not (lay.weight.requires_grad or lay.bias.requires_grad):
            continue
        g = opt.param_groups[gi]; gi += 1
        if l == len(net.layers) - 1 and not head:
            g["lr"] = base_lr; lrs.append(base_lr); continue
        n = float(row_norms(lay).median())
        lr = eta * n / math.sqrt(fan_in(lay))
        g["lr"] = lr; lrs.append(lr)
    return lrs


@torch.no_grad()
def layer_stats(net, xb):
    """Diagnostics: per hidden layer (pre-act RMS, median row norm, dead frac, always-on frac)."""
    zs = hidden_preacts(net, xb)
    rows = []
    for l, z in enumerate(zs):
        zc = z.transpose(0, 1).reshape(z.shape[1], -1)      # (C, N*positions)
        pos = (zc > 0).float().mean(1)
        rows.append(dict(layer=l, rms=float(z.pow(2).mean().sqrt()),
                         row_norm=float(row_norms(net.layers[l]).median()),
                         dead=int((pos < 1e-3).sum()), always_on=int((pos > 1 - 1e-3).sum()),
                         n=int(z.shape[1])))
    rows.append(dict(layer=len(net.layers) - 1, rms=float("nan"),
                     row_norm=float(row_norms(net.layers[-1]).median()), dead=0, always_on=0,
                     n=int(net.layers[-1].weight.shape[0])))
    return rows


@torch.no_grad()
def keep_alive_(net, xb, pinned, margin_frac=0.05, dead_frac=1e-3):
    """An unpinned hidden unit that is silent on the data (positive on < dead_frac of the batch x positions) is NOT
    allowed to die: its bias is reset so that its maximum pre-activation on the batch is -margin, i.e. its kink sits
    just outside the data (margin = margin_frac x the layer's pre-activation RMS).  Direction untouched.  Returns the
    number of units touched per hidden layer.  Rationale: a dead unit is a collapsed posterior ("this channel does
    nothing") that makes all members agree where nothing is known; alive-at-the-boundary units disagree exactly where
    an unlearned teacher unit could still turn on, which is where the disagreement sampler should go."""
    zs = hidden_preacts(net, xb); touched = []
    for l, z in enumerate(zs):
        zc = z.transpose(0, 1).reshape(z.shape[1], -1)        # (C, N*positions)
        pos = (zc > 0).float().mean(1); zmax = zc.amax(1)
        rms = z.pow(2).mean().sqrt().clamp_min(1e-6); margin = margin_frac * rms
        dead = pos < dead_frac
        if pinned is not None and pinned.get(l) is not None:
            dead = dead & ~pinned[l].to(dead.device)
        idx = dead.nonzero().flatten()
        if len(idx):
            net.layers[l].bias.data[idx] += (-margin - zmax[idx]).to(net.layers[l].bias.dtype)
        touched.append(int(len(idx)))
    return touched
