"""Isolated two-sided per-layer consensus for ConvNets (the CNN analogue of the tanh-era `--layer-consensus`
"iso" report, ReversePeel.report_only):

  * units (channels) of every hidden layer are paired to the reference member by the correlation of their
    pre-activations on a probe set (each member run with ITS OWN prefix; Hungarian assignment), so a layer's
    pairing does not depend on the weights or the consensus of the layers below;
  * `identified` units = paired with |corr| > 0.99 in at least `quorum` members;
  * ROWS: a unit's incoming row is compared across members only on the input channels identified one layer down
    (sub-row renormalised, inf-norm eps ball, quorum); skipped (reported as -1) when < 10% of the inputs are identified;
  * COLS: a unit's outgoing column in the next layer (restricted to that layer's identified rows) compared the same way.

Pure report (no pinning).  Returns {l: dict(n, paired, rows, cols, ident_in, n_in)} for hidden layers l.
"""
import math
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from align import cnn_canonicalize_


@torch.no_grad()
def _preacts(net, x):
    """per hidden layer: pre-activation per channel, flattened over (batch, positions) -> (N, C)"""
    if hasattr(net, "forward_preacts"):                 # the net's own forward (LeNet-5 subsampling + C3 mask included)
        return [z.permute(0, 2, 3, 1).reshape(-1, z.shape[1]) if z.dim() == 4 else z for z in net.forward_preacts(x)]
    h = x.view(x.shape[0], *net.input_shape); zs = []
    for i, lay in enumerate(net.layers[:-1]):
        if lay.weight.dim() == 4:
            z = lay(h); zs.append(z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])); h = net.act(z)
            if net.pools[i] > 0:
                h = F.avg_pool2d(h, net.pools[i])
        else:
            if h.dim() > 2:
                h = h.reshape(h.shape[0], -1)
            z = lay(h); zs.append(z); h = net.act(z)
    return zs


def _pair(Zr, Z, odd=False):
    """odd activations (tanh family): a unit and its negation are the same unit -> pair on |corr|; returns the
    signed correlation so the caller can flip anti-correlated units into the reference's sign."""
    A = Zr - Zr.mean(0); B = Z - Z.mean(0)
    A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12); B = B / B.norm(dim=0, keepdim=True).clamp_min(1e-12)
    C = (A.T @ B).float()
    r, c = linear_sum_assignment((-(C.abs() if odd else C)).cpu().numpy())
    perm = torch.tensor(c, device=Z.device); return perm, C[r, c]


def _rowfeat(net, l, unit, insel, inperm=None):
    """unit [w|b] of layer l, input channels reordered by `inperm` (member channel for each reference channel) and
    restricted to the reference-order mask `insel`, renormalised"""
    W = net.layers[l].weight[unit]
    if W.dim() == 3:                                   # conv: (Cin, kh, kw)
        Wp = W if inperm is None else W[inperm]
        w = Wp[insel].reshape(-1)
    else:                                              # fc: input = channels x spatial (channel-major)
        per = W.numel() // insel.numel()
        W3 = W.reshape(insel.numel(), per)
        if inperm is not None:
            W3 = W3[inperm]
        w = W3[insel].reshape(-1)
    v = torch.cat([w, net.layers[l].bias[unit].reshape(1)]).double()
    return v / v.norm().clamp_min(1e-12)


def _colfeat(net, l, unit, outsel, outperm=None):
    """outgoing column of unit `unit` of layer l in layer l+1, rows reordered by `outperm` (member row for each
    reference row) and restricted to the reference-order mask `outsel`, renormalised"""
    Wn = net.layers[l + 1].weight
    Wo = Wn if outperm is None else Wn[outperm]
    if Wo.dim() == 4:
        v = Wo[outsel][:, unit].reshape(-1)
    else:
        cin = net.layers[l].weight.shape[0]; per = Wo.shape[1] // cin
        v = Wo[outsel].reshape(int(outsel.sum()), cin, per)[:, unit].reshape(-1)
    v = v.double(); return v / v.norm().clamp_min(1e-12)


def _ball_count(feats, eps, quorum):
    """feats (P, d): size of the largest inf-norm eps-ball across members >= quorum?"""
    D = (feats[:, None] - feats[None]).abs().amax(-1)
    return int((D < eps).sum(1).max()) >= quorum


@torch.no_grad()
def cnn_two_sided(pop, X, ref_idx, eps, quorum, corr_thr=0.99):
    mems = []
    for m in pop:
        c = m.clone(); cnn_canonicalize_(c); mems.append(c)
    ref = mems[ref_idx]
    Zr = _preacts(ref, X)
    nh = len(ref.layers) - 1
    from align import _sign_gauge, _lenet5_flip_
    odd = _sign_gauge(ref) is False
    perms = []; corrs = []
    for m in mems:
        Z = _preacts(m, X); ps = []; cs = []
        for l in range(nh):
            p, c = _pair(Zr[l], Z[l], odd); ps.append(p); cs.append(c.abs() if odd else c)
            if odd:                                      # flip anti-correlated member units into the reference's sign
                for u in (c < 0).nonzero().flatten().tolist():
                    _lenet5_flip_(m, l, int(p[u]), -1.0)
        perms.append(ps); corrs.append(cs)
    rep = {}
    ident = []
    for l in range(nh):
        n = ref.layers[l].weight.shape[0]
        good = torch.stack([corrs[i][l] > corr_thr for i in range(len(mems))]).sum(0)   # per ref unit: members paired well
        ident.append(good >= quorum)
    for l in range(nh):
        n = ref.layers[l].weight.shape[0]
        insel = (torch.ones(ref.layers[l].weight.shape[1], dtype=torch.bool, device=X.device) if l == 0 else ident[l - 1])
        rows = -1
        if l == 0 or int(insel.sum()) >= max(1, insel.numel() // 10):
            rows = 0
            for u in range(n):
                feats = torch.stack([_rowfeat(mems[i], l, int(perms[i][l][u]), insel, None if l == 0 else perms[i][l - 1]) for i in range(len(mems))])
                rows += int(_ball_count(feats, eps, quorum))
        cols = -1
        if l + 1 <= nh:                                      # next layer exists (hidden or head)
            outsel = (ident[l + 1] if l + 1 < nh else torch.ones(ref.layers[l + 1].weight.shape[0], dtype=torch.bool, device=X.device))
            if int(outsel.sum()) >= max(1, outsel.numel() // 10):
                cols = 0
                for u in range(n):
                    feats = torch.stack([_colfeat(mems[i], l, int(perms[i][l][u]), outsel, perms[i][l + 1] if l + 1 < nh else None) for i in range(len(mems))])
                    cols += int(_ball_count(feats, eps, quorum))
        rep[l] = dict(n=n, paired=int(ident[l].sum()), rows=rows, cols=cols, ident_in=int(insel.sum()), n_in=insel.numel())
    return rep


def fmt(rep):
    parts = []
    for l in sorted(rep):
        v = rep[l]
        r = f"{v['rows']}/{v['n']}" if v['rows'] >= 0 else f"*/{v['n']} (inputs identified {v['ident_in']}/{v['n_in']})"
        c = f" cols {v['cols']}/{v['n']}" if v['cols'] >= 0 else ""
        parts.append(f"L{l + 1}[rows {r}{c} | paired {v['paired']}/{v['n']}]")
    return "  ".join(parts)
