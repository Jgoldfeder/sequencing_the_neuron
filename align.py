"""Network alignment under scaling/permutation isomorphisms (paper App. D).

CNN support (ported from the refactor Standardizer): conv layers are aligned by
permuting OUTPUT CHANNELS (the conv analog of neurons) and canonicalizing their
scale (ReLU: unit-L2 per out-channel, pushed into the next layer's input dim;
tanh: sign, odd-symmetry flip). The scale/permutation propagates to the next
layer, handling Conv->Conv (input-channel dim) and Conv->Linear (the flattened
fan-in is blocked by channel: block c is [c*sec:(c+1)*sec], sec = spatial size).
See cnn_canonicalize_, cnn_align_to_, cnn_param_errors.

Torch convention: for nn.Linear, weight is (out, in). Hidden neuron c of
layer l owns row c of layers[l].weight and bias c of layers[l].bias, and is
consumed via column c of layers[l+1].weight.

LeakyReLU is positively homogeneous, so the scaling isomorphism applies
(alpha > 0): scale row c of W_l and b_l by 1/alpha, scale column c of
W_{l+1} by alpha. Polarity does NOT apply (LeakyReLU is not odd).

Sigmoid / tanh have NO scaling isomorphism; their only per-neuron gauge is
polarity: sigma(-z) = 1 - sigma(z) (flip + absorb a constant into the next
bias) and tanh(-z) = -tanh(z) (pure odd flip, nothing to absorb). Both are
canonicalized by sign_canonicalize_.
"""
import torch
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def sign_canonicalize_(net, absorb=True):
    """Canonicalize the polarity isomorphism of an odd-symmetric-ish
    activation: sigmoid sigma(-z) = 1 - sigma(z) (absorb=True) or tanh
    tanh(-z) = -tanh(z) (absorb=False). Flip every hidden neuron whose
    largest-|.| entry of [row|bias] is negative. (The refactor Standardizer
    keyed on the SUM of the row; that is unstable -- a row whose entries sum
    to ~0 flips under 1e-2 noise and scores as a ~2||w|| error -- whereas the
    max-|.| entry only flips if the top two entries tie.) Flipping neuron k
    of layer l: negate row k of W_l/b_l; for
    sigmoid ABSORB column k of W_{l+1} into b_{l+1} (the "1" in 1 - sigma;
    tanh needs no such term); then negate that column.
    Function-preserving and exact; in-place."""
    for l in range(len(net.layers) - 1):
        W = net.layers[l].weight  # (out_l, in_l)
        b = net.layers[l].bias
        Wb = torch.cat([W, b[:, None]], 1)
        s = torch.sign(Wb.gather(1, Wb.abs().argmax(1, keepdim=True)).squeeze(1))
        s[s == 0] = 1.0
        flip = s < 0
        if not flip.any():
            continue
        nxt = net.layers[l + 1]
        if absorb:
            nxt.bias.add_(nxt.weight[:, flip].sum(1))
        nxt.weight.mul_(s.unsqueeze(0))
        W.mul_(s.unsqueeze(1))
        b.mul_(s)


@torch.no_grad()
def scale_normalize_(net):
    """Canonicalize the net's activation isomorphism, in-place.

    (Leaky)ReLU (positively homogeneous): every hidden neuron's incoming row
    gets unit L2 norm; the scale is pushed into the next layer's column.
    Sigmoid / tanh (no scaling isomorphism): polarity canonicalization
    instead, via sign_canonicalize_ (sigmoid absorbs the 1 - sigma constant
    into the next bias; tanh is odd so nothing is absorbed). Either way,
    functionally equivalent nets emerge with identical parameters up to
    neuron permutation, so downstream matching / clustering / averaging
    operate on comparable raw weights."""
    if isinstance(net.act, torch.nn.Sigmoid):
        sign_canonicalize_(net, absorb=True)
        return
    if isinstance(net.act, torch.nn.Tanh):
        sign_canonicalize_(net, absorb=False)
        return
    if not isinstance(net.act, torch.nn.LeakyReLU):
        return
    for l in range(len(net.layers) - 1):
        W = net.layers[l].weight  # (out_l, in_l)
        b = net.layers[l].bias
        norms = W.norm(dim=1).clamp_min(1e-12)  # per hidden neuron
        W.div_(norms.unsqueeze(1))
        b.div_(norms)
        net.layers[l + 1].weight.mul_(norms.unsqueeze(0))


@torch.no_grad()
def _match_features(net, l):
    """Feature vector per hidden neuron of layer l for matching."""
    W = net.layers[l].weight
    b = net.layers[l].bias.unsqueeze(1)
    return torch.cat([W, b], dim=1)


@torch.no_grad()
def greedy_perm(A, B, signed=False):
    """Greedy L1 matching between rows of A and rows of B (paper App. D).
    Returns perm: list such that B's row perm[i] should be moved to position
    i to align with A's row i. signed=True (sign-gauge activations): each B
    row may also be matched as its NEGATIVE; returns (perm, flip) where
    flip[j] says B's row j matched better negated."""
    n = A.shape[0]
    D = torch.cdist(A.float(), B.float(), p=1)
    if signed:
        Dn = torch.cdist(A.float(), -B.float(), p=1)
        neg = Dn < D
        D = torch.minimum(D, Dn)
    INF = torch.tensor(float("inf"), device=D.device)
    perm = [None] * n
    flip = torch.zeros(n, dtype=torch.bool, device=D.device)
    for _ in range(n):
        idx = D.argmin().item()
        i, j = idx // n, idx % n
        perm[i] = j
        if signed:
            flip[j] = neg[i, j]
        D[i, :] = INF
        D[:, j] = INF
    return (perm, flip) if signed else perm


@torch.no_grad()
def flip_neurons_(net, l, flip, absorb):
    """Negate hidden neurons `flip` (bool mask) of layer l, function-
    preserving: negate their rows/biases, for sigmoid (absorb=True) add the
    affected columns of W_{l+1} into b_{l+1} (1 - sigma), negate the columns."""
    if not bool(flip.any()):
        return
    s = torch.where(flip, -1.0, 1.0).to(net.layers[l].weight.dtype)
    nxt = net.layers[l + 1]
    if absorb:
        nxt.bias.add_(nxt.weight[:, flip].sum(1))
    nxt.weight.mul_(s.unsqueeze(0))
    net.layers[l].weight.mul_(s.unsqueeze(1))
    net.layers[l].bias.mul_(s)


def _sign_gauge(net):
    """None for (leaky)ReLU (scale gauge, handled by scale_normalize_);
    else the absorb flag for the activation's polarity gauge."""
    if isinstance(net.act, torch.nn.Sigmoid):
        return True
    if isinstance(net.act, torch.nn.Tanh):
        return False
    return None


@torch.no_grad()
def match_layer_(ref, net, l):
    """Align hidden layer l of `net` (already scale_normalize_d) into `ref`'s
    frame, in place: greedy L1 row matching, sign-aware for sigmoid/tanh (a
    neuron is matched as +row or -row, whichever is closer, and flipped
    accordingly -- robust where the absolute polarity convention of
    sign_canonicalize_ ties), then permuted; the permutation propagates into
    layer l+1. Returns perm."""
    absorb = _sign_gauge(net)
    A, B = _match_features(ref, l), _match_features(net, l)
    if absorb is None:
        perm = greedy_perm(A, B)
    else:
        perm, flip = greedy_perm(A, B, signed=True)
        flip_neurons_(net, l, flip, absorb)
    permute_layer_(net, l, perm)
    return perm


@torch.no_grad()
def permute_layer_(net, l, perm):
    """Apply neuron permutation to hidden layer l of net (in-place).
    perm[i] = old index that now sits at position i."""
    idx = torch.tensor(perm, device=net.layers[l].weight.device)
    net.layers[l].weight.copy_(net.layers[l].weight[idx])
    net.layers[l].bias.copy_(net.layers[l].bias[idx])
    W_next = net.layers[l + 1].weight
    W_next.copy_(W_next[:, idx])


@torch.no_grad()
def align_clone_to(recon, teacher):
    """Return (normalized teacher, aligned+normalized recon copy)."""
    t = teacher.clone()
    r = recon.clone()
    # measure in the higher precision of the two: an fp64 reconstruction vs the
    # fp32 teacher must be compared in fp64, else normalizing the fp32 teacher
    # in fp32 caps the reported eps at fp32 rounding (~6e-8) and hides genuine
    # fp64 recovery. (fp32-only case is unaffected.)
    dt = torch.promote_types(t.layers[0].weight.dtype, r.layers[0].weight.dtype)
    t = t.to(dt)
    r = r.to(dt)
    scale_normalize_(t)
    scale_normalize_(r)
    for l in range(len(t.layers) - 1):
        match_layer_(t, r, l)
    return t, r


@torch.no_grad()
def layer_eps_split(recon, teacher, masks):
    """Per-hidden-layer weight-eps split into FROZEN vs UNSOLVED, given
    `masks` = {layer: bool mask over RECON neurons that are already solved}.
    Aligns exactly like param_errors, maps each mask through the alignment
    permutation (perm[i] = recon neuron now at position i), and returns
    {layer: {"n": n_frozen, "fz": (max, mean), "uf": (max, mean)}}. Same
    max (incl. bias) / mean (weight) conventions as param_errors' per-matrix."""
    t = teacher.clone(); r = recon.clone()
    dt = torch.promote_types(t.layers[0].weight.dtype, r.layers[0].weight.dtype)
    t = t.to(dt); r = r.to(dt)
    scale_normalize_(t); scale_normalize_(r)
    out = {}
    for l in range(len(t.layers) - 1):
        perm = greedy_perm(_match_features(t, l), _match_features(r, l))
        permute_layer_(r, l, perm)
        if l in masks:
            pidx = torch.tensor(perm, device=r.layers[l].weight.device)
            am = masks[l].to(pidx.device)[pidx]              # frozen mask, aligned order
            dW = (t.layers[l].weight - r.layers[l].weight).abs()
            db = (t.layers[l].bias - r.layers[l].bias).abs()
            pmax = torch.maximum(dW.max(dim=1).values, db)   # per-neuron max (incl bias)
            pmean = dW.mean(dim=1)                            # per-neuron mean (weight)
            def st(sel):
                return ((pmax[sel].max().item(), pmean[sel].mean().item())
                        if int(sel.sum()) else (float("nan"), float("nan")))
            out[l] = {"n": int(am.sum()), "fz": st(am), "uf": st(~am)}
    return out


@torch.no_grad()
def hard_fix_out_(t, r):
    """Quotient the HARD-LABEL output invariance before comparing (in place).

    An argmax oracle cannot identify W_out -> s*W_out + 1 u^T, b_out ->
    s*b_out + beta*1 (s>0, u, beta): every logit shifts by the same
    u.h(x)+beta and scales by s, so all labels are preserved on every input
    -- these h+2 dims are unlearnable in principle. Project BOTH nets' heads
    onto the identifiable subspace (remove the class-mean of every W column
    and of b) and rescale r's head by the LSQ-optimal s. Hidden layers are
    fully identifiable from hard labels (boundary bends = kinks): untouched."""
    for net in (t, r):
        W, b = net.layers[-1].weight, net.layers[-1].bias
        W.sub_(W.mean(0, keepdim=True))
        b.sub_(b.mean())
    Wr, br = r.layers[-1].weight, r.layers[-1].bias
    u = torch.cat([Wr.flatten(), br])
    v = torch.cat([t.layers[-1].weight.flatten(), t.layers[-1].bias])
    s = max(float(u @ v) / max(float(u @ u), 1e-30), 0.0)  # invariance needs s>0
    Wr.mul_(s)
    br.mul_(s)


@torch.no_grad()
def param_errors(recon, teacher, hard=False):
    """Max/mean absolute parameter error after alignment + scale
    normalization. Does not mutate recon. Returns dict with per-layer max
    errors, overall max, and mean per matrix. hard=True: additionally quotient
    the label-preserving output-layer family (hard_fix_out_) so eps covers
    only what a hard-label attacker could in principle recover."""
    t, r = align_clone_to(recon, teacher)
    if hard:
        hard_fix_out_(t, r)
    max_errs, mean_errs = [], []
    for la, lb in zip(t.layers, r.layers):
        for pa, pb in zip(la.parameters(), lb.parameters()):
            d = (pa - pb).abs()
            max_errs.append(d.max().item())
            mean_errs.append(d.mean().item())
    return {
        "max_eps": max(max_errs),
        "mean_eps_per_matrix": mean_errs,
        "max_eps_per_matrix": max_errs,
    }


# --------------------------------------------------------------- CNN align --
def _push_to_next(nxt, factor, conv_prev):
    """Multiply the next layer's INPUT dim by `factor` ([out_prev]). Handles
    conv->conv (input-channel dim), conv->linear (blocked fan-in), linear->*."""
    w = nxt.weight
    if conv_prev and w.dim() == 4:                      # conv -> conv
        w.mul_(factor.view(1, -1, 1, 1))
    elif conv_prev and w.dim() == 2:                    # conv -> linear (blocks)
        k = factor.shape[0]
        sec = w.shape[1] // k
        if sec * k != w.shape[1]:
            raise ValueError("conv out-channels must divide the linear fan-in")
        w.mul_(factor.repeat_interleave(sec)[None, :])
    else:                                               # linear -> *
        w.mul_(factor[None, :])


@torch.no_grad()
def cnn_canonicalize_(net):
    """In-place scale/sign canonicalization of every hidden (non-final) layer.
    ReLU: unit-L2 per out-unit; tanh: sign (odd flip). Function-preserving; the
    factor is pushed into the next layer. The final Linear head is left as-is
    (its outputs are the fixed logits -- no output gauge)."""
    L = net.layers
    relu = net.act_name in ("relu", "leaky_relu")
    for i in range(len(L) - 1):
        W, b = L[i].weight, L[i].bias
        conv = W.dim() == 4
        Wb = torch.cat([W.view(W.shape[0], -1), b[:, None]], 1)
        if relu:
            f = Wb.norm(dim=1).clamp_min(1e-8)          # positive scale
            W.div_(f.view(-1, 1, 1, 1) if conv else f[:, None]); b.div_(f)
            _push_to_next(L[i + 1], f, conv)
        else:                                           # tanh: sign of max-|.| entry
            s = torch.sign(Wb.gather(1, Wb.abs().argmax(1, keepdim=True)).squeeze(1))
            s[s == 0] = 1.0
            W.mul_(s.view(-1, 1, 1, 1) if conv else s[:, None]); b.mul_(s)
            _push_to_next(L[i + 1], s, conv)


@torch.no_grad()
def cnn_align_to_(net, ref):
    """Permute every hidden layer's out-channels/neurons of `net` to match `ref`
    (Hungarian on |flattened filter|), propagating into the next layer. Assumes
    both already canonicalized. In-place on `net`. Returns the list of per-layer
    permutations idx (layers 0..L-2): new position i holds old unit idx[i], so a
    per-unit mask in net's original frame maps to the aligned frame via mask[idx]."""
    L, R = net.layers, ref.layers
    perms = []
    for i in range(len(L) - 1):
        W, b = L[i].weight, L[i].bias
        conv = W.dim() == 4
        sf = W.view(W.shape[0], -1).abs()
        rf = R[i].weight.view(R[i].weight.shape[0], -1).abs()
        D = torch.cdist(rf, sf, p=1).cpu().numpy()
        idx = torch.as_tensor(linear_sum_assignment(D)[1], device=W.device)
        W.copy_(W[idx]); b.copy_(b[idx])
        nxt = L[i + 1]
        if conv and nxt.weight.dim() == 4:              # conv -> conv
            nxt.weight.copy_(nxt.weight[:, idx])
        elif conv and nxt.weight.dim() == 2:            # conv -> linear (blocks)
            k = W.shape[0]; m, n = nxt.weight.shape; sec = n // k
            nxt.weight.copy_(nxt.weight.view(m, k, sec)[:, idx, :].reshape(m, n))
        else:                                           # linear -> *
            nxt.weight.copy_(nxt.weight[:, idx])
        perms.append(idx)
    return perms


@torch.no_grad()
def cnn_param_errors(recon, teacher, hard=False):
    """Max/mean |param| error of a CNN reconstruction after canonicalization +
    channel alignment. Same return shape as param_errors. hard=True: quotient
    the label-preserving head family (hard_fix_out_; the head is Linear)."""
    t, r = teacher.clone(), recon.clone()
    cnn_canonicalize_(t); cnn_canonicalize_(r)
    cnn_align_to_(r, t)
    if hard:
        hard_fix_out_(t, r)
    max_errs, mean_errs = [], []
    for la, lb in zip(t.layers, r.layers):
        for pa, pb in ((la.weight, lb.weight), (la.bias, lb.bias)):
            d = (pa - pb).abs()
            max_errs.append(d.max().item()); mean_errs.append(d.mean().item())
    return {"max_eps": max(max_errs), "mean_eps_per_matrix": mean_errs,
            "max_eps_per_matrix": max_errs}
