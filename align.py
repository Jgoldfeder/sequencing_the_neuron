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

Plain ReLU ('nonleakyrelu') has exactly the leaky-ReLU gauge below (positive
scale, no polarity); the only difference is that a unit can go fully silent.

LeakyReLU is positively homogeneous, so the scaling isomorphism applies
(alpha > 0): scale row c of W_l and b_l by 1/alpha, scale column c of
W_{l+1} by alpha. Polarity does NOT apply (LeakyReLU is not odd).

Sigmoid / tanh have NO scaling isomorphism; their only per-neuron gauge is
polarity: sigma(-z) = 1 - sigma(z) (flip + absorb a constant into the next
bias) and tanh(-z) = -tanh(z) (pure odd flip, nothing to absorb). Both are
canonicalized by sign_canonicalize_.

Square f(z) = z^2 has BOTH: it is even, f(-z) = f(z), so a neuron's row and
bias can be negated with NO change to the next layer at all (nothing to absorb,
no column flip); and it is 2-homogeneous, f(a z) = a^2 f(z), so scaling row c
of W_l and b_l by 1/alpha scales column c of W_{l+1} by alpha^2. Together: any
nonzero real per-neuron scale s on the row with 1/s^2 on the column.
Canonical form: unit-L2 row (norm^2 pushed into the next column) and the
max-|.| entry of [row|bias] positive (sign_canonicalize_ with absorb="even").

GLU / SwiGLU (nets.GatedUnit): a hidden unit is a (value row, gate row) PAIR
of the 2w-row hidden Linear and one next-layer column. The value half is
linear, so any nonzero s on the value row/bias with 1/s on the column is a
gauge (scale p = 1 plus a tanh-style sign flip that touches the value row
only); the gate row has no symmetry. All per-unit operations below act on the
value rows via _unit_rows and permute both halves together (permute_layer_).

PReLU with one shared learned slope a (f(z) = z for z > 0, a z else) has the
ReLU scale gauge (positive, p = 1) and a GLOBAL discrete flip: f_{1/a}(-z) =
-(1/a) f_a(z) for all z, so negating EVERY hidden row/bias, multiplying every
next-layer column by -a and replacing a by 1/a is function-preserving.
Canonical form: |a| <= 1 (_prelu_canonicalize_, before the scale rule). The
slope is a genuine parameter and is included in the eps metrics.

Softmax over a hidden layer's units has NO per-neuron gauge but two LAYER-level
ones: (a) adding the same affine function v.x + c to every unit's pre-activation
(W_l + 1 v^T, b_l + c 1) leaves the softmax unchanged, and (b) since the outputs
sum to 1, the next layer may add u to every column and subtract it from its
bias (W_{l+1} + u 1^T, b_{l+1} - u). Canonical form: zero-mean rows across the
layer's units (mean row and mean bias removed) and zero-mean rows of the next
layer (_softmax_canonicalize_). Softmax is NOT scale invariant (temperature).

Sine sin(z) is odd (tanh-style flip) and PERIODIC: b -> b + 2 pi k is free and
b -> b + pi together with negating the next column is free (sin(z + pi) =
-sin z). Canonical form: sign by the ROW's largest entry (the bias is not a
stable sign key when it is only defined mod pi), then the bias reduced to
[-pi/2, pi/2) with the column sign adjusted (_periodic_reduce_).

Monomials z^k (nets.Power, 'pow3'/'pow5'/...) generalise square: scale gauge
with exponent k (row/a, column*a^k), sign gauge odd (k odd: negate the next
column) or even (k even: nothing). Together: any nonzero s on the row, 1/s^k on
the column.

|z| is even (sign gauge like square: nothing in the next layer) and positively
homogeneous (scale gauge like ReLU: norm pushed linearly, p = 1).

Softsign z/(1+|z|) is odd like tanh: same polarity gauge (sign flip, next
column negated), no scale gauge.

ELU and the other nets.PERMUTATION_ONLY_ACTS are neither homogeneous (no scale
gauge) nor odd / even (no sign gauge; e.g. softplus(-z) = softplus(z) - z
leaves a linear remainder the next layer cannot absorb). Their only per-neuron
symmetry is permutation: the canonical form is the identity and units are
matched unsigned.

The per-neuron gauge helpers below are keyed on the activation:
  _scale_power(net): 1 (leaky-ReLU), 2 (square) or None (sigmoid/tanh/
                     permutation-only);
  _sign_gauge(net):  None (leaky-ReLU, permutation-only), True (sigmoid:
                     absorb), False (tanh: negate the next column) or "even"
                     (square: nothing).
"""
import torch
from scipy.optimize import linear_sum_assignment

import math

from nets import Square, Abs, Power, Sine, GatedUnit, LeCunTanh, LENET5_C3


def _scale_power(net):
    """Exponent p of the scaling isomorphism: scaling a hidden neuron's row (and
    bias) by 1/a is undone by scaling its next-layer column by a^p. 1 for
    (leaky) ReLU (positively homogeneous, a > 0), 2 for square (a != 0),
    None for sigmoid / tanh / permutation-only (no scaling isomorphism)."""
    if isinstance(net.act, (torch.nn.LeakyReLU, torch.nn.ReLU, torch.nn.PReLU, Abs, GatedUnit)):
        return 1
    if isinstance(net.act, Square):
        return 2
    if isinstance(net.act, Power):
        return net.act.k
    return None


@torch.no_grad()
def _periodic_reduce_(net, layers=None):
    """Sine: reduce every hidden bias to [-pi/2, pi/2) by b -> b - k pi and negate the
    next column by (-1)^k (sin(z - k pi) = (-1)^k sin z). Function-preserving; no-op
    for other activations. `layers` = hidden layer indices to touch (default all)."""
    if not isinstance(net.act, Sine):
        return
    L = net.layers
    for l in (range(len(L) - 1) if layers is None else layers):
        b = L[l].bias
        k = torch.floor(b / math.pi + 0.5)                  # nearest multiple: residue in [-pi/2, pi/2)
        if not bool(k.any()):
            continue
        b.sub_(k * math.pi)
        sgn = torch.where(k.remainder(2) == 0, 1.0, -1.0).to(b.dtype)
        _push_to_next(L[l + 1], sgn, L[l].weight.dim() == 4)


@torch.no_grad()
def _softmax_canonicalize_(net):
    """Softmax hidden layers: remove the shared affine shift (mean row / mean bias
    over the layer's units) and the next layer's per-row constant (row mean of
    W_{l+1} moved into b_{l+1}; valid because the softmax outputs sum to 1).
    Function-preserving; no-op for other activations."""
    if not isinstance(net.act, torch.nn.Softmax):
        return
    L = net.layers
    for l in range(len(L) - 1):
        W, b = L[l].weight, L[l].bias
        W.sub_(W.mean(0, keepdim=True)); b.sub_(b.mean())
        nxt = L[l + 1]
        u = nxt.weight.mean(1)
        nxt.weight.sub_(u[:, None]); nxt.bias.add_(u)


@torch.no_grad()
def _prelu_canonicalize_(net):
    """PReLU (one shared slope a): if |a| > 1 apply the global flip -- negate every
    hidden row and bias, multiply every next-layer column by -a, set a <- 1/a --
    so the canonical slope satisfies |a| <= 1. Function-preserving; no-op otherwise."""
    if not isinstance(net.act, torch.nn.PReLU):
        return
    a = net.act.weight
    if a.numel() != 1:
        raise ValueError("PReLU gauge: expected ONE shared slope (nn.PReLU())")
    av = float(a.item())
    if abs(av) <= 1.0 or av == 0.0:
        return
    L = net.layers
    for l in range(len(L) - 1):
        L[l].weight.mul_(-1.0); L[l].bias.mul_(-1.0)
        L[l + 1].weight.mul_(-av)
    a.fill_(1.0 / av)


@torch.no_grad()
def _pre_canonicalize_(net):
    """Layer-level gauges that must be fixed BEFORE the per-neuron scale/sign rules
    and before activation pairing: the softmax shifts, the PReLU global flip.
    Function-preserving; no-op for other activations."""
    _softmax_canonicalize_(net)
    _prelu_canonicalize_(net)


def _needs_pre_canon(net):
    return isinstance(net.act, (torch.nn.Softmax, torch.nn.PReLU))


def _act_param_pairs(t, r):
    """(teacher, recon) pairs of ACTIVATION parameters (PReLU slope); empty otherwise."""
    return list(zip(t.act.parameters(), r.act.parameters()))


@torch.no_grad()
def _layer_canonicalize_(net):
    """Layer-level (not per-neuron sign/scale) parts of the canonical form that
    come AFTER the sign rule: the periodic bias reduction of sine.
    Function-preserving; no-op for activations without such a gauge."""
    _periodic_reduce_(net)


def _is_gated(net):
    return isinstance(net.act, GatedUnit)


def _unit_rows(net, l):
    """(W, b) rows owned by the hidden UNITS of layer l for the per-unit gauge: the
    whole layer for ordinary activations, the VALUE half (first w rows) for gated
    units. Views: in-place ops write through."""
    W, b = net.layers[l].weight, net.layers[l].bias
    if _is_gated(net):
        w = W.shape[0] // 2
        return W[:w], b[:w]
    return W, b


def _flip_next_(nxt, flip, s, absorb):
    """Next-layer side of negating the hidden neurons `flip` (bool mask; s = -1
    there, +1 elsewhere). sigmoid (absorb=True): add the flipped columns into the
    bias (1 - sigma) then negate them; tanh (absorb=False): negate the columns;
    square (absorb="even"): nothing -- (-z)^2 = z^2."""
    if absorb == "even":
        return
    if absorb:
        nxt.bias.add_(nxt.weight[:, flip].sum(1))
    nxt.weight.mul_(s.unsqueeze(0))


@torch.no_grad()
def sign_canonicalize_(net, absorb=True):
    """Canonicalize the polarity isomorphism of a symmetric activation:
    sigmoid sigma(-z) = 1 - sigma(z) (absorb=True), tanh tanh(-z) = -tanh(z)
    (absorb=False) or square (-z)^2 = z^2 (absorb="even"). Flip every hidden neuron whose
    largest-|.| entry of [row|bias] is negative. (The refactor Standardizer
    keyed on the SUM of the row; that is unstable -- a row whose entries sum
    to ~0 flips under 1e-2 noise and scores as a ~2||w|| error -- whereas the
    max-|.| entry only flips if the top two entries tie.) Flipping neuron k
    of layer l: negate row k of W_l/b_l; for
    sigmoid ABSORB column k of W_{l+1} into b_{l+1} (the "1" in 1 - sigma;
    tanh needs no such term); then negate that column (tanh, sigmoid) or
    leave the next layer untouched (square: even activation).
    Function-preserving and exact; in-place."""
    use_bias = not isinstance(net.act, Sine)          # sine: the bias is only defined mod pi -> key on the row
    for l in range(len(net.layers) - 1):
        W, b = _unit_rows(net, l)                      # (w, in_l): value rows for gated units
        Wb = torch.cat([W, b[:, None]], 1) if use_bias else W
        s = torch.sign(Wb.gather(1, Wb.abs().argmax(1, keepdim=True)).squeeze(1))
        s[s == 0] = 1.0
        flip = s < 0
        if not flip.any():
            _periodic_reduce_(net, [l])
            continue
        _flip_next_(net.layers[l + 1], flip, s, absorb)
        W.mul_(s.unsqueeze(1))
        b.mul_(s)
        _periodic_reduce_(net, [l])                   # sine: reduce b_l (negates columns of l+1) BEFORE l+1's sign rule


@torch.no_grad()
def sign_align_clone(net, Zref, X, odd=None):
    """Clone of `net` whose hidden units are paired to a REFERENCE net (whose hidden
    pre-activations on X are Zref, a list per hidden layer) and, for activations with
    a polarity gauge (tanh / sigmoid / square), flipped so that each unit's polarity
    matches its reference partner's. Returns (clone, perms, corrs): perms[l][i] = clone unit paired with
    reference unit i of hidden layer l, corrs[l][i] = |correlation| of the pair.

    Why not the max-|entry| sign rule (sign_canonicalize_): while rows are still
    imprecise (mid-transition) that rule flips a few units differently in different
    members and in the black box; every such unit negates one COLUMN of the next
    layer, so a max-element row test reads ~2x a typical entry (~0.13) on every row of
    that layer although the weights agree. The activation sign is unambiguous whenever
    the pairing is (|corr| ~ 1). Leaky-ReLU has no polarity gauge: no flips, plain corr."""
    from head_refine import _act_perm
    mc = net.clone()
    _pre_canonicalize_(mc); _layer_canonicalize_(mc)
    absorb = _sign_gauge(mc)
    if odd is None:
        odd = absorb is not None
    Z = _preacts(mc, X)                       # flipping layer l-1 units leaves layer l pre-activations unchanged
    perms, corrs = [], []
    for l in range(len(mc.layers) - 1):
        A = Zref[l] - Zref[l].mean(0); B = Z[l] - Z[l].mean(0)
        A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12); B = B / B.norm(dim=0, keepdim=True).clamp_min(1e-12)
        C = A.T @ B
        if odd:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment((-torch.nan_to_num(C.abs(), nan=0.0, posinf=0.0, neginf=0.0)).cpu().numpy())
            perm = torch.empty(C.shape[0], dtype=torch.long, device=X.device); perm[torch.as_tensor(rows, device=X.device)] = torch.as_tensor(cols, device=X.device)
            sgn = torch.sign(C[rows, cols]); sgn[sgn == 0] = 1.0
            flip_units = perm[sgn < 0]
            if flip_units.numel():
                mask = torch.zeros(C.shape[1], dtype=torch.bool, device=X.device); mask[flip_units] = True
                flip_neurons_(mc, l, mask, absorb)
            corrs.append(C[rows, cols].abs()); perms.append(perm)
        else:
            perm, c = _act_perm(Zref[l], Z[l]); perms.append(perm); corrs.append(c)
    return mc, perms, corrs


def pairing_preacts(net, X):
    """Pre-activations of a CANONICALISED copy of `net` for activation pairing.
    Correlation is scale-invariant, so for leaky-ReLU this equals _preacts(net, X);
    for tanh/sigmoid/square the sign canonicalisation fixes the polarity gauge
    tanh(-z) = -tanh(z): without it half the members' units are anti-correlated
    (-0.99) with the reference's and the Hungarian pairing misses them."""
    mc = net.clone(); scale_normalize_(mc)
    return _preacts(mc, X)


@torch.no_grad()
def scale_normalize_(net):
    """Canonicalize the net's activation isomorphism, in-place.

    (Leaky)ReLU (positively homogeneous): every hidden neuron's incoming row
    gets unit L2 norm; the scale is pushed into the next layer's column.
    Sigmoid / tanh (no scaling isomorphism): polarity canonicalization
    instead, via sign_canonicalize_ (sigmoid absorbs the 1 - sigma constant
    into the next bias; tanh is odd so nothing is absorbed).
    Square (2-homogeneous AND even): unit L2 row with the SQUARED norm pushed
    into the next column, then polarity (row/bias sign, next layer untouched).
    Either way, functionally equivalent nets emerge with identical parameters
    up to neuron permutation, so downstream matching / clustering / averaging
    operate on comparable raw weights."""
    _pre_canonicalize_(net)                            # softmax shifts (layer-level, before per-neuron rules)
    p = _scale_power(net)
    if p is not None:
        for l in range(len(net.layers) - 1):
            W, b = _unit_rows(net, l)                  # (w, in_l): value rows for gated units
            norms = W.norm(dim=1).clamp_min(1e-12)  # per hidden neuron
            W.div_(norms.unsqueeze(1))
            b.div_(norms)
            net.layers[l + 1].weight.mul_((norms ** p).unsqueeze(0))
    absorb = _sign_gauge(net)
    if absorb is not None:
        sign_canonicalize_(net, absorb=absorb)
    _layer_canonicalize_(net)                          # sine: bias mod pi (after the sign rule)


@torch.no_grad()
def _match_features(net, l):
    """Feature vector per hidden neuron of layer l for matching (gated units:
    [value row | value bias | gate row | gate bias])."""
    W = net.layers[l].weight
    b = net.layers[l].bias.unsqueeze(1)
    if _is_gated(net):
        w = W.shape[0] // 2
        return torch.cat([W[:w], b[:w], W[w:], b[w:]], dim=1)
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
    # the greedy sweep is n sequential argmins over an n x n matrix: run it on the CPU in numpy
    # (one device->host copy) instead of n GPU argmin+sync round trips (was ~14 ms per call at
    # n=200, ~2 ms now). Same first-minimum tie-breaking as torch.argmin on the flattened matrix.
    import numpy as np
    Dn_ = D.detach().cpu().numpy().copy()
    negn = neg.detach().cpu().numpy() if signed else None
    perm = [None] * n
    flipn = np.zeros(n, dtype=bool)
    INF = np.inf
    for _ in range(n):
        idx = int(Dn_.argmin())
        i, j = idx // n, idx % n
        perm[i] = j
        if signed:
            flipn[j] = negn[i, j]
        Dn_[i, :] = INF
        Dn_[:, j] = INF
    flip = torch.from_numpy(flipn).to(D.device)
    return (perm, flip) if signed else perm


@torch.no_grad()
def flip_neurons_(net, l, flip, absorb):
    """Negate hidden neurons `flip` (bool mask) of layer l, function-
    preserving: negate their rows/biases, for sigmoid (absorb=True) add the
    affected columns of W_{l+1} into b_{l+1} (1 - sigma), negate the columns
    (sigmoid, tanh); square (absorb="even") leaves the next layer as is."""
    if not bool(flip.any()):
        return
    s = torch.where(flip, -1.0, 1.0).to(net.layers[l].weight.dtype)
    _flip_next_(net.layers[l + 1], flip, s, absorb)
    W, b = _unit_rows(net, l)                          # value rows only for gated units
    W.mul_(s.unsqueeze(1))
    b.mul_(s)
    _periodic_reduce_(net, [l])                        # sine: -b may leave [-pi/2, pi/2) at the boundary


def _sign_gauge(net):
    """None for (leaky)ReLU (no polarity gauge; its scale gauge is handled by
    scale_normalize_); else the absorb flag for the activation's polarity
    gauge: True (sigmoid), False (tanh), "even" (square: next layer untouched).
    None also for the permutation-only activations (ELU, ...)."""
    if isinstance(net.act, torch.nn.Sigmoid):
        return True
    if isinstance(net.act, (torch.nn.Tanh, LeCunTanh, torch.nn.Softsign, Sine, GatedUnit)):   # odd / linear value half: negate the next column
        return False
    if isinstance(net.act, Power):
        return False if net.act.k % 2 else "even"
    if isinstance(net.act, (Square, Abs)):
        return "even"
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
    if _is_gated(net):                                 # permute value AND gate rows together
        w = net.layers[l].weight.shape[0] // 2
        idx = torch.cat([idx, idx + w])
    net.layers[l].weight.copy_(net.layers[l].weight[idx])
    net.layers[l].bias.copy_(net.layers[l].bias[idx])
    W_next = net.layers[l + 1].weight
    W_next.copy_(W_next[:, idx[:len(perm)]])


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
    for pa, pb in _act_param_pairs(t, r):              # PReLU slope
        d = (pa - pb).abs(); max_errs.append(d.max().item()); mean_errs.append(d.mean().item())
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
    ReLU: unit-L2 per out-unit; tanh: sign (odd flip); square: unit-L2 with the
    squared norm pushed, then sign (even: nothing pushed); permutation-only
    activations (ELU, ...): nothing. Function-preserving; the factor is pushed
    into the next layer. The final Linear head is left as-is (its outputs are
    the fixed logits -- no output gauge)."""
    L = net.layers
    if getattr(net, "lenet5", False):
        _lenet5_canonicalize_(net); return
    p, sg = _scale_power(net), _sign_gauge(net)        # same gauge table as the MLP path
    if sg is True:
        raise ValueError("cnn_canonicalize_: sigmoid (absorbing flip) is not supported for ConvNets")
    if p is None and sg is None:
        return                                          # permutation only: identity canonical form
    for i in range(len(L) - 1):
        W, b = L[i].weight, L[i].bias
        conv = W.dim() == 4
        Wb = torch.cat([W.view(W.shape[0], -1), b[:, None]], 1)
        if p is not None:
            f = Wb.norm(dim=1).clamp_min(1e-8)          # positive scale
            W.div_(f.view(-1, 1, 1, 1) if conv else f[:, None]); b.div_(f)
            _push_to_next(L[i + 1], f ** p, conv)
        if sg is not None:                              # sign of max-|.| entry (row only for sine)
            Wb = torch.cat([W.view(W.shape[0], -1), b[:, None]], 1) if not isinstance(net.act, Sine) else W.view(W.shape[0], -1)
            s = torch.sign(Wb.gather(1, Wb.abs().argmax(1, keepdim=True)).squeeze(1))
            s[s == 0] = 1.0
            W.mul_(s.view(-1, 1, 1, 1) if conv else s[:, None]); b.mul_(s)
            if sg is False:                             # odd: negate the next column; even: nothing
                _push_to_next(L[i + 1], s, conv)
        _periodic_reduce_(net, [i])                     # sine: bias mod pi, before layer i+1's sign rule


def _lenet5_flip_(net, i, k, s):
    """multiply hidden unit k of weight layer i by s (+-1) in a function-preserving way for LeNet-5 (odd squashing):
    row and bias of layer i; if a subsampling layer follows, its coefficient absorbs the flip (c -> s c) so nothing
    else changes; otherwise the next layer's input column(s) for unit k are multiplied by s."""
    L = net.layers; lay = L[i]
    lay.weight[k].mul_(s); lay.bias[k].mul_(s)
    if i in net.sub_of:
        net.sub_w[net.sub_of[i]][k].mul_(s)
        return
    _push_unit_(net, i, k, s)


def _push_unit_(net, i, k, s):
    L = net.layers; nxt = L[i + 1]; conv = L[i].weight.dim() == 4
    if conv and nxt.weight.dim() == 4:
        nxt.weight[:, k].mul_(s)
    elif conv and nxt.weight.dim() == 2:
        C = L[i].weight.shape[0]; sec = nxt.weight.shape[1] // C
        nxt.weight[:, k * sec:(k + 1) * sec].mul_(s)
    else:
        nxt.weight[:, k].mul_(s)


@torch.no_grad()
def _lenet5_canonicalize_(net):
    """LeNet-5 gauges (odd squashing => sign flips only, no scale): every hidden unit's row gets the sign of its
    max-|.| entry; a following subsampling unit absorbs it in its coefficient, and then the subsampling unit itself is
    made positive (c > 0; its flip (c, b) -> (-c, -b) is pushed into the next layer's input column).  The RBF head
    takes the push like a linear layer (||(-x) - (-w)|| = ||x - w||)."""
    L = net.layers
    for i in range(len(L) - 1):
        W, b = L[i].weight, L[i].bias
        Wb = torch.cat([W.reshape(W.shape[0], -1), b[:, None]], 1)
        sg = torch.sign(Wb.gather(1, Wb.abs().argmax(1, keepdim=True)).squeeze(1)); sg[sg == 0] = 1.0
        for k in (sg < 0).nonzero().flatten().tolist():
            _lenet5_flip_(net, i, k, -1.0)
        if i in net.sub_of:
            j = net.sub_of[i]; c = net.sub_w[j]
            s2 = torch.sign(c); s2[s2 == 0] = 1.0
            for k in (s2 < 0).nonzero().flatten().tolist():
                net.sub_w[j][k].mul_(-1.0); net.sub_b[j][k].mul_(-1.0)
                _push_unit_(net, i, k, -1.0)


_LENET5_AUTOS = None


def lenet5_c3_automorphisms():
    """(perm1, perm3) pairs: reorderings of the 6 C1/S2 maps and the 16 C3 maps that leave the C3 connection table
    invariant (new C1 map a holds old map perm1[a]; new C3 map r holds old map perm3[r])."""
    global _LENET5_AUTOS
    if _LENET5_AUTOS is None:
        import itertools
        sets = [frozenset(c) for c in LENET5_C3]; out = []
        for p1 in itertools.permutations(range(6)):
            inv = {old: new for new, old in enumerate(p1)}
            img = [frozenset(inv[c] for c in S) for S in sets]          # old C3 map r's inputs, in NEW C1 indices
            if set(img) == set(sets):
                p3 = [img.index(S) for S in sets]                      # new map r = old map p3[r]
                out.append((list(p1), p3))
        _LENET5_AUTOS = out
    return _LENET5_AUTOS


@torch.no_grad()
def _permute_unit_block_(net, i, idx):
    """reorder hidden units of weight layer i by idx (new u = old idx[u]); carries the subsampling params and the
    next layer's input columns (the connection mask of the next layer is NOT permuted: callers must use idx that
    preserve it, i.e. table automorphisms)."""
    L = net.layers; W, b = L[i].weight, L[i].bias
    W.copy_(W[idx]); b.copy_(b[idx])
    if i in net.sub_of:
        j = net.sub_of[i]; net.sub_w[j].copy_(net.sub_w[j][idx]); net.sub_b[j].copy_(net.sub_b[j][idx])
    nxt = L[i + 1]; conv = W.dim() == 4
    if conv and nxt.weight.dim() == 4:
        nxt.weight.copy_(nxt.weight[:, idx])
    elif conv and nxt.weight.dim() == 2:
        k = W.shape[0]; m, n = nxt.weight.shape; sec = n // k
        nxt.weight.copy_(nxt.weight.view(m, k, sec)[:, idx, :].reshape(m, n))
    else:
        nxt.weight.copy_(nxt.weight[:, idx])


@torch.no_grad()
def _lenet5_align_to_(net, ref):
    """LeNet-5 alignment: C1 and C3 maps are pinned by the connection table up to its automorphisms -> pick the
    automorphism with the closest C1 filters (|.|, sign-free); C5 and F6 are freely permutable -> Hungarian.  Then a
    sign-aware pass flips any unit anti-aligned with the reference (the max-|entry| sign rule can tie on inexact
    members).  Returns per-layer permutations like cnn_align_to_."""
    L, R = net.layers, ref.layers
    rf = R[0].weight.reshape(R[0].weight.shape[0], -1).abs(); sf = L[0].weight.reshape(L[0].weight.shape[0], -1).abs()
    best = None
    for p1, p3 in lenet5_c3_automorphisms():
        c = float((rf - sf[p1]).abs().sum())
        if best is None or c < best[0]:
            best = (c, p1, p3)
    _, p1, p3 = best
    dev = L[0].weight.device
    idx1 = torch.tensor(p1, device=dev); idx3 = torch.tensor(p3, device=dev)
    _permute_unit_block_(net, 0, idx1)                      # C1 (+S2) rows; C3 input columns
    _permute_unit_block_(net, 1, idx3)                      # C3 (+S4) rows; C5 input columns
    perms = [idx1, idx3]
    for i in range(2, len(L) - 1):                          # C5, F6: free permutations
        W = L[i].weight
        sf_ = W.reshape(W.shape[0], -1).abs(); rf_ = R[i].weight.reshape(R[i].weight.shape[0], -1).abs()
        idx = torch.as_tensor(linear_sum_assignment(torch.cdist(rf_, sf_, p=1).cpu().numpy())[1], device=dev)
        _permute_unit_block_(net, i, idx); perms.append(idx)
    for i in range(len(L) - 1):                             # sign-aware correction
        a = torch.cat([L[i].weight.reshape(L[i].weight.shape[0], -1), L[i].bias[:, None]], 1)
        r_ = torch.cat([R[i].weight.reshape(R[i].weight.shape[0], -1), R[i].bias[:, None]], 1)
        for k in ((a * r_).sum(1) < 0).nonzero().flatten().tolist():
            _lenet5_flip_(net, i, k, -1.0)
            if i in net.sub_of:                             # keep the subsampling coefficient positive (canonical)
                j = net.sub_of[i]
                if float(net.sub_w[j][k]) < 0:
                    net.sub_w[j][k].mul_(-1.0); net.sub_b[j][k].mul_(-1.0); _push_unit_(net, i, k, -1.0)
    return perms


@torch.no_grad()
def cnn_align_to_(net, ref):
    """Permute every hidden layer's out-channels/neurons of `net` to match `ref`
    (Hungarian on |flattened filter|), propagating into the next layer. Assumes
    both already canonicalized. In-place on `net`. Returns the list of per-layer
    permutations idx (layers 0..L-2): new position i holds old unit idx[i], so a
    per-unit mask in net's original frame maps to the aligned frame via mask[idx]."""
    L, R = net.layers, ref.layers
    if getattr(net, "lenet5", False):
        return _lenet5_align_to_(net, ref)
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
    if _sign_gauge(net) is False:                       # odd activations (tanh family): sign-aware correction -- the
        for i in range(len(L) - 1):                     # max-|entry| sign rule can tie on inexact members
            a = torch.cat([L[i].weight.reshape(L[i].weight.shape[0], -1), L[i].bias[:, None]], 1)
            r_ = torch.cat([R[i].weight.reshape(R[i].weight.shape[0], -1), R[i].bias[:, None]], 1)
            for k in ((a * r_).sum(1) < 0).nonzero().flatten().tolist():
                _lenet5_flip_(net, i, k, -1.0)
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


# ------------------------------------------------ activation-matched eps --
@torch.no_grad()
def _preacts(net, X):
    """Hidden-layer pre-activations [(n, w_l)] of an MLP on probe X. Activations with a
    layer-level shift gauge (softmax) are read in their canonical gauge (clone), so the
    unit correlations of two equivalent nets agree."""
    if _needs_pre_canon(net):
        net = net.clone(); _pre_canonicalize_(net)
    gated = _is_gated(net)
    zs = []
    h = X
    for layer in net.layers[:-1]:
        z = layer(h)
        h = net.act(z)
        zs.append(h if gated else z)                   # gated: one feature per UNIT (a*gate(g); sign/scale follow the value row)
    return zs


@torch.no_grad()
def param_errors_actmatch(recon, teacher, probe=None, n_probe=4096, seed=0,
                          hard=False, return_match=False, head_opt=True):
    """'align-inverse' metric: per-layer parameter error with units paired by
    ACTIVATION matching, independently per layer, instead of the chained
    weight-permutation alignment of param_errors.

    param_errors matches layer 1 by weights, then layer 2 in the coordinates
    that match induces, etc., so a layer's eps inherits any mis-pairing below
    it (a wrong L1 pairing scrambles L2's input columns even when L2 is right
    in the recon's own frame). Here every hidden layer l is paired by the
    correlation of its units' pre-activations on a probe set, each net run
    with ITS OWN prefix, via an optimal assignment (Hungarian). Layer l's
    weight matrix is then compared with rows ordered by layer l's pairing and
    columns by layer l-1's pairing -- no pairing depends on weights below.
    Sign gauge (tanh/sigmoid/square): units may match negated (flipped function-
    preservingly). Scale gauge (leaky relu, square): scale_normalize_ as usual.

    probe: (n, d) inputs on the teacher's device/dtype or None (N(0,1), fixed
    seed). Returns the param_errors dict (+ 'match_corr_per_layer': mean
    matched |corr| per hidden layer, a pairing-confidence diagnostic; and the
    perms when return_match)."""
    from scipy.optimize import linear_sum_assignment
    t = teacher.clone()
    r = recon.clone()
    dt = torch.promote_types(t.layers[0].weight.dtype, r.layers[0].weight.dtype)
    t = t.to(dt)
    r = r.to(dt)
    scale_normalize_(t)
    scale_normalize_(r)
    dev = t.layers[0].weight.device
    if probe is None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        probe = torch.randn(n_probe, t.layers[0].weight.shape[1], generator=g)
    X = probe.to(dev, dt)
    Zt = _preacts(t, X)
    Zr = _preacts(r, X)
    absorb = _sign_gauge(r)
    perms, flips, corrs = [], [], []
    for l in range(len(t.layers) - 1):
        A = Zt[l] - Zt[l].mean(0, keepdim=True)
        B = Zr[l] - Zr[l].mean(0, keepdim=True)
        A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)
        B = B / B.norm(dim=0, keepdim=True).clamp_min(1e-12)
        C = (A.T @ B).float()                      # (w_t, w_r) unit correlations
        S = C.abs() if absorb is not None else C   # sign gauge: negated unit is as good
        S = torch.nan_to_num(S, nan=-1.0, posinf=-1.0, neginf=-1.0)   # a diverged member must not crash the run
        rows, cols = linear_sum_assignment((-S).cpu().numpy())
        perm = [0] * S.shape[0]
        for i, j in zip(rows, cols):
            perm[i] = int(j)
        perms.append(perm)
        flips.append((C[rows, cols] < 0).to(dev) if absorb is not None else None)
        corrs.append(float(S[rows, cols].mean()))
    # apply: flips first (indexed in the recon's ORIGINAL unit order), then the
    # per-layer permutations in order (permute_layer_ touches layer l's rows and
    # layer l+1's columns only, so later perms don't disturb earlier ones)
    for l, fl in enumerate(flips):
        if fl is not None and bool(fl.any()):
            mask = torch.zeros(len(perms[l]), dtype=torch.bool, device=dev)
            mask[torch.tensor(perms[l], device=dev)[fl]] = True
            flip_neurons_(r, l, mask, absorb)
    for l, perm in enumerate(perms):
        permute_layer_(r, l, perm)
    if hard:
        hard_fix_out_(t, r)
    max_errs, mean_errs = [], []
    for la, lb in zip(t.layers, r.layers):
        for pa, pb in zip(la.parameters(), lb.parameters()):
            d = (pa - pb).abs()
            max_errs.append(d.max().item())
            mean_errs.append(d.mean().item())
    act_errs = [((pa - pb).abs().max().item(), (pa - pb).abs().mean().item()) for pa, pb in _act_param_pairs(t, r)]
    out = {
        "max_eps": max(max_errs),
        "mean_eps_per_matrix": mean_errs,
        "max_eps_per_matrix": max_errs,
        "match_corr_per_layer": corrs,
        "align": "actmatch",
    }
    if head_opt and _scale_power(r) is not None:
        # HEAD in its gauge-OPTIMAL frame: the remaining head gauge is a
        # permutation + positive per-column scale of the last hidden units
        # (leaky-ReLU: alpha > 0; square: alpha^2 > 0 -- the sign gauge was
        # already fixed by the activation pairing above).
        # Pick the pairing AND the scales that minimize the head error (the
        # activation pairing + unit-row-norm scale is one admissible choice;
        # this is the best admissible one) -- so head eps is a lower bound over
        # the gauge, charging nothing to the head that a hidden-unit relabel /
        # rescale could absorb.
        A = t.layers[-1].weight                       # (out, w) teacher columns
        B = r.layers[-1].weight                       # (out, w) recon columns, act-paired
        w = A.shape[1]
        # per-pair optimal s>0 minimizing ||a_i - s b_j||: s = max(a.b,0)/|b|^2
        G = A.T @ B                                   # (w_t, w_r) a_i . b_j
        bn = (B * B).sum(0).clamp_min(1e-24)          # |b_j|^2
        an = (A * A).sum(0)                           # |a_i|^2
        S_ = G.clamp_min(0) / bn[None, :]
        cost = (an[:, None] - 2 * S_ * G + S_ ** 2 * bn[None, :]).clamp_min(0)  # ||a_i - s b_j||^2
        rows, cols = linear_sum_assignment(torch.nan_to_num(cost.float(), nan=1e30, posinf=1e30, neginf=1e30).cpu().numpy())
        hp = [0] * w
        for i, j in zip(rows, cols):
            hp[i] = int(j)
        idx = torch.tensor(hp, device=dev)
        sc = S_[rows, cols]                            # (w,) in teacher order
        Bh = B[:, idx] * sc[None, :]
        dW = (A - Bh).abs()
        db = (t.layers[-1].bias - r.layers[-1].bias).abs()
        out["head_opt"] = {
            "max_eps_W": dW.max().item(), "mean_eps_W": dW.mean().item(),
            "max_eps_b": db.max().item(), "mean_eps_b": db.mean().item(),
            "pair_agree_with_act": float(sum(1 for i in range(w) if hp[i] == i)) / w,
            "scale_median": sc.median().item(),
            "scale_minmax": (sc.min().item(), sc.max().item()),
        }
        # replace the head's entries in the per-matrix lists with the gauge-optimal ones
        max_errs[-2], mean_errs[-2] = out["head_opt"]["max_eps_W"], out["head_opt"]["mean_eps_W"]
        max_errs[-1], mean_errs[-1] = out["head_opt"]["max_eps_b"], out["head_opt"]["mean_eps_b"]
        out["max_eps"] = max(max_errs)
        out["align"] = "actmatch+headopt"
    if act_errs:                                       # PReLU slope, appended after the head
        max_errs.extend(e[0] for e in act_errs); mean_errs.extend(e[1] for e in act_errs)
        out["max_eps"] = max(max_errs)
    if return_match:
        out["perms"] = perms
    return out


@torch.no_grad()
def act_align_members(pop, idxs, ref_idx, probe):
    """Per-layer ISOLATED alignment by activation pairing (the tanh-style consensus pairing): every hidden layer l of
    every member is paired to the reference member's layer l by the Hungarian assignment on pre-activation
    correlations over `probe`, each net run with its OWN prefix. The pairing of layer l does not depend on the weights
    or the consensus of the layers below, and it is a full permutation, so layer l+1's input columns are reordered
    consistently even for units that are not (yet) in consensus. Returns (aligned clones, normalised ref)."""
    from head_refine import _act_perm
    ref = pop[ref_idx].clone(); scale_normalize_(ref)
    X = probe.to(ref.layers[0].weight.device, ref.layers[0].weight.dtype)
    Zref = _preacts(ref, X)
    out = []
    for i in idxs:
        c = pop[i].clone(); scale_normalize_(c)
        Z = _preacts(c, X)
        for l in range(len(c.layers) - 1):
            perm, _ = _act_perm(Zref[l], Z[l])
            permute_layer_(c, l, perm.tolist())
            # permute_layer_ reorders layer l's rows and layer l+1's columns; pre-activations of layer l+1 are unchanged
        out.append(c)
    return out, ref
