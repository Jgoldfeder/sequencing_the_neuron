"""MLP definitions and initialization matching the paper's assumptions.

Paper assumptions (Sec 3.1): weights initialized mean 0, std sqrt(2/(nin+nout))
(Glorot-style), networks trained with first-order optimization. Hidden
activations: LeakyReLU (paper uses LeakyReLU to avoid dying ReLUs).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Square(nn.Module):
    """Hidden activation f(z) = z^2. Even (f(-z) = f(z)) and 2-homogeneous
    (f(a z) = a^2 f(z)), so a hidden neuron's per-unit gauge is row -> s*row for
    ANY nonzero s with the next layer's column scaled by 1/s^2 (align.py)."""

    def forward(self, z):
        return z * z


class GatedUnit(nn.Module):
    """GLU-style gated hidden unit. The hidden Linear produces 2w pre-activations:
    value half a (rows 0..w-1) and gate half g (rows w..2w-1); the unit outputs
    a * gate(g) with gate = sigmoid ('glu') or SiLU ('swiglu'). A unit therefore
    owns TWO rows of W_l (value, gate) and ONE column of W_{l+1}. Gauge: any
    nonzero scale s on the VALUE row/bias with 1/s on the column (sign included:
    the value half is linear); the gate row has no symmetry (sigmoid(-g) = 1 -
    sigmoid(g) leaves a linear remainder a - a*sigmoid(g) that the next layer
    cannot absorb). See align: unit rows via _unit_rows."""

    def __init__(self, gate="sigmoid"):
        super().__init__()
        self.gate_name = gate
        self.gate = torch.sigmoid if gate == "sigmoid" else F.silu

    def forward(self, z):
        a, g = z.chunk(2, dim=-1)
        return a * self.gate(g)


class Sine(nn.Module):
    """sin(z) (SIREN-style, omega = 1). Odd (tanh-style sign gauge) AND periodic:
    bias + 2*pi*k is free, bias + pi with the next column negated is free. Canonical
    form (align): sign of the ROW's max-|.| entry positive, bias reduced to
    [-pi/2, pi/2) with the column sign adjusted."""

    def forward(self, z):
        return torch.sin(z)


class Power(nn.Module):
    """Monomial z^k (names 'pow3', 'pow5', ...): k-homogeneous, f(a z) = a^k f(z), so a
    row scaled by 1/a pushes a^k into the next column; odd k is odd (sign flip negates
    the next column, like tanh), even k is even (next layer untouched, like square).
    Square is the k = 2 case (kept as its own class for the existing code paths)."""

    def __init__(self, k):
        super().__init__(); self.k = int(k)

    def forward(self, z):
        return z ** self.k


class Abs(nn.Module):
    """|z|: even (sign gauge, next layer untouched) AND positively homogeneous
    (scale gauge, norm pushed linearly) -- ReLU's scale gauge plus square's sign gauge."""

    def forward(self, z):
        return z.abs()


class Squareplus(nn.Module):
    """squareplus(z) = (z + sqrt(z^2 + b)) / 2 (b = 4): a smooth ReLU like softplus;
    squareplus(-z) = squareplus(z) - z leaves a linear remainder, so no sign gauge."""

    def __init__(self, b=4.0):
        super().__init__(); self.b = float(b)

    def forward(self, z):
        return 0.5 * (z + torch.sqrt(z * z + self.b))


# Hidden activations with NO per-neuron gauge beyond permutation: not homogeneous
# (no scale gauge) and neither odd nor even (no sign gauge). align.py treats them
# as "permutation only": canonical form = identity, unsigned matching.
PERMUTATION_ONLY_ACTS = ("elu", "gelu", "silu", "mish", "softplus", "celu", "selu", "hardswish", "squareplus",)


class LeCunTanh(nn.Module):
    """LeNet-5 (LeCun et al. 1998, appendix A) squashing function f(a) = A tanh(S a), A = 1.7159, S = 2/3. Odd."""
    A, S = 1.7159, 2.0 / 3.0

    def forward(self, x):
        return self.A * torch.tanh(self.S * x)


# LeNet-5 C3 connection table (LeCun et al. 1998, Table I): LENET5_C3[r] = S2 maps feeding C3 map r
LENET5_C3 = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (0, 4, 5), (0, 1, 5),
             (0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (0, 3, 4, 5), (0, 1, 4, 5), (0, 1, 2, 5),
             (0, 1, 3, 4), (1, 2, 4, 5), (0, 2, 3, 5),
             (0, 1, 2, 3, 4, 5)]


def lenet5_c3_mask():
    m = torch.zeros(16, 6)
    for r, cols in enumerate(LENET5_C3):
        m[r, list(cols)] = 1.0
    return m


def make_act(act, negative_slope=0.01):
    """Hidden activation module by name (shared by MLP and ConvNet)."""
    if act == "leaky_relu":
        return nn.LeakyReLU(negative_slope=negative_slope)
    if act in ("relu", "nonleakyrelu"):
        return nn.ReLU()             # plain ReLU ('nonleakyrelu' for MLPs): same positive scale gauge as leaky
    if act == "sigmoid":
        return nn.Sigmoid()
    if act == "tanh":
        return nn.Tanh()
    if act in ("lecun_tanh", "lenet5", "lenet5lin"):
        return LeCunTanh()
    if act == "square":
        return Square()
    if act == "elu":
        return nn.ELU()              # alpha = 1
    if act == "gelu":
        return nn.GELU()  # exact erf form; GELU' = Phi(z) + z phi(z)
    if act == "silu":
        return nn.SiLU()  # z*sigmoid(z); SiLU' = s(1 + z(1-s))
    if act == "mish":
        return nn.Mish()  # z*tanh(softplus(z)); Mish' = t + z*sigmoid(z)*(1-t^2), t = tanh(softplus(z))
    if act == "softplus":
        return nn.Softplus()  # log(1+e^z); softplus(-z) = softplus(z) - z is NOT a gauge (linear remainder)
    if act == "softsign":
        return nn.Softsign()  # z/(1+|z|): ODD -> tanh-style sign gauge (align._sign_gauge)
    if act == "celu":
        return nn.CELU()  # alpha = 1: identical to ELU
    if act == "selu":
        return nn.SELU()  # lambda*elu(z, alpha) with the self-normalising constants
    if act == "hardswish":
        return nn.Hardswish()  # z*relu6(z+3)/6: piecewise, permutation only
    if act == "squareplus":
        return Squareplus()  # (z + sqrt(z^2+4))/2, smooth ReLU; permutation only
    if act == "sine":
        return Sine()
    if act == "softmax":
        return nn.Softmax(dim=-1)    # over the layer's units: layer-level gauges (align._softmax_canonicalize_)
    if act == "glu":
        return GatedUnit("sigmoid")
    if act == "swiglu":
        return GatedUnit("silu")
    if act == "prelu":
        return nn.PReLU()            # ONE learned slope a (init 0.25) shared by all hidden units; it is a
                                     # parameter of the net (trained, compared by param_errors). Gauge: ReLU
                                     # scale + a GLOBAL flip a -> 1/a (align._prelu_canonicalize_: |a| <= 1)
    if act.startswith("pow") and act[3:].isdigit():
        return Power(int(act[3:]))   # z^k monomial (odd k = odd gauge, even k = square-like)
    if act == "abs":
        return Abs()  # |z|: even + positively homogeneous (sign gauge w/ next layer untouched + linear scale push)
    raise ValueError(f"unknown activation {act!r}")


def init_layer(layer: nn.Linear) -> None:
    std = math.sqrt(2.0 / (layer.in_features + layer.out_features))
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """Feed-forward net, LeakyReLU (default), sigmoid, tanh, square (z^2) or
    any of PERMUTATION_ONLY_ACTS as hidden activation, linear output."""

    def __init__(self, dims, negative_slope=0.01, act="leaky_relu"):
        super().__init__()
        self.dims = list(dims)
        self.act_name = act
        self.gated = act in ("glu", "swiglu")       # hidden Linear has 2*width rows (value | gate)
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], (2 if self.gated and i < len(dims) - 2 else 1) * dims[i + 1])
             for i in range(len(dims) - 1)]
        )
        if act == "relu":
            raise ValueError("MLP uses leaky_relu, not relu (kink refiners assume it)")
        self.act = make_act(act, negative_slope)
        for layer in self.layers:
            init_layer(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)

    def clone(self):
        # Build the copy on the meta device (no random init, no allocation), materialise it
        # uninitialised at the source's device/dtype, then load. The old path constructed a
        # real MLP (Glorot normal_ + nn.Linear's own uniform_ on every layer) just to overwrite
        # it: ~8 ms per clone, 1.2 s per consensus pass. dtype is matched BEFORE the load, else
        # an fp64 source would be truncated to fp32 (caps param_errors at ~6e-8).
        p = next(self.parameters())
        with torch.device("meta"):
            new = MLP(self.dims, getattr(self.act, "negative_slope", 0.01), self.act_name)
        new = new.to_empty(device=p.device).to(dtype=p.dtype)
        new.load_state_dict(self.state_dict())
        return new


class ConvNet(nn.Module):
    """CNN for the extraction pipeline. conv_cfgs is a list of conv specs
    (in, out, kernel, stride[, pad[, pool]]) -- pad defaults to 1, pool (avg-pool
    size, kernel=stride) defaults to 0 (none). fc_dims are hidden fully-connected
    widths between the conv stack and the Linear output head. Accepts FLAT input
    (N, C*H*W) and reshapes internally, so the pipeline's flat query machinery
    works unchanged. Avg-pool is used (commutes with the channel scale/sign gauge,
    so alignment is unaffected)."""

    def __init__(self, input_shape, conv_cfgs, fc_dims=(), out_dim=10, act="relu"):
        super().__init__()
        self.input_shape = tuple(input_shape)      # (C, H, W)
        self.out_dim = out_dim
        self.conv_cfgs = [self._norm_cfg(c) for c in conv_cfgs]
        self.fc_dims = tuple(int(x) for x in fc_dims)
        self.act_name = act
        if act in ("sigmoid", "softmax", "prelu", "glu", "swiglu"):
            raise ValueError(f"ConvNet does not support {act}")
        self.act = make_act(act)
        self.layers = nn.ModuleList()
        self.pools = []                             # per-conv avg-pool size (0=none)
        cur = self.input_shape
        for (ic, oc, k, s, pad, pool) in self.conv_cfgs:
            self.layers.append(nn.Conv2d(ic, oc, k, s, padding=pad))
            cur = self._out_shape(cur, oc, k, s, pad)
            if pool > 0:
                cur = (cur[0], (cur[1] - pool) // pool + 1, (cur[2] - pool) // pool + 1)
            self.pools.append(pool)
        self.n_conv = len(self.conv_cfgs)
        prev = cur[0] * cur[1] * cur[2]
        self.feat_dim = prev
        for h in self.fc_dims:                      # hidden FC layers
            self.layers.append(nn.Linear(prev, h)); prev = h
        self.layers.append(nn.Linear(prev, out_dim))  # output head
        for l in self.layers:
            fan_in = l.weight.shape[1] if l.weight.dim() == 2 else l.weight[0].numel()
            nn.init.normal_(l.weight, 0.0, math.sqrt(2.0 / (fan_in + l.weight.shape[0])))
            nn.init.zeros_(l.bias)
        # ---- LeNet-5 after LeCun et al. 1998 (act="lenet5"): scaled tanh everywhere, the C3 partial-connection table,
        #      and Euclidean RBF output units (output = -||x - w_i||^2, so argmax / cross-entropy conventions hold).
        #      S2/S4 are PLAIN 2x2 average pooling (the paper's trainable subsampling coefficient/bias + second squashing
        #      is deliberately not used; sub_of stays empty, the gauge code keeps supporting it).
        self.lenet5 = act in ("lenet5", "lenet5lin")
        self.rbf = (act == "lenet5")                 # "lenet5lin": same net with a plain linear output layer
        self.sub_of = {}
        self.masked = []
        if self.lenet5:
            for i, (ic, oc, k, s_, pad, pool) in enumerate(self.conv_cfgs):
                if ic == 6 and oc == 16:
                    self.register_buffer(f"conn_mask{i}", lenet5_c3_mask()[:, :, None, None])
                    self.masked.append(i)
                    with torch.no_grad():
                        self.layers[i].weight.mul_(getattr(self, f"conn_mask{i}"))
            if self.rbf:
                with torch.no_grad():
                    self.layers[-1].bias.zero_()
                self.layers[-1].bias.requires_grad_(False)      # RBF units have no bias

    @staticmethod
    def _norm_cfg(c):
        c = tuple(int(x) for x in c)
        if len(c) == 4:                             # (in,out,k,s) -> pad=1, pool=0
            return (*c, 1, 0)
        if len(c) == 5:                             # (in,out,k,s,pad) -> pool=0
            return (*c, 0)
        return c[:6]

    @staticmethod
    def _out_shape(shape, oc, k, s, p):
        _, h, w = shape
        return oc, (h - k + 2 * p) // s + 1, (w - k + 2 * p) // s + 1

    def conv_weight(self, i):
        """effective weight of conv layer i (connection-table mask applied for LeNet-5 C3)"""
        w = self.layers[i].weight
        return w * getattr(self, f"conn_mask{i}").to(w.dtype) if i in self.masked else w

    def _conv(self, i, x):
        lay = self.layers[i]
        if i in self.masked:
            return F.conv2d(x, self.conv_weight(i), lay.bias, lay.stride, lay.padding)
        return lay(x)

    def _sub(self, i, x):
        """pooling after conv i (+ the LeNet-5 trainable subsampling affine and squashing)"""
        if self.pools[i] > 0:
            x = F.avg_pool2d(x, self.pools[i])
            if i in self.sub_of:
                j = self.sub_of[i]
                x = self.act(x * self.sub_w[j].view(1, -1, 1, 1) + self.sub_b[j].view(1, -1, 1, 1))
        return x

    def head(self, x):
        if getattr(self, "rbf", False):             # Euclidean RBF output units, sign-flipped: -||x - w_i||^2
            W = self.layers[-1].weight
            return -((x * x).sum(1, keepdim=True) - 2 * x @ W.T + (W * W).sum(1)[None, :])
        return self.layers[-1](x)

    def forward(self, x):
        if x.dim() == 2:                            # flat -> image
            x = x.view(x.shape[0], *self.input_shape)
        for i in range(self.n_conv):
            x = self._sub(i, self.act(self._conv(i, x)))
        x = torch.flatten(x, 1)
        for j in range(self.n_conv, len(self.layers) - 1):   # hidden FC
            x = self.act(self.layers[j](x))
        return self.head(x)

    def forward_preacts(self, x):
        """pre-activations of every hidden weight layer (conv: (N, C, H, W), fc: (N, C)), in layer order"""
        if x.dim() == 2:
            x = x.view(x.shape[0], *self.input_shape)
        out = []
        for i in range(self.n_conv):
            z = self._conv(i, x); out.append(z); x = self._sub(i, self.act(z))
        x = torch.flatten(x, 1)
        for j in range(self.n_conv, len(self.layers) - 1):
            z = self.layers[j](x); out.append(z); x = self.act(z)
        return out

    def clone(self):
        new = ConvNet(self.input_shape, self.conv_cfgs, self.fc_dims,
                      self.out_dim, self.act_name)
        p = next(self.parameters())
        # match dtype BEFORE load_state_dict: loading fp64 weights into a fresh
        # fp32 net truncates them (~1e-8), which silently capped every "exact
        # prefix" at 1e-9 in the conv kink solver. (Same fix as MLP.clone.)
        new = new.to(device=p.device, dtype=p.dtype)
        new.load_state_dict(self.state_dict())
        return new.to(next(self.parameters()).device)


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
