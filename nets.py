"""MLP definitions and initialization matching the paper's assumptions.

Paper assumptions (Sec 3.1): weights initialized mean 0, std sqrt(2/(nin+nout))
(Glorot-style), networks trained with first-order optimization. Hidden
activations: LeakyReLU (paper uses LeakyReLU to avoid dying ReLUs).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer: nn.Linear) -> None:
    std = math.sqrt(2.0 / (layer.in_features + layer.out_features))
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """Feed-forward net, LeakyReLU (default), sigmoid or tanh hidden
    activations, linear output."""

    def __init__(self, dims, negative_slope=0.01, act="leaky_relu"):
        super().__init__()
        self.dims = list(dims)
        self.act_name = act
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        if act == "leaky_relu":
            self.act = nn.LeakyReLU(negative_slope=negative_slope)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"unknown activation {act!r}")
        for layer in self.layers:
            init_layer(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)

    def clone(self):
        new = MLP(self.dims, getattr(self.act, "negative_slope", 0.01),
                  self.act_name)
        p = next(self.parameters())
        # match dtype BEFORE load_state_dict, else an fp64 source is truncated
        # to the fresh net's default (fp32) -- which silently caps every
        # param_errors/scale_normalize comparison at fp32 (~6e-8).
        new = new.to(device=p.device, dtype=p.dtype)
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
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "tanh":
            self.act = nn.Tanh()
        elif act == "leaky_relu":
            self.act = nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"unknown activation {act!r}")
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

    def forward(self, x):
        if x.dim() == 2:                            # flat -> image
            x = x.view(x.shape[0], *self.input_shape)
        for i in range(self.n_conv):
            x = self.act(self.layers[i](x))
            if self.pools[i] > 0:
                x = F.avg_pool2d(x, self.pools[i])
        x = torch.flatten(x, 1)
        for j in range(self.n_conv, len(self.layers) - 1):   # hidden FC
            x = self.act(self.layers[j](x))
        return self.layers[-1](x)

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
