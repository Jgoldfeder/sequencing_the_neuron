"""MLP definitions and initialization matching the paper's assumptions.

Paper assumptions (Sec 3.1): weights initialized mean 0, std sqrt(2/(nin+nout))
(Glorot-style), networks trained with first-order optimization. Hidden
activations: LeakyReLU (paper uses LeakyReLU to avoid dying ReLUs).
"""
import math

import torch
import torch.nn as nn


def init_layer(layer: nn.Linear) -> None:
    std = math.sqrt(2.0 / (layer.in_features + layer.out_features))
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """Feed-forward net, LeakyReLU hidden activations, linear output."""

    def __init__(self, dims, negative_slope=0.01):
        super().__init__()
        self.dims = list(dims)
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.act = nn.LeakyReLU(negative_slope=negative_slope)
        for layer in self.layers:
            init_layer(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)

    def clone(self):
        new = MLP(self.dims, self.act.negative_slope)
        new.load_state_dict(self.state_dict())
        return new.to(next(self.parameters()).device)


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
