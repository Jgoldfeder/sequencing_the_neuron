"""Restrict queries to the recovered first-layer row space, without teacher access.

For an exact full-row-rank first layer this loses no possible first-layer
activation: its row-space coordinates still map onto every preactivation.
With an approximate recovered layer this is an approximation, not a certificate.
"""
import copy
import torch

class ProjectedOracle(torch.nn.Module):
    def __init__(self, oracle, basis):
        super().__init__()
        self.oracle=oracle
        self.register_buffer('basis',basis)
    def forward(self,x):return self.oracle(x@self.basis.T)
    def clone(self):return copy.deepcopy(self)

@torch.no_grad()
def project_first_rowspace(oracle,guess):
    w=guess.layers[0].weight
    if w.ndim!=2 or w.shape[0]>=w.shape[1]:
        raise ValueError('Requires a wide-input MLP first layer')
    basis=torch.linalg.qr(w.T,mode='reduced').Q
    projected=copy.deepcopy(guess)
    first=torch.nn.Linear(basis.shape[1],w.shape[0],bias=True,device=w.device,dtype=w.dtype)
    first.weight.copy_(w@basis);first.bias.copy_(guess.layers[0].bias)
    projected.layers[0]=first
    projected.dims=list(guess.dims);projected.dims[0]=basis.shape[1]
    projected.requires_grad_(False)
    return ProjectedOracle(oracle,basis),projected,basis
