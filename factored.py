"""Optimizer-only reparameterizations that add optimization coordinates while
keeping the represented function class IDENTICAL (Judah's relaxation program).
Members stay plain MLPs: each step we materialise the effective weight back into
net.layers[l].weight, so the pipeline's consensus/alignment/scoring are untouched
and the factors are auto-collapsed. W1 (or any subset) can be included/excluded.

Two modes:

  AdditiveResidual (mode="residual", PREFERRED): W_eff = W + U V^T. W is a full
    unrestricted matrix trained normally at the FIT lr (baseline -G path kept),
    plus low-rank express lanes U(out x r), V(in x r) at the FACTOR lr. Init
    V=0, U orthonormal -> W_eff = W_baseline exactly at t=0, but dL/dV = G^T U
    != 0 so the lane wakes immediately. rank r is a pure relaxation knob (any r
    leaves the function class = all matrices). Robust: even if lanes are useless,
    W optimises like baseline. Gradient flow: dW_eff = -(G + U U^T G + G V V^T),
    so <G,-dW_eff> = ||G||^2 + ||U^T G||^2 + ||G V||^2 >= ||G||^2 (augments, never
    replaces, the baseline geometry).

  FactoredTail (mode="square"): W_l = A_l B_l, inner dim k=round(inner_mult*out),
    A=sQ, B=(1/s)Q^T W (Q orthonormal rows) so A B = W_baseline exactly at t=0;
    per-step exact rebalance + moment rescale keeps the factors from blowing up
    under Adam. Removes the -G path (less robust; kept for comparison).
"""
import torch


def resolve_layers(spec, L):
    """spec -> list of weight-layer indices to reparameterize (L = #layers)."""
    if spec == "tail":
        return list(range(1, L))          # every hidden layer after the first
    if spec == "l1":
        return [0]                        # first layer only
    if spec == "all":
        return list(range(0, L))          # every weight matrix
    raise ValueError(f"factor_layers={spec!r} (want tail|l1|all)")


def _adam(P, g, m, v, lr, t, b1, b2, eps):
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    P.addcdiv_(m / (1 - b1 ** t), (v / (1 - b2 ** t)).sqrt().add_(eps), value=-lr)


class AdditiveResidual:
    """W_eff = W + U V^T on the given layers. W at fit_lr, (U,V) at factor_lr."""

    def __init__(self, net, layers, fit_lr, factor_lr, rank,
                 betas=(0.9, 0.999), eps=1e-8):
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.fit_lr = fit_lr
        self.fac_lr = factor_lr
        self.state = []
        for l in layers:
            W = net.layers[l].weight.data
            out, inp = W.shape
            r = min(rank, out, inp) if rank > 0 else min(out, inp)
            U = torch.linalg.qr(torch.randn(out, r, device=W.device,
                                            dtype=W.dtype))[0]   # out x r, orthonormal cols
            V = torch.zeros(inp, r, device=W.device, dtype=W.dtype)
            Wb = W.clone()
            z = torch.zeros_like
            self.state.append(dict(l=l, Wb=Wb, U=U, V=V,
                                   mW=z(Wb), vW=z(Wb), mU=z(U), vU=z(U),
                                   mV=z(V), vV=z(V)))
            net.layers[l].weight.data.copy_(Wb + U @ V.t())      # == Wb (V=0)

    @torch.no_grad()
    def step(self, net):
        self.t += 1
        for s in self.state:
            lyr = net.layers[s["l"]]
            G = lyr.weight.grad
            if G is None:
                continue
            _adam(s["Wb"], G, s["mW"], s["vW"], self.fit_lr, self.t,
                  self.b1, self.b2, self.eps)                    # baseline -G path
            _adam(s["U"], G @ s["V"], s["mU"], s["vU"], self.fac_lr, self.t,
                  self.b1, self.b2, self.eps)                    # express lanes
            _adam(s["V"], G.t() @ s["U"], s["mV"], s["vV"], self.fac_lr, self.t,
                  self.b1, self.b2, self.eps)
            lyr.weight.data.copy_(s["Wb"] + s["U"] @ s["V"].t())

    def decay(self, f=0.1):
        self.fit_lr *= f
        self.fac_lr *= f


class FactoredTail:
    """W_l = A_l B_l (square/over-square), exact baseline embedding + rebalance."""

    def __init__(self, net, layers, factor_lr, inner_mult=1.0,
                 betas=(0.9, 0.999), eps=1e-8):
        self.lr = factor_lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.state = []
        for l in layers:
            W = net.layers[l].weight.data
            out, inp = W.shape
            k = max(out, int(round(inner_mult * out)))
            Qc, _ = torch.linalg.qr(torch.randn(k, out, device=W.device, dtype=W.dtype))
            Q = Qc.t().contiguous()                              # out x k, Q Q^T = I
            s = (W.norm().clamp_min(1e-12) / (out ** 0.5)) ** 0.5
            A = s * Q
            B = (1.0 / s) * (Q.t() @ W)                          # A B = W exactly
            self.state.append(dict(l=l, A=A, B=B,
                                   mA=torch.zeros_like(A), vA=torch.zeros_like(A),
                                   mB=torch.zeros_like(B), vB=torch.zeros_like(B)))
            net.layers[l].weight.data.copy_(A @ B)

    @torch.no_grad()
    def step(self, net):
        self.t += 1
        for s in self.state:
            W = net.layers[s["l"]].weight
            G = W.grad
            if G is None:
                continue
            A, B = s["A"], s["B"]
            _adam(A, G @ B.t(), s["mA"], s["vA"], self.lr, self.t, self.b1, self.b2, self.eps)
            _adam(B, A.t() @ G, s["mB"], s["vB"], self.lr, self.t, self.b1, self.b2, self.eps)
            c = (B.norm().clamp_min(1e-12) / A.norm().clamp_min(1e-12)).sqrt()
            A.mul_(c); B.div_(c)
            s["mA"].div_(c); s["vA"].div_(c * c)
            s["mB"].mul_(c); s["vB"].mul_(c * c)
            W.data.copy_(A @ B)

    def decay(self, f=0.1):
        self.lr *= f
