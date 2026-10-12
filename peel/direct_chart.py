"""Local forward-only prefix charts for MLP disagreement sampling.

The query variable specifies a hidden displacement, projected onto numerically
resolvable directions and restricted to the anchor's prefix activation region.
No teacher parameters are used to construct the chart.
"""
import torch


class PrefixChart:
    @torch.no_grad()
    def __init__(self, frozen, frontier, act, device, generator=None, anchor=None):
        if act != 'leaky_relu':
            raise ValueError('Local direct queries currently require leaky_relu')
        self.weights = [frozen[i][0].detach().to(device).double() for i in range(frontier)]
        self.biases = [frozen[i][1].detach().to(device).double() for i in range(frontier)]
        self.x0 = (torch.randn(self.weights[0].shape[1], device=device,
                              dtype=torch.float64, generator=generator)
                   if anchor is None else anchor.to(device).double())
        a = self.x0
        jac = None
        preacts, pre_jacs = [], []
        for w, b in zip(self.weights, self.biases):
            z = w @ a + b
            jac = w if jac is None else w @ jac
            preacts.append(z); pre_jacs.append(jac)
            slope = torch.where(z >= 0, torch.ones_like(z), torch.full_like(z, .01))
            a = z * slope
            jac = slope[:, None] * jac
        self.h0 = a
        # Truncate ill-conditioned directions instead of manufacturing huge inputs.
        u, s, vh = torch.linalg.svd(jac, full_matrices=False)
        keep = s > s.max() * 1e-5
        self.rank = int(keep.sum())
        self.inverse = (vh[keep].T / s[keep]) @ u[:, keep].T
        self.response = jac @ self.inverse
        self.scale = jac.norm(dim=1).clamp_min(1e-12)
        self.pre = preacts
        self.changes = [j @ self.inverse for j in pre_jacs]
        self.radius = max(1., float(self.x0.norm()))

    def map(self, z):
        delta = z.double() * self.scale
        dx = delta @ self.inverse.T
        fraction = (self.radius / dx.norm(dim=1).clamp_min(1e-30)).clamp(max=1.)
        for base, change in zip(self.pre, self.changes):
            dz = delta @ change.T
            toward = dz * base.sign() < 0
            limit = .8 * base.abs() / dz.abs().clamp_min(1e-300)
            fraction = torch.minimum(fraction, torch.where(toward, limit,
                                      torch.full_like(limit, float('inf'))).amin(1))
        x = self.x0 + fraction[:, None] * dx
        h = self.h0 + fraction[:, None] * (delta @ self.response.T)
        return x, h

    def prefix(self, x):
        for w, b in zip(self.weights, self.biases):
            x = x @ w.T + b
            x = torch.where(x >= 0, x, .01*x)
        return x

    @torch.no_grad()
    def checked(self, z):
        x, predicted = self.map(z)
        actual = self.prefix(x)
        error = (actual-predicted).abs().amax(1)
        tolerance = 1e-9 * (1+actual.abs().amax(1))
        if not bool((torch.isfinite(x).all(1) & torch.isfinite(actual).all(1)
                     & (error <= tolerance)).all()):
            raise RuntimeError('Local direct-query prefix roundtrip failed; refusing inaccurate labels')
        return x, actual


class ChartStudent(torch.nn.Module):
    def __init__(self, tail, chart):
        super().__init__()
        self.tail = tail
        self.chart = chart

    def forward(self, z):
        return self.tail(self.chart.map(z)[1].to(z.dtype))
