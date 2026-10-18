"""Shared setup for the 10-layer refinement diagnostics: load the snapshot (guess + teacher), build the
jet residual r(theta) and matrix-free J / J^T products exactly as tanh_solve.refine does (layer-1 weights
frozen), on a given probe set."""
import sys, os, math, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from torch.func import jvp, vjp
from nets import MLP
import tanh_solve as ts
from method import param_errors

SNAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "recon", "_l1fixed_refined__10layer_tanh.pt")


def load(dev):
    ck = torch.load(SNAP, map_location=dev, weights_only=False)
    dims = ck["dims"]
    teacher = MLP(dims, act="tanh").to(dev); teacher.load_state_dict(ck["teacher_state"]); t64 = teacher.double()
    net = MLP(dims, act="tanh").to(dev).double(); net.load_state_dict(ck["state_dict"])
    bb = ts.Oracle(lambda x: t64(x.double()))
    return dims, t64, net, bb


class Problem:
    """residual(theta) = [y - Y ; JU - JU_bb] on probe points X with directions U; J products by torch.func."""

    def __init__(self, bb, net, X, dirs, act="tanh", h=1e-3, seed=0, freeze=("layers.0.weight",)):
        dev = X.device
        self.X = X.double(); N, d = X.shape
        self.dims = list(net.dims); self.n_layers = len(self.dims) - 1; self.act = act
        gen = torch.Generator(device="cpu").manual_seed(seed)
        if dirs and dirs < d:
            U = torch.randn(N, dirs, d, generator=gen, dtype=torch.float64).to(dev)
            self.U = U / U.norm(dim=2, keepdim=True)
        else:
            self.U = None
        q0 = bb.n
        self.Y, self.JU = ts.measure_jets(bb, self.X, self.U, h=h)
        self.queries = bb.n - q0
        self.freeze = set(freeze)
        self.keys = [k for k, _ in net.named_parameters() if k not in self.freeze]
        self.theta = {k: v.detach().clone().double() for k, v in net.named_parameters()}
        self.numel = sum(self.theta[k].numel() for k in self.keys)
        self.net = net

    def residual(self, p):
        y, ju = ts.model_jets(p, self.X, self.U, self.act, self.n_layers)
        return torch.cat([(y - self.Y).reshape(-1), (ju - self.JU).reshape(-1)])

    def loss(self, p):
        with torch.no_grad():
            r = self.residual(p)
        return float(r @ r)

    def flat(self, dct):
        return torch.cat([dct[k].reshape(-1) for k in self.keys])

    def unflat(self, v):
        out, i = {k: torch.zeros_like(self.theta[k]) for k in self.freeze}, 0
        for k in self.keys:
            n = self.theta[k].numel(); out[k] = v[i:i + n].view_as(self.theta[k]); i += n
        return out

    def linearize(self, theta=None):
        """Returns (r, g = J^T r, JtJ(v) operator, Jv(v) operator) at theta."""
        theta = self.theta if theta is None else theta
        r, vjp_fn = vjp(self.residual, theta)

        def Jv(v):
            return jvp(self.residual, (theta,), (self.unflat(v),))[1]

        def JtJ(v):
            return self.flat(vjp_fn(Jv(v))[0])
        g = self.flat(vjp_fn(r)[0])
        return r.detach(), g, JtJ, Jv

    def step(self, theta, delta):
        return {k: theta[k] + dk for k, dk in self.unflat(delta).items()}

    def errors(self, theta, t64):
        with torch.no_grad():
            self.net.load_state_dict({k: theta[k] for k in self.keys}, strict=False)
        e = param_errors(self.net, t64)
        return e["max_eps"], e["max_eps_per_matrix"], e["mean_eps_per_matrix"]


def fmt(v):
    return " ".join(f"{x:.1e}" for x in v)
