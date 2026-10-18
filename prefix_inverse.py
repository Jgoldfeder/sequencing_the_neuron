"""Invert the (exactly known) prefix of a leaky-ReLU ConvNet: given target frontier inputs Q (N x D_phi), find
images X with phi(X) = Q. The prefix is piecewise affine with a full-rank Jacobian (784 -> 400, condition ~5e3),
so Levenberg-Marquardt with per-sample adaptive damping (never accepting a worsening step) converges once the
activation pattern settles. This is what lets the SAME attack that works on the tail directly be run through the
solved layers: query the black box at X and you get tail(Q)."""
import torch, torch.nn.functional as F


def prefix_fn(net, K):
    def phi(x):
        x = x.view(x.shape[0], *net.input_shape)
        for i in range(K):
            x = net.act(net.layers[i](x))
            if net.pools[i] > 0:
                x = F.avg_pool2d(x, net.pools[i])
        return x.flatten(1)
    return phi


def invert_prefix(net, K, Q, x0=None, iters=60, tol=1e-10, chunk=64, lam0=1e-2, verbose=False):
    """Returns X (N x D_in, same dtype as Q) and per-sample relative residuals ||phi(X)-Q|| / ||Q||."""
    phi = prefix_fn(net, K)
    d_in = int(torch.tensor(net.input_shape).prod())
    N = Q.shape[0]
    X = torch.zeros(N, d_in, device=Q.device, dtype=Q.dtype) if x0 is None else x0.clone()
    res = torch.zeros(N, device=Q.device, dtype=Q.dtype)
    from torch.func import vmap, jacrev
    jac = vmap(jacrev(lambda x: phi(x[None])[0]))
    for i0 in range(0, N, chunk):
        x = X[i0:i0 + chunk].clone(); q = Q[i0:i0 + chunk]
        n = x.shape[0]
        lam = torch.full((n,), lam0, device=x.device, dtype=x.dtype)
        qn = q.norm(dim=1).clamp_min(1e-30)
        with torch.no_grad():
            r = phi(x) - q; rnorm = r.norm(dim=1)
        it = -1
        for it in range(iters):
            if float((rnorm / qn).max()) < tol:
                break
            J = jac(x)                                             # (n, D_phi, d_in)
            JJt = J @ J.transpose(1, 2)
            I = torch.eye(JJt.shape[-1], device=J.device, dtype=J.dtype)
            improved = torch.zeros(n, dtype=torch.bool, device=x.device)
            with torch.no_grad():
                for _ in range(10):                                # per-sample damping search
                    y = torch.linalg.solve(JJt + lam.view(-1, 1, 1) * I, r.unsqueeze(-1))
                    dx = -(J.transpose(1, 2) @ y).squeeze(-1)
                    xn = x + dx
                    rn = phi(xn) - q; rnn = rn.norm(dim=1)
                    ok = (rnn < rnorm) & ~improved
                    x[ok] = xn[ok]; r[ok] = rn[ok]; rnorm[ok] = rnn[ok]
                    lam[ok] = (lam[ok] / 3).clamp_min(1e-12)
                    improved |= ok
                    if bool(improved.all()):
                        break
                    lam[~improved] = lam[~improved] * 10
                if not bool(improved.any()) and float(lam.min()) > 1e8:
                    break                                          # stuck everywhere
        with torch.no_grad():
            X[i0:i0 + chunk] = x
            res[i0:i0 + chunk] = rnorm / qn
        if verbose:
            print(f"  chunk {i0 // chunk}: iters {it + 1}, rel residual median {float(res[i0:i0 + chunk].median()):.2e} max {float(res[i0:i0 + chunk].max()):.2e}", flush=True)
    return X, res


if __name__ == "__main__":
    import time
    from data import make_teacher_cnn
    dev = "cuda"
    t = make_teacher_cnn((1, 28, 28), [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)], (84,), 10, epochs=25, seed=0, device=dev, act="leaky_relu").double()
    for p in t.parameters():
        p.requires_grad_(False)
    torch.manual_seed(0)
    phi = prefix_fn(t, 2)
    with torch.no_grad():
        Qr = phi(torch.randn(64, 784, device=dev, dtype=torch.double) * 0.5)
    t0 = time.time(); X, res = invert_prefix(t, 2, Qr, x0=torch.randn(64, 784, device=dev, dtype=torch.double) * 0.5)
    print(f"(a) reachable targets phi(x'): rel residual median {float(res.median()):.1e} max {float(res.max()):.1e} | {time.time()-t0:.0f}s")
    Q = torch.randn(128, 400, device=dev, dtype=torch.double) * 0.5          # the direct attack's query distribution
    t0 = time.time(); X, res = invert_prefix(t, 2, Q, x0=torch.randn(128, 784, device=dev, dtype=torch.double) * 0.5)
    print(f"(b) Gaussian targets N(0,0.5^2)^400: rel residual median {float(res.median()):.1e}, 90% {float(res.quantile(0.9)):.1e}, max {float(res.max()):.1e}; "
          f"|X| rms {float(X.pow(2).mean().sqrt()):.2f}, max |pixel| {float(X.abs().max()):.1f}; frac solved<1e-6: {float((res < 1e-6).float().mean()):.2f} | {time.time()-t0:.0f}s")
    with torch.no_grad():
        tail_direct = t.layers[4](t.act(t.layers[3](t.act(t.layers[2](Q.view(-1, 16, 5, 5)).flatten(1)))))
        via = t(X)
    print(f"    black box at X vs tail(Q): max |diff| {float((via - tail_direct).abs().max()):.2e}, median {float((via - tail_direct).abs().median()):.2e}")
