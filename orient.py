"""Orientation certificate for pinned leaky-ReLU rows (2026-10-16).

A kink test only certifies the HYPERPLANE of a frontier unit; leaky(-z) = leaky(z) - 1.0101 z, so a sign-flipped row is a
genuine kink surface whose linear remainder is absorbed by always-on units when the data live in the nonnegative orthant
(C2 arm: 3/158 pinned rows flipped, G: 1, J48k: 1).  Global test: with hyperplanes w_j (candidate sign) and jump
coefficients u_j (from the Jacobian jump across each kink, identical for both orientations), the black-box Jacobian
w.r.t. the frontier input at ANY point is

    J(h) = sum_j s_j u_j w_j^T leaky'(s_j q_j(h)),   q_j = w_j.h + b_j,  s_j = +1 (as pinned) / -1 (flipped)

and the two hypotheses for unit j differ by the CONSTANT rank-1 matrix 1.0101 u_j w_j^T wherever h is.  So
J_measured - J_predicted(all s=+1) = -1.0101 sum_{flipped} u_j w_j^T (+ contamination from unpinned units), and a least
squares fit on the rank-1 basis {u_j w_j^T} returns alpha_j ~ -1 for flipped rows and ~0 otherwise.  Offline on the C2
snapshot: flipped rows -1.02..-1.18, correct rows in [-0.10, 0.25] with 39 unpinned units present.

Queries: Jacobians w.r.t. h are taken by central differences along d_h directions mapped through the (known, exact)
prefix: x + eps * Dx e_i with Dx = pinv(d prefix / dx at x0); frontier 0 => Dx = I.  ~4 d_h queries per row for the
jumps + 2 d_h per test point.
"""
import torch


def _prefix(net, K, x):
    h = x
    for l in range(K):
        z = net.layers[l](h)
        h = torch.where(z >= 0, z, 0.01 * z)
    return h


def _prefix_jac(net, K, x0):
    if K == 0:
        return torch.eye(x0.shape[0], device=x0.device, dtype=x0.dtype)
    x = x0.clone().requires_grad_(True)
    h = _prefix(net, K, x[None])[0]
    rows = [torch.autograd.grad(h[i], x, retain_graph=True)[0] for i in range(h.shape[0])]
    return torch.stack(rows)                                      # (d_h, d_x)


class _Counter:
    def __init__(self):
        self.n = 0


def orientation_check(teacher, net, K, rows, device, gen=None, n_test=2, x0_std=0.5, eps_jac=1e-6, eps_side=1e-5,
                      thresh=-0.5, verbose=True):
    """rows: (n, d_h+1) unit [w|b] rows of frontier K in net's gauge (pinned + candidates).  net: fp64 record whose layers
    < K are exact.  Returns (alpha (n,), flipped_mask (n,), n_queries)."""
    t64 = teacher if next(teacher.parameters()).dtype == torch.float64 else teacher.clone().double()
    for p in t64.parameters():
        p.requires_grad_(False)
    rec = net if next(net.parameters()).dtype == torch.float64 else net.clone().double()
    gen = gen or torch.Generator(device=device).manual_seed(0)
    d_x = rec.layers[0].weight.shape[1]
    W = rows[:, :-1].double(); b = rows[:, -1].double(); n = W.shape[0]; d_h = W.shape[1]
    cnt = _Counter()

    def bb(X):
        cnt.n += X.shape[0]
        with torch.no_grad():
            return t64(X)

    def jac_h(x, Dx):
        """black-box Jacobian w.r.t. h at x (d_out, d_h): central differences along the d_h prefix-mapped directions"""
        E = eps_jac * Dx.T                                        # (d_h, d_x): row i = eps * Dx e_i
        Yp = bb(x[None] + E); Ym = bb(x[None] - E)
        return ((Yp - Ym) / (2 * eps_jac)).T                      # (d_out, d_h)

    def base_point():
        for _ in range(8):
            x0 = torch.randn(d_x, device=device, dtype=torch.double, generator=gen) * x0_std
            with torch.no_grad():
                h0 = _prefix(rec, K, x0[None])[0]
            Jp = _prefix_jac(rec, K, x0)
            if torch.linalg.matrix_rank(Jp) == d_h:
                return x0, h0, torch.linalg.pinv(Jp)
        raise RuntimeError("prefix Jacobian rank-deficient at every base point")

    # (1) jump coefficients u_j (same for both orientations)
    U = torch.zeros(n, t64.layers[-1].weight.shape[0], device=device, dtype=torch.double)
    ok = torch.zeros(n, dtype=torch.bool, device=device)
    for j in range(n):
        w = W[j]; bj = b[j]
        for _ in range(6):
            x0, h0, Dx = base_point()
            hs = h0 - (h0 @ w + bj) / (w @ w) * w                  # foot of the hyperplane
            xs = x0 + Dx @ (hs - h0)                               # first-order pre-image
            with torch.no_grad():
                hh = _prefix(rec, K, xs[None])[0]
            if float((hh - hs).norm()) > 1e-6 * max(1.0, float(hs.norm())):
                continue                                           # left the activation region: retry
            dA = Dx @ w
            JA = jac_h(xs + eps_side * dA, Dx); JB = jac_h(xs - eps_side * dA, Dx)
            M = JA - JB
            U[j] = (M @ w) / (0.99 * (w @ w)); ok[j] = True
            break
    # (2) measured vs all-positive predicted Jacobian at fresh points; least squares on the rank-1 basis
    B = torch.stack([torch.outer(U[j], W[j]).flatten() for j in range(n)], 1)  # (d_out*d_h, n)
    alphas = []
    for _ in range(n_test):
        x0, h0, Dx = base_point()
        J = jac_h(x0, Dx)
        a = torch.where(W @ h0 + b > 0, 1.0, 0.01).double()
        Jpred = (U.T * a[None, :]) @ W                              # sum_j a_j u_j w_j^T
        D = (J - Jpred).flatten()
        alphas.append(torch.linalg.lstsq(B[:, ok], D[:, None]).solution.flatten())
    alpha = torch.full((n,), float("nan"), device=device, dtype=torch.double)
    alpha[ok] = torch.stack(alphas).mean(0)
    flipped = ok & (alpha < thresh)
    if verbose:
        srt = alpha[ok].sort().values
        print(f"  [orient] frontier L{K + 1}: {n} rows, {int(ok.sum())} with jump coefficients; alpha min/median/max "
              f"{float(srt[0]):.3f}/{float(srt[len(srt) // 2]):.3f}/{float(srt[-1]):.3f}; flipped (alpha < {thresh}): "
              f"{int(flipped.sum())} | {cnt.n:,} queries", flush=True)
    return alpha, flipped, cnt.n
