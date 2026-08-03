"""Network alignment under scaling/permutation isomorphisms (paper App. D).

Torch convention: for nn.Linear, weight is (out, in). Hidden neuron c of
layer l owns row c of layers[l].weight and bias c of layers[l].bias, and is
consumed via column c of layers[l+1].weight.

LeakyReLU is positively homogeneous, so the scaling isomorphism applies
(alpha > 0): scale row c of W_l and b_l by 1/alpha, scale column c of
W_{l+1} by alpha. Polarity does NOT apply (LeakyReLU is not odd).
"""
import torch


@torch.no_grad()
def scale_normalize_(net):
    """Canonicalize scales: every hidden neuron's incoming row gets unit L2
    norm; the scale is pushed into the next layer's column. In-place."""
    for l in range(len(net.layers) - 1):
        W = net.layers[l].weight  # (out_l, in_l)
        b = net.layers[l].bias
        norms = W.norm(dim=1).clamp_min(1e-12)  # per hidden neuron
        W.div_(norms.unsqueeze(1))
        b.div_(norms)
        net.layers[l + 1].weight.mul_(norms.unsqueeze(0))


@torch.no_grad()
def _match_features(net, l):
    """Feature vector per hidden neuron of layer l for matching."""
    W = net.layers[l].weight
    b = net.layers[l].bias.unsqueeze(1)
    return torch.cat([W, b], dim=1)


@torch.no_grad()
def greedy_perm(A, B):
    """Greedy L1 matching between rows of A and rows of B (paper App. D).
    Returns perm: list such that B's row perm[i] should be moved to position
    i to align with A's row i."""
    n = A.shape[0]
    D = torch.cdist(A.float(), B.float(), p=1)
    INF = torch.tensor(float("inf"), device=D.device)
    perm = [None] * n
    for _ in range(n):
        idx = D.argmin().item()
        i, j = idx // n, idx % n
        perm[i] = j
        D[i, :] = INF
        D[:, j] = INF
    return perm


@torch.no_grad()
def permute_layer_(net, l, perm):
    """Apply neuron permutation to hidden layer l of net (in-place).
    perm[i] = old index that now sits at position i."""
    idx = torch.tensor(perm, device=net.layers[l].weight.device)
    net.layers[l].weight.copy_(net.layers[l].weight[idx])
    net.layers[l].bias.copy_(net.layers[l].bias[idx])
    W_next = net.layers[l + 1].weight
    W_next.copy_(W_next[:, idx])


@torch.no_grad()
def align_clone_to(recon, teacher):
    """Return (normalized teacher, aligned+normalized recon copy)."""
    t = teacher.clone()
    r = recon.clone()
    scale_normalize_(t)
    scale_normalize_(r)
    for l in range(len(t.layers) - 1):
        perm = greedy_perm(_match_features(t, l), _match_features(r, l))
        permute_layer_(r, l, perm)
    return t, r


@torch.no_grad()
def param_errors(recon, teacher):
    """Max/mean absolute parameter error after alignment + scale
    normalization. Does not mutate recon. Returns dict with per-layer max
    errors, overall max, and mean per matrix."""
    t, r = align_clone_to(recon, teacher)
    max_errs, mean_errs = [], []
    for la, lb in zip(t.layers, r.layers):
        for pa, pb in zip(la.parameters(), lb.parameters()):
            d = (pa - pb).abs()
            max_errs.append(d.max().item())
            mean_errs.append(d.mean().item())
    return {
        "max_eps": max(max_errs),
        "mean_eps_per_matrix": mean_errs,
        "max_eps_per_matrix": max_errs,
    }
