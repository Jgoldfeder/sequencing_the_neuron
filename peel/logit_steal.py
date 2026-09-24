"""Realistic (black-box, real-token) transformer extraction, front end.

Implements the core of Carlini et al., "Stealing Part of a Production Language
Model" (2024), specialized to MicroT: from REAL-TOKEN queries and their output
logits alone -- no embedding injection, no teacher internals -- recover

  * the hidden dimension d   (the rank of the logit matrix), and
  * the unembedding subspace  col-space(W_U),  W_U = embed . diag(g)  (g=out_norm),
    i.e. embed (times the final RMSNorm gain) up to a d x d transform.

Why it works: logits(x) = out_norm(h(x)) @ embed^T = (h/rms(h)) @ (embed diag g)^T,
and h is only d-dimensional, so stacking logits over many prompts gives a matrix of
rank exactly d whose right-singular subspace is col-space(W_U). MicroT hands us full
logits, so we skip the paper's hard part (reconstructing logits from a restricted
API) and SVD directly.

This is the honest front end for a full real-token attack: once W_U's subspace is in
hand you can (a) read each query's final hidden state h up to the same gauge
(h_normed = logits @ pinv(W_U^T)), giving a rich per-query signal, and (b) fix the
output side so the downstream block-fit stops having to learn the tied embedding
from scratch. Resolving the residual d x d gauge down to the RMSNorm-respecting
(signed-permutation) basis is the remaining step and is left to the fit phase.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from microt import MicroT, default_cfg, load_st, fetch, build, infer_dims  # noqa: E402


@torch.no_grad()
def collect_logits(teacher, n_prompts, T, vocab, device, gen, drop_first=1):
    """Query the teacher on random REAL token sequences; return the stacked
    per-position logit matrix L (N, vocab). Positions < drop_first are dropped
    (very short contexts are near-degenerate). Real tokens + logits only."""
    rows = []
    for _ in range(0, n_prompts, 64):
        b = min(64, n_prompts - len(rows) // max(1, (T - drop_first)))
        X = torch.randint(0, vocab, (max(b, 1), T), generator=gen, device=device)
        lg = teacher(idx=X)                      # (b, T, vocab)
        rows.append(lg[:, drop_first:, :].reshape(-1, vocab))
        if sum(r.shape[0] for r in rows) >= n_prompts * (T - drop_first):
            break
    return torch.cat(rows, 0)


@torch.no_grad()
def recover(teacher, n_prompts=64, T=32, device="cpu", seed=0):
    """Recover (d_estimate, singular_values, W_U_subspace V) from real-token logits."""
    vocab = teacher.vocab
    gen = torch.Generator(device=device).manual_seed(seed)
    L = collect_logits(teacher, n_prompts, T, vocab, device, gen)      # (N, vocab)
    Lc = L - L.mean(0, keepdim=True)                                   # center (drop the const)
    U, S, Vh = torch.linalg.svd(Lc, full_matrices=False)              # Vh: (k, vocab)
    return S, Vh, L.shape[0]


def dim_from_spectrum(S, ratio=1e-3):
    """Hidden dim = number of singular values above `ratio` * S[0] (the rank cliff)."""
    thr = ratio * S[0].item()
    return int((S > thr).sum().item())


if __name__ == "__main__":
    import argparse
    torch.set_default_dtype(torch.float64)                             # clean SVD spectrum
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None); ap.add_argument("--size", default=None)
    ap.add_argument("--n-prompts", type=int, default=64); ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--device", default=None); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location=dev); cfg = ck["cfg"]
        teacher = MicroT(cfg).to(dev).double().eval(); teacher.load_state_dict(ck["sd"])
    else:
        sd = load_st(fetch(a.size or "10K")); cfg = default_cfg(**infer_dims(sd), n_head=1)
        teacher = build(cfg, sd, dev).double()
    d_true = teacher.d_model
    S, Vh, N = recover(teacher, a.n_prompts, a.T, dev, a.seed)
    d_est = dim_from_spectrum(S)
    print(f"[logit-steal] queried {a.n_prompts} real-token prompts (T={a.T}) -> {N} logit rows")
    print(f"  singular-value cliff: {[f'{s:.1e}' for s in S[:min(len(S), d_true+3)].tolist()]}")
    print(f"  recovered hidden dim d = {d_est}   (true {d_true})   "
          f"{'OK' if d_est == d_true else 'MISMATCH'}")
    # validate the recovered subspace matches col-space(embed . diag(g))
    with torch.no_grad():
        g = teacher.out_norm.weight                       # (d,)
        W_U = teacher.embed.weight * g[None, :]           # (vocab, d) = embed diag(g)
        V = Vh[:d_est].t()                                # (vocab, d_est) recovered basis
        # projection of W_U onto span(V): residual should be ~0 if subspaces match
        P = V @ V.t()
        resid = (W_U - P @ W_U).norm() / W_U.norm().clamp_min(1e-30)
        print(f"  unembedding-subspace recovery: relative residual "
              f"||W_U - proj_V W_U|| / ||W_U|| = {resid.item():.2e}  (want ~0)")
        # bonus: recover a query's final hidden state (up to gauge) from logits alone
        gen = torch.Generator(device=dev).manual_seed(123)
        X = torch.randint(0, teacher.vocab, (1, a.T), generator=gen, device=dev)
        lg = teacher(idx=X)[0]                            # (T, vocab)
        h_hat = (lg - lg.mean(-1, keepdim=True)) @ V      # (T, d_est) hidden state up to gauge
        print(f"  bonus: inverted logits -> per-position hidden state, shape {tuple(h_hat.shape)} "
              f"(the top-of-residual signal, up to the d x d gauge)")
