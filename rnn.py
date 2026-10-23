"""RNN extraction: Elman tanh RNN black box (sequential MNIST, one 28-pixel row
per time step), population + disagreement queries + RNN-aware alignment.

Model: per layer h_t = tanh(W_ih x_t + b + W_hh h_{t-1}), h_0 = 0; Linear head
on the last layer's final state. Only b_ih + b_hh is identifiable, so a layer
carries ONE bias (the original code's nn.RNN merged them in canonical_forms).
Inputs are FLAT (B, T*d) so the shared sampler (method.gen_queries), replay
buffer and fit losses work unchanged; the net reshapes internally.

Gauge of a tanh layer: hidden permutation P and signs S (tanh odd):
  W_ih -> S P W_ih,  b -> S P b,  W_hh -> S P W_hh P^T S,  next W_in -> W_in P^T S.
The row [W_ih | b] only sees (P, S) through its OWN unit (its columns are fixed
by the already-aligned previous layer), so matching is a signed Hungarian
assignment on those rows -- polynomial, no 2^(n-1) polarity search and no
norm-sort heuristic (cf. original paper App. D / reviewer item #21). W_hh is
then carried along and compared as a parameter, never used for matching.

Usage: python run.py --variant mergedbest --rnn 32 --q 2000 --outer 40 --device cuda:0
"""
import copy
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import linear_sum_assignment

from data import load_mnist, TEACHER_DIR


class RNN(nn.Module):
    """Stacked Elman tanh RNN + Linear head on the last step. Flat input (B, T*d)."""

    def __init__(self, input_size, hidden, seq_len, out_dim=10):
        super().__init__()
        self.input_size, self.seq_len, self.out_dim = int(input_size), int(seq_len), int(out_dim)
        self.hidden = [int(h) for h in hidden]
        ins = [self.input_size] + self.hidden[:-1]
        self.ih = nn.ModuleList([nn.Linear(i, h) for i, h in zip(ins, self.hidden)])
        self.hh = nn.ModuleList([nn.Linear(h, h, bias=False) for h in self.hidden])
        self.head = nn.Linear(self.hidden[-1], self.out_dim)
        for h, ih, hh in zip(self.hidden, self.ih, self.hh):     # nn.RNN's default init
            k = 1.0 / math.sqrt(h)
            for p_ in (ih.weight, ih.bias, hh.weight):
                nn.init.uniform_(p_, -k, k)

    def forward(self, x):
        B = x.shape[0]
        z = x.reshape(B, self.seq_len, self.input_size)
        for ih, hh in zip(self.ih, self.hh):
            U = ih(z)                                            # (B, T, H): input drive of every step at once
            h = U.new_zeros(B, U.shape[-1])
            hs = []
            for t in range(self.seq_len):
                h = torch.tanh(U[:, t] + hh(h))
                hs.append(h)
            z = torch.stack(hs, 1)
        return self.head(z[:, -1])

    def states(self, x):
        """Per-layer hidden trajectories [(B, T, H_l)] (diagnostics)."""
        out, B = [], x.shape[0]
        z = x.reshape(B, self.seq_len, self.input_size)
        for ih, hh in zip(self.ih, self.hh):
            U = ih(z); h = U.new_zeros(B, U.shape[-1]); hs = []
            for t in range(self.seq_len):
                h = torch.tanh(U[:, t] + hh(h)); hs.append(h)
            z = torch.stack(hs, 1); out.append(z)
        return out

    def matrices(self):
        """Every parameter in a fixed order with a readable name (eps reporting)."""
        out = []
        for l, (ih, hh) in enumerate(zip(self.ih, self.hh)):
            out += [(f"W_ih{l + 1}", ih.weight), (f"b{l + 1}", ih.bias), (f"W_hh{l + 1}", hh.weight)]
        return out + [("W_out", self.head.weight), ("b_out", self.head.bias)]

    def clone(self):
        return copy.deepcopy(self)


# ------------------------------------------------------------------ teacher --
def teacher_path_rnn(hidden, seq_len, out_dim, epochs, seed):
    os.makedirs(TEACHER_DIR, exist_ok=True)
    return os.path.join(TEACHER_DIR, f"teacher_rnn_T{seq_len}__h{'-'.join(map(str, hidden))}"
                                     f"__o{out_dim}_e{epochs}_s{seed}.pt")


def make_teacher_rnn(hidden, seq_len=28, out_dim=10, epochs=10, seed=0, lr=1e-3,
                     batch=128, device="cpu", verbose=True, clip=1.0):
    """Train (or load) the RNN black box on row-sequential MNIST (T=28, d=28).
    seq_len must divide 784: T=28 rows of 28, T=784 pixels of 1, T=49 blocks of 16, ..."""
    if 784 % seq_len:
        raise ValueError(f"--seq-len {seq_len} must divide 784 (MNIST)")
    d = 784 // seq_len
    path = teacher_path_rnn(hidden, seq_len, out_dim, epochs, seed)
    net = RNN(d, hidden, seq_len, out_dim)
    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=device))
        return net.to(device)
    torch.manual_seed(seed)
    net = RNN(d, hidden, seq_len, out_dim).to(device)
    (xtr, ytr), (xte, yte) = load_mnist(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=batch, shuffle=True)
    for ep in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            lossf(net(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)   # tanh RNN: exploding-gradient guard
            opt.step()
        if verbose:
            net.eval()
            with torch.no_grad():
                acc = sum((net(xte[i:i + 4096]).argmax(1) == yte[i:i + 4096]).sum().item()
                          for i in range(0, len(xte), 4096)) / len(xte)
            print(f"  rnn teacher epoch {ep + 1}/{epochs}  test acc {acc:.4f}", flush=True)
    torch.save(net.state_dict(), path)
    return net


# ---------------------------------------------------------------- alignment --
@torch.no_grad()
def _permute_flip_(net, l, perm, s):
    """Apply hidden-unit gauge of layer l in place: new unit i = s[i] * old unit perm[i]."""
    ih, hh = net.ih[l], net.hh[l]
    sv = s.to(ih.weight.dtype)
    ih.weight.data = ih.weight.data[perm] * sv[:, None]
    ih.bias.data = ih.bias.data[perm] * sv
    hh.weight.data = hh.weight.data[perm][:, perm] * sv[:, None] * sv[None, :]
    nxt = net.ih[l + 1] if l + 1 < len(net.ih) else net.head
    nxt.weight.data = nxt.weight.data[:, perm] * sv[None, :]


@torch.no_grad()
def align_rnn_to_(net, ref):
    """Permute + sign-flip net's hidden units (layer by layer) into ref's frame.
    Signed Hungarian on [W_ih | b] rows, L1 cost (min over the two polarities)."""
    for l in range(len(net.ih)):
        A = torch.cat([ref.ih[l].weight, ref.ih[l].bias[:, None]], 1).double()
        B = torch.cat([net.ih[l].weight, net.ih[l].bias[:, None]], 1).double().to(A.device)
        Dp = torch.cdist(A, B, p=1)
        Dn = torch.cdist(A, -B, p=1)
        D = torch.minimum(Dp, Dn)
        _, cols = linear_sum_assignment(D.cpu().numpy())
        perm = torch.as_tensor(cols, device=net.ih[l].weight.device)
        ar = torch.arange(len(cols), device=A.device)
        s = torch.where(Dn[ar, perm.to(A.device)] < Dp[ar, perm.to(A.device)], -1.0, 1.0).to(perm.device)
        _permute_flip_(net, l, perm, s)
    return net


@torch.no_grad()
def rnn_param_errors(recon, teacher):
    """Max/mean absolute parameter error after RNN gauge alignment (recon -> teacher frame).
    Same keys as align.param_errors, plus the matrix names."""
    r = recon.clone().to(dtype=next(teacher.parameters()).dtype)
    align_rnn_to_(r, teacher)
    names, mx, mn = [], [], []
    for (n, pt), (_, pr) in zip(teacher.matrices(), r.matrices()):
        d = (pt - pr).abs()
        names.append(n); mx.append(d.max().item()); mn.append(d.mean().item())
    return {"max_eps": max(mx), "mean_eps_per_matrix": mn, "max_eps_per_matrix": mx,
            "matrix_names": names}


@torch.no_grad()
def rnn_consensus(pop, ref_idx, eps=0.02, quorum_ratio=0.625):
    """Align every member into member[ref_idx]'s frame; per hidden unit, the largest
    inf-norm eps ball over the members' full unit feature [W_ih row | b | W_hh row]
    (W_hh row is comparable once the layer and its inputs are aligned). Units whose
    ball reaches quorum take the ball mean. The head counts per output row.
    Returns (consensus net, [(H, n_consensus) per hidden layer], (out, n_consensus))."""
    P = len(pop)
    quorum = math.ceil(quorum_ratio * P)
    ref = pop[ref_idx]
    aligned = [align_rnn_to_(m.clone(), ref) if i != ref_idx else ref.clone() for i, m in enumerate(pop)]
    net = ref.clone()

    def ball(feats):                                             # (P, H, f) -> ok[H], mean[H, f]
        D = (feats[:, None] - feats[None]).abs().amax(-1)        # (P, P, H)
        inb = D < eps
        a = inb.sum(1).argmax(0)                                 # best anchor per unit
        S = inb[a, :, torch.arange(feats.shape[1], device=feats.device)].T.to(feats.dtype)   # (P, H)
        n = S.sum(0)
        return n >= quorum, torch.einsum("ph,phf->hf", S, feats) / n.clamp_min(1)[:, None]

    per_layer = []
    for l in range(len(net.ih)):
        H, I = net.ih[l].weight.shape
        F = torch.stack([torch.cat([a.ih[l].weight, a.ih[l].bias[:, None], a.hh[l].weight], 1) for a in aligned])
        ok, M = ball(F)
        net.ih[l].weight.data[ok] = M[ok, :I]
        net.ih[l].bias.data[ok] = M[ok, I]
        net.hh[l].weight.data[ok] = M[ok, I + 1:]
        per_layer.append((H, int(ok.sum())))
    F = torch.stack([torch.cat([a.head.weight, a.head.bias[:, None]], 1) for a in aligned])
    ok, M = ball(F)
    net.head.weight.data[ok] = M[ok, :-1]
    net.head.bias.data[ok] = M[ok, -1]
    return net, per_layer, (F.shape[1], int(ok.sum()))


# -------------------------------------------------------------- reconstruct --
@torch.no_grad()
def _batched(net, X, bs=4096):
    return torch.cat([net(X[i:i + bs]) for i in range(0, X.shape[0], bs)])


def reconstruct_rnn(teacher, cfg, device, eval_pts, seed=0, cons_eps=0.02, cons_quorum=0.625,
                    cons_every=1):
    """Paper Algorithms 1+2 on an RNN black box: committee of cfg.p RNNs, cfg.outer
    iterations of cfg.q disagreement queries (method.gen_queries, the shared sampler),
    each followed by cfg.epochs of fit on the replay buffer (cfg.window iters, 0 = all)."""
    from method import gen_queries
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    d, T, hid, out = teacher.input_size, teacher.seq_len, teacher.hidden, teacher.out_dim
    in_dim = d * T
    pop = [RNN(d, hid, T, out).to(device) for _ in range(cfg.p)]
    lr = cfg.lr
    opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in pop]
    decay_at = {int(s * cfg.outer) for s in cfg.lr_sched}
    Xs, Ys = [], []
    Yeval = _batched(teacher, eval_pts)
    log, losses, t0, nq = [], None, time.time(), 0
    best_eps, best_state = float("inf"), None
    for t in range(cfg.outer):
        if t in decay_at:
            lr /= 10
            for o in opts:
                for g in o.param_groups:
                    g["lr"] = lr
        tq = time.time()
        if t < cfg.warmstart_iters:
            Q = torch.randn(cfg.q, in_dim, generator=gen, device=device) * cfg.qg_init_std
        else:
            for m in pop:
                m.eval()
            Q = gen_queries(pop, cfg, in_dim, device, gen, losses=losses)
        Q = Q.to(device)
        Y = _batched(teacher, Q)
        nq += Q.shape[0]
        Xs.append(Q); Ys.append(Y)
        if cfg.window and len(Xs) > cfg.window:
            Xs.pop(0); Ys.pop(0)
        X, Yb = torch.cat(Xs), torch.cat(Ys)
        tq = time.time() - tq
        tf = time.time()
        for m, o in zip(pop, opts):
            m.train()
            for _ in range(cfg.epochs):
                perm = torch.randperm(X.shape[0], generator=gen, device=device)
                for i in range(0, X.shape[0], cfg.batch):
                    idx = perm[i:i + cfg.batch]
                    diff = m(X[idx]) - Yb[idx]
                    loss = diff.abs().mean() if cfg.fit_loss == "l1" else (diff ** 2).mean()
                    o.zero_grad(set_to_none=True)
                    loss.backward()
                    o.step()
        tf = time.time() - tf
        with torch.no_grad():
            losses = [(_batched(m, eval_pts) - Yeval).abs().mean().item() for m in pop]
        bi = min(range(cfg.p), key=lambda i: losses[i])
        pe = rnn_param_errors(pop[bi], teacher)
        if pe["max_eps"] < best_eps:
            best_eps, best_state = pe["max_eps"], copy.deepcopy(pop[bi].state_dict())
        rec = {"it": t + 1, "queries": nq, "best_loss": losses[bi], "losses": losses,
               "best_max_eps": pe["max_eps"], "best_eps_per_matrix": pe["max_eps_per_matrix"],
               "t_query": round(tq, 2), "t_fit": round(tf, 2)}
        msg = ""
        if cfg.p > 1 and ((t + 1) % cons_every == 0 or t == cfg.outer - 1):
            cons, per_layer, head = rnn_consensus(pop, bi, cons_eps, cons_quorum)
            ce = rnn_param_errors(cons, teacher)
            rec.update(consensus=per_layer, consensus_head=head, cons_max_eps=ce["max_eps"])
            msg = (" | cons " + " ".join(f"L{l + 1} {c}/{h}" for l, (h, c) in enumerate(per_layer))
                   + f" out {head[1]}/{head[0]} eps {ce['max_eps']:.2e}")
        log.append(rec)
        eps_s = " ".join(f"{n} {e:.1e}" for n, e in zip(pe["matrix_names"], pe["max_eps_per_matrix"]))
        print(f"[it {t + 1:3d}] q {nq} | best loss {losses[bi]:.2e} | max_eps {pe['max_eps']:.2e} "
              f"({eps_s}){msg} | {tq:.1f}s+{tf:.1f}s", flush=True)
    bi = min(range(cfg.p), key=lambda i: losses[i])
    best = pop[bi]
    final = {"final_max_eps": rnn_param_errors(best, teacher)["max_eps"],
             "final_loss": losses[bi], "best_seen_max_eps": best_eps,
             "total_queries": nq, "wall_s": round(time.time() - t0, 1)}
    if cfg.p > 1:
        cons, per_layer, head = rnn_consensus(pop, bi, cons_eps, cons_quorum)
        final.update(consensus=per_layer, consensus_head=head,
                     cons_max_eps=rnn_param_errors(cons, teacher)["max_eps"])
    with torch.no_grad():
        final["final_agree"] = (_batched(best, eval_pts).argmax(1) == Yeval.argmax(1)).float().mean().item()
    return best, pop, log, final
