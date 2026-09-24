"""v0 population+disagreement extraction on a MicroT decoder transformer (any size).
Members are trained to match the teacher; per-member recovery is measured with
microt_align (discrete alignment + circuit error). Two query modes:

  --query gaussian : queries live in CONTINUOUS EMBEDDING space (inputs_embeds=Z),
      optimized by GRADIENT ascent to maximize committee disagreement. Powerful
      (exact gradients) but GRAY-BOX: injecting raw residual vectors is not a
      capability you have against a real deployed LM, and Z is off the real-token
      manifold, which may under-constrain the true weights.

  --query discrete : queries are REAL TOKEN IDs (teacher(idx=...)) -- an honest
      black-box primitive (text in, logits out) that is automatically on-manifold.
      No gradient through tokens, so disagreement is found by SEARCH: sample a big
      pool (random bytes + optional real text), score it by student disagreement,
      keep the top-q, optionally hill-climb by mutation. Coarser, but legitimate.

Recipe otherwise IDENTICAL to the canonical MLP pipeline (method.py Cfg): L1 fit
loss, mean_pair/L1 disagreement w/ L1 output normalization, p=8, q=1500, batch=512,
epochs=10, lr=1e-3 (/10 at .6,.85), qg_steps=30, qg_lr=0.1 (/10 at .5,.8), std=0.5.
STATUS: scaffold; recovery is the open q.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from microt import (MicroT, default_cfg, load_st, fetch, build, infer_dims, nll)  # noqa: E402
from microt_align import (align_to_, circuit_error, circuit_report, format_report,  # noqa: E402
                          rand_model)


def _normalize(v):                                       # method.py:192
    return v / v.abs().sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _per_query_disagreement(f, mode, dist):
    """f: (p, S, V) L1-normalized committee outputs. Returns (S,) POSITIVE
    per-query disagreement. Shared by the gradient loss and the discrete search."""
    p = f.shape[0]
    if mode == "variance":
        mean = f.mean(0, keepdim=True)
        return ((f - mean) ** 2).sum(-1).mean(0)
    diff = f.unsqueeze(1) - f.unsqueeze(0)               # (p, p, S, V)
    D = (diff ** 2).sum(-1).clamp_min(1e-24).sqrt() if dist == "l2" else diff.abs().sum(-1)
    i, j = torch.triu_indices(p, p, offset=1, device=f.device)
    pairs = D[i, j, :]                                    # (n_pairs, S)
    if mode == "mean_pair":
        return pairs.mean(0)
    if mode == "min_pair":
        return pairs.min(0).values
    if mode == "median_pair":
        return pairs.median(0).values
    raise ValueError(mode)


def disagreement(outs, mode="mean_pair", dist="l1"):
    """(negated) mean committee disagreement to MINIMIZE. outs: (p, ..., V)."""
    f = _normalize(outs).reshape(outs.shape[0], -1, outs.shape[-1])
    return -_per_query_disagreement(f, mode, dist).mean()


def _run(model, X):
    """Dispatch by dtype: float -> inputs_embeds (gaussian), int -> idx (discrete)."""
    return model(inputs_embeds=X) if torch.is_floating_point(X) else model(idx=X)


# ---------------------------------------------------- gaussian (gradient) ----
def disagree_embed_queries(students, q, T, device, gen, steps=30, lr=0.1, std=0.5,
                           chunk=128, mode="mean_pair", dist="l1", sched=(0.5, 0.8)):
    d = students[0].d_model
    Z = (torch.randn(q, T, d, generator=gen, device=device) * std).requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=lr)
    decay = {int(s * steps) for s in sched}
    ch = chunk if (chunk and chunk > 0) else q
    for step in range(steps):
        if step in decay:
            for g in opt.param_groups:
                g["lr"] /= 10
        opt.zero_grad()
        if ch >= q:
            disagreement(torch.stack([s(inputs_embeds=Z) for s in students]), mode, dist).backward()
        else:                                            # exact chunked q-mean grad
            for c0 in range(0, q, ch):
                sl = slice(c0, min(c0 + ch, q))
                outs = torch.stack([s(inputs_embeds=Z[sl]) for s in students])
                (disagreement(outs, mode, dist) * ((sl.stop - sl.start) / q)).backward()
        opt.step()
    return Z.detach()


# ---------------------------------------------------- discrete (search) ------
@torch.no_grad()
def _score_sequences(students, X, mode, dist, chunk):
    """Per-sequence disagreement (mean over positions). X: (N, T) token ids."""
    out = []
    for c0 in range(0, len(X), chunk):
        outs = torch.stack([s(idx=X[c0:c0 + chunk]) for s in students])   # (p, cb, T, V)
        p, cb, T, V = outs.shape
        f = _normalize(outs).reshape(p, cb * T, V)
        out.append(_per_query_disagreement(f, mode, dist).reshape(cb, T).mean(1))
    return torch.cat(out)


@torch.no_grad()
def disagree_discrete_queries(students, q, T, vocab, device, gen, mode="mean_pair",
                              dist="l1", chunk=256, pool_mult=8, text_pool=None,
                              mutate_steps=0, mutate_frac=0.1):
    N = q * pool_mult
    pool = torch.randint(0, vocab, (N, T), generator=gen, device=device)
    if text_pool is not None and len(text_pool):        # mix in real public text
        k = min(len(text_pool), N // 2)
        ti = torch.randperm(len(text_pool), generator=gen, device=device)[:k]
        pool = torch.cat([text_pool[ti].to(device), pool[:N - k]])
    sc = _score_sequences(students, pool, mode, dist, chunk)
    top = sc.topk(min(q, len(pool))).indices
    sel, sel_sc = pool[top], sc[top]
    for _ in range(mutate_steps):                        # gradient-free hill-climb
        m = torch.rand(sel.shape, generator=gen, device=device) < mutate_frac
        rnd = torch.randint(0, vocab, sel.shape, generator=gen, device=device)
        cand = torch.where(m, rnd, sel)
        cs = _score_sequences(students, cand, mode, dist, chunk)
        better = cs > sel_sc
        sel = torch.where(better[:, None], cand, sel)
        sel_sc = torch.where(better, cs, sel_sc)
    return sel


# ---------------------------------------------- soft-token (gradient) --------
def disagree_soft_queries(students, q, T, vocab, device, gen, steps=40, lr=0.1,
                          tau0=1.0, tau1=0.1, mode="mean_pair", dist="l1", chunk=0,
                          sched=(0.5, 0.8)):
    """Gradient-optimized DISCRETE queries -- as sharp as gaussian, but real tokens.
    Optimize per-position vocab logits to maximize student disagreement, with a
    STRAIGHT-THROUGH estimator: forward the HARD argmax token (so the disagreement is
    the true real-token one), backprop through the soft softmax. Each student embeds
    the soft token with its OWN embedding (student-owned -> no oracle). Temperature
    anneals tau0->tau1 so argmax stays faithful. Returns (q, T) hard token ids for
    the teacher query."""
    embs = [s.embed.weight.detach() for s in students]
    L = (torch.randn(q, T, vocab, generator=gen, device=device) * 0.1).requires_grad_(True)
    opt = torch.optim.Adam([L], lr=lr)
    decay = {int(s * steps) for s in sched}
    ch = chunk if (chunk and chunk > 0) else q
    for step in range(steps):
        if step in decay:
            for g in opt.param_groups:
                g["lr"] /= 10
        tau = tau0 * (tau1 / tau0) ** (step / max(1, steps - 1))
        opt.zero_grad()
        for c0 in range(0, q, ch):
            sl = slice(c0, min(c0 + ch, q))
            soft = torch.softmax(L[sl] / tau, dim=-1)          # (cq, T, vocab)
            hard = F.one_hot(soft.argmax(-1), vocab).to(soft.dtype)
            p = hard + soft - soft.detach()                    # straight-through
            outs = torch.stack([s(inputs_embeds=p @ e) for s, e in zip(students, embs)])
            (disagreement(outs, mode, dist) * ((sl.stop - sl.start) / q)).backward()
        opt.step()
    with torch.no_grad():
        return L.argmax(-1)                                     # (q, T) real tokens


def load_text_pool(path, T, device):
    """Read a UTF-8 text file, encode to bytes, slice into non-overlapping length-T
    windows -> (M, T) int64 token ids on `device`."""
    b = open(path, "rb").read()
    n = (len(b) // T) * T
    return torch.tensor(list(b[:n]), dtype=torch.long, device=device).view(-1, T)


def reconstruct_gpt(teacher, cfg, P=8, outer=40, q=1500, T=32, warm=0, epochs=10,
                    batch=512, lr=1e-3, device="cpu", seed=0, log_every=5,
                    window=0, qg_steps=30, qg_lr=0.1, qg_chunk=128,
                    disagree="mean_pair", qg_dist="l1", fit_loss="l1",
                    qg_init_std=0.5, lr_sched=(0.6, 0.85), qg_sched=(0.5, 0.8),
                    query="gaussian", pool_mult=8, mutate_steps=0, text_pool=None,
                    svd_init=False, save_path=None):
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time(); d, vocab = cfg["d_model"], cfg["vocab"]
    teacher = teacher.to(device).eval()
    students = [MicroT(cfg).to(device) for _ in range(P)]
    if svd_init:                                          # honest real-token embedding init:
        from logit_steal import recover, dim_from_spectrum  # recover unembedding subspace V
        S, Vh, _ = recover(teacher, n_prompts=64, T=T, device=device, seed=seed)
        V = Vh[:dim_from_spectrum(S)].t().contiguous().to(students[0].embed.weight.dtype)
        if V.shape == students[0].embed.weight.shape:    # d matched -> init every student's embed
            with torch.no_grad():
                for s in students:
                    s.embed.weight.copy_(V)
            print(f"  [svd-init] initialized {P} student embeddings from logit-SVD "
                  f"subspace (d={V.shape[1]})", flush=True)
        else:
            print(f"  [svd-init] SKIPPED: recovered d {V.shape[1]} != model d {d}", flush=True)
    opts = [torch.optim.Adam(s.parameters(), lr=lr) for s in students]
    decay_at = {int(s * outer) for s in lr_sched}
    lossfn = (lambda a, b: (a - b).abs().mean()) if fit_loss == "l1" else (lambda a, b: ((a - b) ** 2).mean())
    Xb = Yb = None
    for t in range(outer):
        if t in decay_at:
            for o in opts:
                for g in o.param_groups:
                    g["lr"] /= 10
        if query in ("discrete", "soft"):
            if t < warm:
                X = torch.randint(0, vocab, (q, T), generator=gen, device=device)
            elif query == "soft":                            # gradient-optimized real tokens
                X = disagree_soft_queries(students, q, T, vocab, device, gen,
                                          steps=qg_steps, lr=qg_lr, chunk=qg_chunk,
                                          mode=disagree, dist=qg_dist, sched=qg_sched)
            else:
                X = disagree_discrete_queries(students, q, T, vocab, device, gen,
                                              mode=disagree, dist=qg_dist, chunk=max(qg_chunk, 1),
                                              pool_mult=pool_mult, text_pool=text_pool,
                                              mutate_steps=mutate_steps)
        else:
            if t < warm:
                X = torch.randn(q, T, d, generator=gen, device=device) * qg_init_std
            else:
                X = disagree_embed_queries(students, q, T, device, gen, steps=qg_steps,
                                           lr=qg_lr, std=qg_init_std, chunk=qg_chunk,
                                           mode=disagree, dist=qg_dist, sched=qg_sched)
        with torch.no_grad():
            Y = _run(teacher, X)
        # accumulate on CPU (the buffer grows to window*q queries -- keeping the
        # teacher logits [N,T,vocab] on GPU OOMs; move each batch to device instead)
        X, Y = X.cpu(), Y.cpu()
        Xb = X if Xb is None else torch.cat([Xb, X]); Yb = Y if Yb is None else torch.cat([Yb, Y])
        if window > 0:
            keep = window * q; Xb, Yb = Xb[-keep:], Yb[-keep:]
        n = len(Xb)
        for _ in range(epochs):
            perm = torch.randperm(n, generator=gen, device=device).cpu()
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                xb, yb = Xb[idx].to(device), Yb[idx].to(device)
                for s, o in zip(students, opts):
                    o.zero_grad(); lossfn(_run(s, xb), yb).backward(); o.step()
        if (t + 1) % log_every == 0 or t == outer - 1:
            with torch.no_grad():
                fit = []
                for s in students:
                    tot = cnt = 0
                    for i in range(0, n, batch):
                        xb, yb = Xb[i:i + batch].to(device), Yb[i:i + batch].to(device)
                        e = (_run(s, xb) - yb).abs()
                        tot += e.sum().item(); cnt += e.numel()
                    fit.append(tot / cnt)
            aligned = [s.clone() for s in students]
            for c in aligned:
                align_to_(c, teacher)
            errs = [circuit_error(c, teacher) for c in aligned]
            bi = min(range(P), key=lambda i: fit[i])
            print(f"  it {t+1:3d} | q {n:6d} | loss(best) {fit[bi]:.2e} | "
                  f"circuit-eps best {min(errs):.2e} mean {sum(errs)/P:.2e} | "
                  f"{round(time.time()-t0,1)}s", flush=True)
            print("        " + format_report(circuit_report(aligned[bi], teacher)), flush=True)
            if save_path:                                 # checkpoint so the run is re-scorable
                tmp = save_path + ".tmp"                   # (no re-run) with any corrected metric
                torch.save({"cfg": cfg, "iter": t + 1, "seed": seed, "query": query,
                            "teacher_sd": {k: v.detach().cpu() for k, v in teacher.state_dict().items()},
                            "student_sds": [{k: v.detach().cpu() for k, v in s.state_dict().items()}
                                            for s in students],
                            "fit": fit, "best_idx": bi}, tmp)
                os.replace(tmp, save_path)
                if t + 1 <= log_every:
                    print(f"        [ckpt] students+teacher -> {save_path} (every log-iter)", flush=True)
    return students


def capacity_test(teacher, cfg, device, n=6000, T=32, epochs=1000, batch=512,
                  lr=3e-3, seed=0, query="gaussian", fit_loss="l1", log_every=100):
    """Isolate the OPTIMIZATION floor: fixed non-adversarial dataset, ONE student
    (same arch as teacher, so fit=0 is representable), trained hard with cosine
    decay. If fit -> ~0 and circuit-eps drops, the population loop is merely
    under-optimizing; if fit plateaus high, it's a genuine landscape wall."""
    gen = torch.Generator(device=device).manual_seed(seed)
    teacher = teacher.to(device).eval()
    if query == "discrete":
        X = torch.randint(0, cfg["vocab"], (n, T), generator=gen, device=device)
    else:
        X = torch.randn(n, T, cfg["d_model"], generator=gen, device=device) * 0.5
    with torch.no_grad():
        Y = _run(teacher, X)
    lossfn = (lambda a, b: (a - b).abs().mean()) if fit_loss == "l1" else (lambda a, b: ((a - b) ** 2).mean())
    with torch.no_grad():                                  # sanity: teacher IS a zero
        tf = lossfn(_run(teacher, X), Y).item()
    print(f"[cap] {query} n={n} T={T} | sanity teacher-as-student: "
          f"fit={tf:.1e} circuit-eps={circuit_error(teacher.clone(), teacher):.1e}", flush=True)
    student = MicroT(cfg).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(); lossfn(_run(student, X[idx]), Y[idx]).backward(); opt.step()
        sched.step()
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            with torch.no_grad():
                fit = lossfn(_run(student, X), Y).item()
            c = student.clone(); align_to_(c, teacher)
            print(f"[cap] ep {ep+1:4d} | loss({fit_loss}) {fit:.2e} | circuit-eps {circuit_error(c, teacher):.2e} "
                  f"| lr {sched.get_last_lr()[0]:.1e} | {round(time.time()-t0,1)}s", flush=True)
    return student


def load_ckpt_teacher(path, device):
    """Load a {cfg, sd} checkpoint saved by train_micro.py."""
    ck = torch.load(path, map_location=device)
    cfg = ck["cfg"]
    print(f"[ckpt] {path}: {cfg['d_model']}d {cfg['n_head']}h {cfg['n_layer']}L ffn{cfg['ffn']}", flush=True)
    return build(cfg, ck["sd"], device), cfg


def load_real_teacher(size, device, probe=None):
    """Load a real MicroT-<size>, inferring dims + choosing n_head by NLL."""
    sd = load_st(fetch(size)); dims = infer_dims(sd)
    probe = probe or ("Once upon a time, there was a little girl named Lily who "
                      "loved to play in the park with her friends every day.")
    best = None
    for nh in (1, 2, 4, 8):
        if dims["d_model"] % nh:
            continue
        cfg = default_cfg(**dims, n_head=nh)
        L = nll(build(cfg, sd, device), probe, device)
        if best is None or L < best[0]:
            best = (L, cfg)
    print(f"[real] MicroT-{size} {best[1]['d_model']}d {best[1]['n_head']}h "
          f"{best[1]['n_layer']}L ffn{best[1]['ffn']}  NLL {best[0]:.3f}", flush=True)
    return build(best[1], sd, device), best[1]


if __name__ == "__main__":
    import argparse
    torch.set_default_dtype(torch.float32)
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--ckpt", default=None)               # teacher from train_micro.py
    ap.add_argument("--size", default="50K", choices=["10K", "50K", "100K"])
    ap.add_argument("--query", default="gaussian", choices=["gaussian", "discrete", "soft"])
    ap.add_argument("--p", type=int, default=8); ap.add_argument("--outer", type=int, default=40)
    ap.add_argument("--q", type=int, default=1500); ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--warm", type=int, default=0); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=0); ap.add_argument("--qg-lr", type=float, default=0.1)
    ap.add_argument("--qg-steps", type=int, default=30); ap.add_argument("--qg-chunk", type=int, default=128)
    ap.add_argument("--disagree", default="mean_pair"); ap.add_argument("--qg-dist", default="l1")
    ap.add_argument("--fit-loss", default="l1"); ap.add_argument("--device", default=None)
    ap.add_argument("--pool-mult", type=int, default=8)   # discrete: candidates = pool_mult*q
    ap.add_argument("--mutate-steps", type=int, default=0)  # discrete: hill-climb rounds
    ap.add_argument("--text-file", default=None)          # discrete: real-text pool
    ap.add_argument("--svd-init", action="store_true")    # init embeddings from logit-SVD recovery
    ap.add_argument("--capacity-test", action="store_true")  # single-student fit-floor probe
    ap.add_argument("--cap-epochs", type=int, default=1000); ap.add_argument("--cap-lr", type=float, default=3e-3)
    ap.add_argument("--cap-n", type=int, default=6000)
    ap.add_argument("--log-every", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None,
                    help="checkpoint path (students+teacher+cfg) written every log-iter; "
                         "re-scorable with any corrected metric, no re-run. Defaults to an "
                         "auto path unless --no-save.")
    ap.add_argument("--no-save", action="store_true", help="disable checkpointing")
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.ckpt:
        teacher, cfg = load_ckpt_teacher(a.ckpt, dev)
    elif a.real:
        teacher, cfg = load_real_teacher(a.size, dev)
    else:
        cfg = default_cfg(); teacher = rand_model(0); print(f"[smoke] random teacher, {dev}")
    if a.capacity_test:
        capacity_test(teacher, cfg, dev, n=a.cap_n, T=a.T, epochs=a.cap_epochs,
                      batch=a.batch, lr=a.cap_lr, seed=a.seed, query=a.query,
                      fit_loss=a.fit_loss, log_every=max(a.cap_epochs // 20, 1))
        sys.exit(0)
    text_pool = load_text_pool(a.text_file, a.T, dev) if a.text_file else None
    if text_pool is not None:
        print(f"[text] {a.text_file}: {len(text_pool)} windows of length {a.T}", flush=True)
    save_path = None if a.no_save else (a.save or
                f"recon_gpt__{a.query}__p{a.p}__s{a.seed}.pt")
    if save_path:
        print(f"[save] checkpointing to {save_path} every {a.log_every} iters "
              f"(re-scorable, no re-run). Disable with --no-save.", flush=True)
    reconstruct_gpt(teacher, cfg, P=a.p, outer=a.outer, q=a.q, T=a.T, warm=a.warm,
                    epochs=a.epochs, batch=a.batch, lr=a.lr, device=dev, log_every=a.log_every,
                    seed=a.seed, window=a.window, qg_steps=a.qg_steps, qg_lr=a.qg_lr,
                    qg_chunk=a.qg_chunk, disagree=a.disagree, qg_dist=a.qg_dist, fit_loss=a.fit_loss,
                    query=a.query, pool_mult=a.pool_mult, mutate_steps=a.mutate_steps,
                    text_pool=text_pool, svd_init=a.svd_init, save_path=save_path)
