"""Committee consensus reports for the MicroT extraction (the transformer port of the MLP/tanh
`cons`, `cons(masked cols)` and `iso` lines in method.py / reverse_peel.py).

Units (the transformer analogue of "neurons"), each compared as a raw weight vector
(GELU/softmax have no scale gauge) after the continuous gauges are removed:
  * res    : residual-stream coordinate j = embed column j (vocab entries)
  * embed  : token t = embed row t (d entries)
  * Bi.qk  : row i of each head's QK circuit W_q^T W_k (key-side coords as features)
  * Bi.ov  : row i of each head's OV circuit W_o W_v (input coords as features)
  * Bi.mlp : neuron u = [fc1 row u | fc2 col u]
  * out_norm (only when not folded): one scalar per coordinate
Gauges: internal RMSNorm gains folded into the projections they feed (rmsnorm_diag_canon_);
under --query gaussian the tied-unembed gauge is folded too (out_norm gain -> embed), since the
input side of the embedding is never exercised and the split is unobservable.

cons          : every member aligned to the lowest-loss member (align_to_, discrete gauges);
                a unit is in consensus if some seed (loss order) has >= quorum members inside its
                inf-norm eps ball; consensus value = mean of that ball; cons-eps = |value - teacher|
                (teacher aligned to the same reference). Teacher used ONLY for cons-eps.
cons(masked)  : same, features restricted to the residual coordinates in consensus (the MLP
                "masked cols" rule: rows compared only on inputs already agreed on).
iso           : teacher-free and alignment-free: residual coordinates paired to the reference by
                activation (|corr| of the residual stream over all layers, signed permutation),
                heads by their residual contribution, MLP neurons by fc1 pre-activation;
                `identified` = corr > 0.99 in >= quorum members. Rows are tested on the identified
                residual coordinates only (skipped, '*', when < 10% are identified).

Standalone on a checkpoint written by gpt_extract.py (re-scores a running run, no restart):
    python gpt_consensus.py recon_gpt__gaussian__p8__s0__10K.pt
"""
import math
import os
import sys

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from microt import apply_rope, build, rope_cos_sin                     # noqa: E402
from microt_align import (align_to_, ffn_perm_, head_perm_, ov_circuit,  # noqa: E402
                          qk_circuit, residual_gauge_, rmsnorm_diag_canon_)


@torch.no_grad()
def canon_(m, fold_out):
    if fold_out:                                         # tied-unembed gauge (gaussian queries)
        m.embed.weight.mul_(m.out_norm.weight[None, :]); m.out_norm.weight.fill_(1.0)
    rmsnorm_diag_canon_(m)
    return m


@torch.no_grad()
def unit_tables(m, fold_out):
    """{group: (n_units, n_feat) tensor} in m's current basis, plus the feature->residual-coord
    index of every group (for masking); -1 = feature not tied to a residual coordinate."""
    d = m.d_model; ar = torch.arange(d, device=m.embed.weight.device)
    U, fc = {"res": m.embed.weight.t(), "embed": m.embed.weight}, {"res": None, "embed": ar}
    for i, b in enumerate(m.blocks):
        H = b.attn.n_head
        U[f"B{i + 1}.qk"] = torch.cat([qk_circuit(b, h) for h in range(H)]); fc[f"B{i + 1}.qk"] = ar
        U[f"B{i + 1}.ov"] = torch.cat([ov_circuit(b, h) for h in range(H)]); fc[f"B{i + 1}.ov"] = ar
        U[f"B{i + 1}.mlp"] = torch.cat([b.mlp.fc1.weight, b.mlp.fc2.weight.t()], 1)
        fc[f"B{i + 1}.mlp"] = torch.cat([ar, ar])
    if not fold_out:
        U["out_norm"] = m.out_norm.weight[:, None]; fc["out_norm"] = None
    return U, fc


def _ball(U, order, eps, quorum):
    """U (P, n, f): consensus mask (n,) and value (n, f); seeds tried in loss order."""
    P, n = U.shape[:2]
    if U.shape[-1] == 0:
        return torch.zeros(n, dtype=torch.bool, device=U.device), U[order[0]].clone()
    D = (U[:, None] - U[None]).abs().amax(-1)            # (P, P, n)
    ok = torch.zeros(n, dtype=torch.bool, device=U.device); val = U[order[0]].clone()
    for r in order:
        A = D[r] < eps                                   # (P, n) members inside seed r's ball
        hit = (A.sum(0) >= quorum) & ~ok
        if hit.any():
            val[hit] = ((U * A[..., None]).sum(0) / A.sum(0)[:, None].clamp_min(1))[hit]
            ok |= hit
    return ok, val


def _masked(U, fc, keep):
    if fc is None:
        return U
    return U[..., keep[fc]]


@torch.no_grad()
def consensus(students, teacher, losses, fold_out, eps=0.02, quorum_ratio=0.625):
    P = len(students); quorum = math.ceil(quorum_ratio * P)
    order = sorted(range(P), key=lambda i: losses[i]); ref = canon_(students[order[0]].clone(), fold_out)
    mems = []
    for s in students:
        c = canon_(s.clone(), fold_out); align_to_(c, ref); mems.append(c)
    t = canon_(teacher.clone(), fold_out); align_to_(t, ref)
    tabs = [unit_tables(c, fold_out) for c in mems]; Ut, fc = unit_tables(t, fold_out)
    full = {}
    for g in Ut:
        U = torch.stack([tb[0][g] for tb in tabs])
        ok, val = _ball(U, order, eps, quorum)
        e = (val - Ut[g]).abs()[ok]
        full[g] = dict(n=U.shape[1], cons=int(ok.sum()),
                       max=e.max().item() if ok.any() else None, mean=e.mean().item() if ok.any() else None)
    # masked: features restricted to the residual coordinates (embed columns) in consensus
    keep = _ball(torch.stack([tb[0]["res"] for tb in tabs]), order, eps, quorum)[0]
    masked = {}
    for g in Ut:
        if g == "res":
            continue
        U = _masked(torch.stack([tb[0][g] for tb in tabs]), fc[g], keep)
        masked[g] = dict(n=U.shape[1], cons=int(_ball(U, order, eps, quorum)[0].sum()))
    masked["res_kept"] = int(keep.sum())
    return full, masked


# ------------------------------------------------------------------ iso ------
@torch.no_grad()
def _internals(m, X):
    """residual stream at every block boundary, per-head residual contributions, fc1 pre-activations
    (exactly MicroT.forward, unrolled)"""
    x = m.embed(X) if not torch.is_floating_point(X) else X
    cos, sin = rope_cos_sin(x.shape[1], m.hdim, m.cfg["theta"], x.device)
    res, heads, pre = [x], [], []
    for b in m.blocks:
        a = b.attn; h = b.norm1(x); B, T, D = h.shape; H, hd = a.n_head, a.hdim
        q, k, v = (getattr(a, nm)(h).view(B, T, H, hd).transpose(1, 2) for nm in ("q_proj", "k_proj", "v_proj"))
        q, k = apply_rope(q, cos, sin, a.cfg["rope"]), apply_rope(k, cos, sin, a.cfg["rope"])
        att = (q @ k.transpose(-1, -2)) / (hd ** 0.5)
        att = (att + torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)).softmax(-1)
        oh = torch.einsum("bhtk,dhk->hbtd", att @ v, a.o_proj.weight.view(D, H, hd))
        heads.append(oh.reshape(H, -1)); x = x + oh.sum(0)
        z = b.mlp.fc1(b.norm2(x)); pre.append(z.reshape(-1, z.shape[-1]))
        x = x + b.mlp.fc2(F.gelu(z, approximate="tanh" if m.cfg["gelu"] == "tanh" else "none"))
        res.append(x)
    return torch.cat([r.reshape(-1, r.shape[-1]) for r in res]), heads, pre


def _corr(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12); B = B / B.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return A.t() @ B                                     # (ref units, member units)


def _pair(C, signed):
    r, c = linear_sum_assignment((-(C.abs() if signed else C)).cpu().numpy())
    perm = torch.as_tensor(c, device=C.device)
    return perm, C[torch.as_tensor(r, device=C.device), perm]


@torch.no_grad()
def iso(students, losses, X, fold_out, eps=0.02, quorum_ratio=0.625, id_corr=0.99):
    P = len(students); quorum = math.ceil(quorum_ratio * P)
    order = sorted(range(P), key=lambda i: losses[i]); ri = order[0]
    mems = [canon_(s.clone(), fold_out) for s in students]
    Rr = _internals(mems[ri], X)[0]
    rc = []
    for m in mems:                                       # residual: signed permutation by activation
        perm, c = _pair(_corr(Rr, _internals(m, X)[0]), signed=True)
        residual_gauge_(m, perm, torch.sign(c).to(m.embed.weight.dtype)); rc.append(c.abs())
    res_id = (torch.stack(rc) > id_corr).sum(0) >= quorum
    _, Hr, Zr = _internals(mems[ri], X)
    nb = len(mems[ri].blocks); hid = [[] for _ in range(nb)]; nid = [[] for _ in range(nb)]
    for m in mems:                                       # heads by residual contribution, neurons by fc1 pre-act
        _, Hm, Zm = _internals(m, X)
        for i, b in enumerate(m.blocks):
            hp, hc = _pair(_corr(Hr[i].t(), Hm[i].t()), signed=False)
            head_perm_(b, hp.tolist()); hid[i].append(hc)
            fp, fcor = _pair(_corr(Zr[i], Zm[i]), signed=False)
            ffn_perm_(b, fp); nid[i].append(fcor)
    rep = {"res": dict(n=len(res_id), paired=int(res_id.sum()))}
    enough = int(res_id.sum()) >= max(1, len(res_id) // 10)
    tabs = [unit_tables(m, fold_out) for m in mems]; fc = tabs[0][1]

    def rows(g, U=None):
        if not enough:
            return -1
        f = fc[g] if U is None else fc[g][:U.shape[-1]]  # mlp halves: each half indexes the d coords
        U = torch.stack([tb[0][g] for tb in tabs]) if U is None else U
        return int(_ball(_masked(U, f, res_id), order, eps, quorum)[0].sum())

    rep["embed"] = dict(n=tabs[0][0]["embed"].shape[0], rows=rows("embed"))
    for i in range(nb):
        b = f"B{i + 1}"; d = mems[ri].d_model
        mlp = torch.stack([tb[0][f"{b}.mlp"] for tb in tabs])
        rep[b] = dict(heads=int(((torch.stack(hid[i]) > id_corr).sum(0) >= quorum).sum()), H=len(hid[i][0]),
                      qk=rows(f"{b}.qk"), ov=rows(f"{b}.ov"), d=d * len(hid[i][0]),
                      mlp_r=rows(f"{b}.mlp", mlp[..., :d]), mlp_c=rows(f"{b}.mlp", mlp[..., d:]),
                      paired=int(((torch.stack(nid[i]) > id_corr).sum(0) >= quorum).sum()), ffn=mlp.shape[1])
    return rep


def probe(cfg, query, n=256, T=32, std=0.5, device="cpu", seed=1234):
    g = torch.Generator(device=device).manual_seed(seed)
    if query == "gaussian":
        return torch.randn(n, T, cfg["d_model"], generator=g, device=device) * std
    return torch.randint(0, cfg["vocab"], (n, T), generator=g, device=device)


# ---------------------------------------------------------------- format -----
def _e(x):
    return "n/a" if x is None else f"{x:.1e}"


def fmt_cons(full):
    toks = [f"{g} {v['cons']}/{v['n']}" for g, v in full.items()]
    agg = [v for v in full.values() if v["max"] is not None]
    mx = max((v["max"] for v in agg), default=None)
    mn = (sum(v["mean"] * v["cons"] for v in agg) / max(1, sum(v["cons"] for v in agg))) if agg else None
    worst = "  ".join(f"{g}[{_e(v['max'])}]" for g, v in full.items())
    return "  ".join(toks) + f" | cons-eps max {_e(mx)} mean {_e(mn)}", worst


def fmt_masked(masked, eps):
    return (f"cons(masked res, eps {eps:g}) [res kept {masked['res_kept']}] "
            + "  ".join(f"{g} {v['cons']}/{v['n']}" for g, v in masked.items() if g != "res_kept"))


def fmt_iso(rep):
    def r(x, n):
        return f"{x}/{n}" if x >= 0 else f"*/{n}"
    out = [f"res {rep['res']['paired']}/{rep['res']['n']}", f"embed {r(rep['embed']['rows'], rep['embed']['n'])}"]
    for k in sorted(k for k in rep if k.startswith("B")):
        v = rep[k]
        out.append(f"{k}[heads {v['heads']}/{v['H']} qk {r(v['qk'], v['d'])} ov {r(v['ov'], v['d'])} "
                   f"mlp r{r(v['mlp_r'], v['ffn'])} c{r(v['mlp_c'], v['ffn'])} paired {v['paired']}/{v['ffn']}]")
    return "  ".join(out)


def report_lines(students, teacher, losses, cfg, query, eps=0.02, quorum_ratio=0.625, X=None):
    fold = query == "gaussian"
    X = probe(cfg, query, device=students[0].embed.weight.device) if X is None else X
    full, masked = consensus(students, teacher, losses, fold, eps, quorum_ratio)
    c, worst = fmt_cons(full)
    return [f"cons {c}", f"cons-eps/unit-group (max) {worst}", fmt_masked(masked, eps),
            "iso  " + fmt_iso(iso(students, losses, X, fold, eps, quorum_ratio))]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--quorum", type=float, default=0.625); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    teacher = build(ck["cfg"], ck["teacher_sd"], a.device)
    students = [build(ck["cfg"], sd, a.device) for sd in ck["student_sds"]]
    print(f"{a.ckpt}: iter {ck['iter']} query {ck['query']} best m{ck['best_idx']} fit {ck['fit'][ck['best_idx']]:.2e}")
    for ln in report_lines(students, teacher, ck["fit"], ck["cfg"], ck["query"], a.eps, a.quorum):
        print("  " + ln)
