"""Alignment for the MicroT decoder transformer (config-driven; dims read from the
model/block, so it works for any MicroT size). Gauges:
  * RESIDUAL signed-permutation of the d_model stream (commutes with RMSNorm).
  * HEAD permutation (per block): reorder the d_head head-blocks in q/k/v rows,o cols.
  * FFN neuron permutation (per block): fc1 rows + fc2 cols.
  * continuous WITHIN-HEAD OV/QK gauges: compared via circuits (W_o W_v, W_q^T W_k),
    canonicalized for consensus (OV by SVD, QK by per-RoPE-pair complex scaling).
Discrete gauges are undone exactly by alignment; continuous ones via circuits/canon.
"""
import copy
import os
import sys

import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from microt import MicroT, default_cfg                         # noqa: E402


def rand_model(seed=0, cfg=None):
    torch.manual_seed(seed)
    return MicroT(cfg or default_cfg()).eval()


# ------------------------------------------------------------- gauges --------
@torch.no_grad()
def residual_gauge_(m, P, s=None):
    dev, dt, d = m.embed.weight.device, m.embed.weight.dtype, m.d_model
    P = torch.as_tensor(P, device=dev)
    s = torch.ones(d, dtype=dt, device=dev) if s is None else torch.as_tensor(s, dtype=dt, device=dev)
    sc, sr = s[None, :], s[:, None]
    m.embed.weight.copy_(m.embed.weight[:, P] * sc)
    for blk in m.blocks:
        blk.norm1.weight.copy_(blk.norm1.weight[P]); blk.norm2.weight.copy_(blk.norm2.weight[P])
        for nm in ("q_proj", "k_proj", "v_proj"):
            w = getattr(blk.attn, nm).weight; w.copy_(w[:, P] * sc)
        blk.mlp.fc1.weight.copy_(blk.mlp.fc1.weight[:, P] * sc)
        blk.attn.o_proj.weight.copy_(blk.attn.o_proj.weight[P, :] * sr)
        blk.mlp.fc2.weight.copy_(blk.mlp.fc2.weight[P, :] * sr)
    m.out_norm.weight.copy_(m.out_norm.weight[P])


@torch.no_grad()
def head_perm_(blk, sigma):
    hd = blk.attn.hdim; dev = blk.attn.q_proj.weight.device
    idx = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in sigma]).to(dev)
    for nm in ("q_proj", "k_proj", "v_proj"):
        w = getattr(blk.attn, nm).weight; w.copy_(w[idx, :])
    blk.attn.o_proj.weight.copy_(blk.attn.o_proj.weight[:, idx])


@torch.no_grad()
def ffn_perm_(blk, pi):
    pi = torch.as_tensor(pi, device=blk.mlp.fc1.weight.device)
    blk.mlp.fc1.weight.copy_(blk.mlp.fc1.weight[pi, :])
    blk.mlp.fc2.weight.copy_(blk.mlp.fc2.weight[:, pi])


@torch.no_grad()
def ov_gauge_(blk, h, N, Ninv):
    hd = blk.attn.hdim; r = slice(h * hd, (h + 1) * hd)
    blk.attn.v_proj.weight.data[r, :] = N @ blk.attn.v_proj.weight.data[r, :]
    blk.attn.o_proj.weight.data[:, r] = blk.attn.o_proj.weight.data[:, r] @ Ninv


def rope_commuting_M(hdim):
    h = hdim // 2; M = torch.eye(hdim, dtype=torch.get_default_dtype())
    for i in range(h):
        a, b = torch.randn(2).tolist()
        M[i, i], M[i, i + h], M[i + h, i], M[i + h, i + h] = a, -b, b, a
    return M


@torch.no_grad()
def qk_gauge_(blk, h, M):
    hd = blk.attn.hdim; r = slice(h * hd, (h + 1) * hd); Minv_T = torch.linalg.inv(M).t()
    blk.attn.q_proj.weight.data[r, :] = M @ blk.attn.q_proj.weight.data[r, :]
    blk.attn.k_proj.weight.data[r, :] = Minv_T @ blk.attn.k_proj.weight.data[r, :]


# ---------------------------------------------------------- circuits ---------
@torch.no_grad()
def qk_circuit(blk, h):
    hd = blk.attn.hdim; r = slice(h * hd, (h + 1) * hd)
    return blk.attn.q_proj.weight[r].t() @ blk.attn.k_proj.weight[r]


@torch.no_grad()
def ov_circuit(blk, h):
    hd = blk.attn.hdim; r = slice(h * hd, (h + 1) * hd)
    return blk.attn.o_proj.weight[:, r] @ blk.attn.v_proj.weight[r]


@torch.no_grad()
def qk_canonicalize_(m):
    for blk in m.blocks:
        hd = blk.attn.hdim; hh = hd // 2
        Wq, Wk = blk.attn.q_proj.weight, blk.attn.k_proj.weight
        for h in range(blk.attn.n_head):
            b = h * hd
            for p in range(hh):
                i, j = b + p, b + p + hh
                vq = torch.complex(Wq[i], Wq[j])
                c = vq[vq.abs().argmax()]
                z = (c.abs() / c) / vq.abs().pow(2).sum().sqrt().clamp_min(1e-12)
                vq = z * vq; Wq.data[i], Wq.data[j] = vq.real, vq.imag
                vk = torch.complex(Wk[i], Wk[j]) / z.conj()
                Wk.data[i], Wk.data[j] = vk.real, vk.imag


@torch.no_grad()
def ov_canonicalize_(m):
    for blk in m.blocks:
        hd = blk.attn.hdim
        for h in range(blk.attn.n_head):
            r = slice(h * hd, (h + 1) * hd)
            C = blk.attn.o_proj.weight[:, r] @ blk.attn.v_proj.weight[r]
            U, S, Vt = torch.linalg.svd(C)
            Ur, Sr, Vr = U[:, :hd].clone(), S[:hd], Vt[:hd].clone()
            for k in range(hd):
                if Ur[Ur[:, k].abs().argmax(), k] < 0:
                    Ur[:, k] *= -1; Vr[k, :] *= -1
            sq = Sr.clamp_min(0).sqrt()
            blk.attn.o_proj.weight.data[:, r] = Ur * sq[None, :]
            blk.attn.v_proj.weight.data[r, :] = sq[:, None] * Vr


@torch.no_grad()
def rmsnorm_diag_canon_(m):
    """Remove the RMSNorm-gain <-> next-projection-column DIAGONAL gauge, a genuine
    function symmetry (d params per internal norm): norm(x)=(x/rms)*g feeds W as
    W.diag(g), so g[i] and column i of W trade freely. This gauge is NOT fixed by
    align_to_ and it RESCALES the qk/ov/mlp circuits, so leaving it inflates their
    eps even for a perfect gauge copy. Canonicalize by folding each internal norm's
    gain into the columns it feeds (gain -> 1). out_norm is left alone: its gain is
    pinned by the tied embedding (scaling it would change the input embedding)."""
    for blk in m.blocks:
        g1 = blk.norm1.weight.data.clone()
        for nm in ("q_proj", "k_proj", "v_proj"):
            getattr(blk.attn, nm).weight.data.mul_(g1[None, :])
        blk.norm1.weight.data.fill_(1.0)
        g2 = blk.norm2.weight.data.clone()
        blk.mlp.fc1.weight.data.mul_(g2[None, :])
        blk.norm2.weight.data.fill_(1.0)


# --------------------------------------------------------- alignment ---------
@torch.no_grad()
def align_to_(m, ref):
    dev = m.embed.weight.device
    D = torch.cdist(ref.embed.weight.t().abs(), m.embed.weight.t().abs(), p=1).cpu().numpy()
    invP = torch.as_tensor(linear_sum_assignment(D)[1], device=dev)
    dots = (ref.embed.weight * m.embed.weight[:, invP]).sum(0)
    signs = torch.where(dots >= 0, torch.ones_like(dots), -torch.ones_like(dots))
    residual_gauge_(m, invP, signs)
    for blk, rb in zip(m.blocks, ref.blocks):
        nh, hd = blk.attn.n_head, blk.attn.hdim
        def hfeat(b):
            rows = torch.cat([b.attn.q_proj.weight, b.attn.k_proj.weight, b.attn.v_proj.weight], 1)
            return rows.view(nh, hd, -1).reshape(nh, -1)
        Dh = torch.cdist(hfeat(rb), hfeat(blk), p=1).cpu().numpy()
        head_perm_(blk, torch.as_tensor(linear_sum_assignment(Dh)[1]).tolist())
        Df = torch.cdist(rb.mlp.fc1.weight, blk.mlp.fc1.weight, p=1).cpu().numpy()
        ffn_perm_(blk, torch.as_tensor(linear_sum_assignment(Df)[1]))


@torch.no_grad()
def raw_error(m, ref):
    return max((a - b).abs().max().item()
               for a, b in zip(m.state_dict().values(), ref.state_dict().values()))


@torch.no_grad()
def circuit_error(m, ref):
    m = copy.deepcopy(m); ref = copy.deepcopy(ref)      # fix the RMSNorm-diagonal gauge
    rmsnorm_diag_canon_(m); rmsnorm_diag_canon_(ref)    # (rescales qk/ov/mlp) before scoring
    e = (m.embed.weight - ref.embed.weight).abs().max().item()
    for blk, rb in zip(m.blocks, ref.blocks):
        for h in range(blk.attn.n_head):
            e = max(e, (qk_circuit(blk, h) - qk_circuit(rb, h)).abs().max().item())
            e = max(e, (ov_circuit(blk, h) - ov_circuit(rb, h)).abs().max().item())
        e = max(e, (blk.mlp.fc1.weight - rb.mlp.fc1.weight).abs().max().item())
    return e


@torch.no_grad()
def _mm(diff):
    """(max, mean) of an abs-error tensor, run.py style."""
    return diff.max().item(), diff.mean().item()


@torch.no_grad()
def circuit_report(m, ref):
    """Per-layer (max, mean) abs error, run.py style. Attention layers are compared
    via gauge-invariant circuits (qk=W_q^T W_k, ov=W_o W_v, aggregated over heads);
    the MLP (fc1+fc2) and embed/out_norm are compared raw (discrete gauge removed by
    align). Each entry is (max_eps, mean_eps)."""
    m = copy.deepcopy(m); ref = copy.deepcopy(ref)      # fix the RMSNorm-diagonal gauge
    rmsnorm_diag_canon_(m); rmsnorm_diag_canon_(ref)    # (rescales qk/ov/mlp) before scoring
    rep = {"embed": _mm((m.embed.weight - ref.embed.weight).abs()),
           "out_norm": _mm((m.out_norm.weight - ref.out_norm.weight).abs()), "blocks": []}
    for blk, rb in zip(m.blocks, ref.blocks):
        qk = torch.cat([(qk_circuit(blk, h) - qk_circuit(rb, h)).abs().flatten()
                        for h in range(blk.attn.n_head)])
        ov = torch.cat([(ov_circuit(blk, h) - ov_circuit(rb, h)).abs().flatten()
                        for h in range(blk.attn.n_head)])
        mlp = torch.cat([(blk.mlp.fc1.weight - rb.mlp.fc1.weight).abs().flatten(),
                         (blk.mlp.fc2.weight - rb.mlp.fc2.weight).abs().flatten()])
        rep["blocks"].append({"qk": _mm(qk), "ov": _mm(ov), "mlp": _mm(mlp)})
    return rep


def format_report(rep):
    """run.py-style one-liner: 'eps/layer (all): L[max X mean Y] ...'."""
    def f(lbl, e):
        return f"{lbl}[max {e[0]:.2e} mean {e[1]:.2e}]"
    parts = [f("embed", rep["embed"])]
    for i, b in enumerate(rep["blocks"]):
        parts += [f(f"B{i+1}.qk", b["qk"]), f(f"B{i+1}.ov", b["ov"]), f(f"B{i+1}.mlp", b["mlp"])]
    parts.append(f("out_norm", rep["out_norm"]))
    return "eps/layer (all): " + "  ".join(parts)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    ref = rand_model(0)
    d, nh, hd = ref.d_model, ref.n_head, ref.hdim
    ffn = ref.blocks[0].mlp.fc1.weight.shape[0]
    x = torch.randint(0, ref.vocab, (2, 24)); y0 = ref(x).clone()

    m = rand_model(0)
    residual_gauge_(m, torch.randperm(d), (torch.randint(0, 2, (d,)) * 2 - 1).double())
    for blk in m.blocks:
        head_perm_(blk, torch.randperm(nh).tolist()); ffn_perm_(blk, torch.randperm(ffn))
    print(f"[discrete+sign] function preserved: {(m(x) - y0).abs().max().item():.2e}")
    align_to_(m, ref)
    print(f"[discrete+sign] aligned raw max_eps = {raw_error(m, ref):.2e}  (want ~0)")

    mqk = rand_model(0)
    for blk in mqk.blocks:
        for h in range(nh):
            qk_gauge_(blk, h, rope_commuting_M(hd))
    print(f"[QK gauge]  function preserved: {(mqk(x) - y0).abs().max().item():.2e}")
    e = max((qk_circuit(b, h) - qk_circuit(rb, h)).abs().max().item()
            for b, rb in zip(mqk.blocks, ref.blocks) for h in range(nh))
    print(f"[QK gauge]  QK-circuit err = {e:.2e}  (~0: invariant)")

    mov = rand_model(0)
    for blk in mov.blocks:
        for h in range(nh):
            A = torch.randn(hd, hd); ov_gauge_(blk, h, A, torch.linalg.inv(A))
    ref_c = rand_model(0); ov_canonicalize_(ref_c); ov_canonicalize_(mov)
    print(f"[OV canon]  raw max_eps after canon = {raw_error(mov, ref_c):.2e}  (want ~0)")

    mqk2 = rand_model(0)
    for blk in mqk2.blocks:
        for h in range(nh):
            qk_gauge_(blk, h, rope_commuting_M(hd))
    ref_q = rand_model(0); qk_canonicalize_(ref_q); qk_canonicalize_(mqk2)
    print(f"[QK canon]  raw max_eps after canon = {raw_error(mqk2, ref_q):.2e}  (want ~0)")
