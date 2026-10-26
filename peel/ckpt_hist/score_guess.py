"""Score a FROZEN teacher-shaped guess (from build_guess.py) against the teacher. Runs only after the guess is
saved.  Strict: one-to-one neuron matching (Hungarian, 56<->56), output weights included, residual gauge either
NONE (identity) or the true RMSNorm symmetry only (scale x orthogonal, Procrustes) per residual stage.
Gauges that are exact function symmetries are removed first: RMSNorm gains folded into the projections they
feed, out_norm gain folded into the tied embedding (unobservable under gaussian queries), QK/OV compared as
circuits.  usage: python score_guess.py <guess.pt> <any ckpt with teacher_sd>"""
import sys
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel")
from microt import build, rope_cos_sin  # noqa: E402
from microt_align import qk_circuit, ov_circuit  # noqa: E402
from gpt_consensus import canon_  # noqa: E402

torch.set_grad_enabled(False); dev = "cuda"


def stages(m, X, cos, sin):
    x = X; S = {"in": x}; Z = []
    for i, b in enumerate(m.blocks):
        x = x + b.attn(b.norm1(x), cos, sin); S[f"B{i+1}.attn"] = x
        Z.append(b.mlp.fc1(b.norm2(x)).reshape(-1, b.mlp.fc1.weight.shape[0]))
        x = x + b.mlp(b.norm2(x)); S[f"B{i+1}.mlp"] = x
    return {k: v.reshape(-1, v.shape[-1]).double() for k, v in S.items()}, Z


def procrustes(Ht, Hs):
    U, S, Vh = torch.linalg.svd(Hs.t() @ Ht); return (S.sum() / Ht.pow(2).sum()).item() * (U @ Vh)


def zcorr(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return F.normalize(A, dim=0).t() @ F.normalize(B, dim=0)


def mm(d):
    return f"{d.abs().max():.1e}/{d.abs().mean():.1e}"


gk = torch.load(sys.argv[1], map_location=dev, weights_only=False)
ck = torch.load(sys.argv[2], map_location=dev, weights_only=False); cfg = ck["cfg"]
t = canon_(build(cfg, ck["teacher_sd"], dev), True); s = canon_(build(gk["cfg"], gk["sd"], dev), True)
X = torch.randn(512, 32, cfg["d_model"], generator=torch.Generator(device=dev).manual_seed(777), device=dev) * 0.5
cos, sin = rope_cos_sin(32, t.hdim, cfg["theta"], dev)
Yt, Ys = t(inputs_embeds=X), s(inputs_embeds=X)
print(f"guess from {gk['source'].split('/')[-1]} member m{gk['member']}: black-box L1 on fresh probe {(Ys - Yt).abs().mean():.4f} "
      f"| argmax agree {(Ys.argmax(-1) == Yt.argmax(-1)).float().mean():.4f}")
St, Zt = stages(t, X, cos, sin); Ss, Zs = stages(s, X, cos, sin)
for mode in ("identity", "scale x rotation"):
    A = {k: (torch.eye(16, device=dev, dtype=torch.float64) if mode == "identity" else procrustes(St[k], Ss[k])) for k in St}
    r = {k: (St[k].pow(2).mean(-1).sqrt() / (St[k] @ A[k].t()).pow(2).mean(-1).sqrt()).median().item() for k in St}
    print(f"\n residual gauge: {mode}" + ("" if mode == "identity" else "   (scale per stage: " + " ".join(f"{k} {torch.linalg.svdvals(A[k])[0]:.3f}" for k in St) + ")"))
    prev = {0: ("in", "B1.attn", "B1.mlp"), 1: ("B1.mlp", "B2.attn", "B2.mlp")}
    for b in (0, 1):
        ka, km, ko = prev[b]; tb, sb = t.blocks[b], s.blocks[b]
        qk = (r[ka] ** 2) * A[ka].t() @ qk_circuit(sb, 0).double() @ A[ka]
        ov = r[ka] * torch.linalg.solve(A[km], ov_circuit(sb, 0).double()) @ A[ka]
        W1t, W2t = tb.mlp.fc1.weight.double(), tb.mlp.fc2.weight.double()
        W1s = sb.mlp.fc1.weight.double() @ A[km] * r[km]; W2s = torch.linalg.solve(A[ko], sb.mlp.fc2.weight.double())
        C = zcorr(Zt[b], Zs[b]).abs().cpu().numpy()
        ri, ci = linear_sum_assignment(-C)                         # one-to-one, 56 <-> 56
        ci = torch.as_tensor(ci, device=dev); ri = torch.as_tensor(ri, device=dev)
        live = W1t.norm(dim=1) > 0.01 * W1t.norm(dim=1).median()
        sg = torch.sign(torch.stack([zcorr(Zt[b][:, u:u + 1], Zs[b][:, v:v + 1])[0, 0] for u, v in zip(ri.tolist(), ci.tolist())])).double()
        d1 = W1s[ci] * sg[:, None] - W1t[ri]; d2 = W2s[:, ci] - W2t[:, ri]
        relr = d1.norm(dim=1) / W1t[ri].norm(dim=1).clamp_min(1e-9)
        lv = live[ri]
        cnt = " ".join(f"<{th:g}%:{int((relr[lv] < th / 100).sum())}" for th in (0.1, 1, 5))
        print(f"   B{b+1}: qk {mm(qk - qk_circuit(tb, 0).double())}  ov {mm(ov - ov_circuit(tb, 0).double())}  "
              f"fc1 {mm(d1[lv])}  fc2 {mm(d2[:, lv])}  | live neurons {int(lv.sum())}, flipped {int((sg[lv] < 0).sum())}, "
              f"fc1 rows {cnt} | dead teacher units matched to guess rows of norm {W1s[ci[~lv]].norm(dim=1).max().item() if (~lv).any() else 0:.1e}")
    E = s.embed.weight.double() @ A["B2.mlp"] * r["B2.mlp"]
    print(f"   unembedding E {mm(E - t.embed.weight.double())}")

# ---- RoPE-aware attention check: per frequency pair p, outer(q_p, conj(k_p)) is the exact invariant ----
def pairs(blk):
    Wq, Wk = blk.attn.q_proj.weight.double(), blk.attn.k_proj.weight.double(); hh = Wq.shape[0] // 2
    return torch.stack([torch.outer(torch.complex(Wq[p], Wq[p + hh]), torch.complex(Wk[p], Wk[p + hh]).conj()) for p in range(hh)])


print("\n per-frequency QK relative error (RoPE pair 0 = fastest ... 7 = slowest):")
for b, (tb, sb) in enumerate(zip(t.blocks, s.blocks)):
    Pt, Ps = pairs(tb), pairs(sb)
    print(f"   B{b+1}: " + " ".join(f"{((Ps[p] - Pt[p]).abs().pow(2).sum() / Pt[p].abs().pow(2).sum()).sqrt().item():6.1%}" for p in range(Pt.shape[0])))
print(" guess vs teacher by context length (L1 / argmax agree):")
g2 = torch.Generator(device=dev).manual_seed(0)
for T in (32, 64, 128, 256, 1024):
    Xl = torch.randn(max(1, 8192 // T), T, cfg["d_model"], generator=g2, device=dev) * 0.5
    a, c = t(inputs_embeds=Xl), s(inputs_embeds=Xl)
    print(f"   T={T:4d}: {(c - a).abs().mean():.4f} / {(c.argmax(-1) == a.argmax(-1)).float().mean():.4f}")
