"""Per-parameter recovery table of a FROZEN teacher-shaped guess (build_guess.py) vs the teacher.
Exact symmetries removed only: RMSNorm gains folded into the next matrix (canon_), RoPE per-pair scale/angle
(same continuous rule on both: |q_p| = |k_p|, sum(q_p) real positive), W_v/W_o by SVD canonical form, neuron
permutation (Hungarian on pre-activation |corr|, one-to-one), out_norm gain folded into E (gaussian queries).
NO residual transform.  usage: python full_table.py <guess.pt> <ckpt with teacher_sd> [T for behaviour probe]"""
import sys
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel")
from microt import build, rope_cos_sin  # noqa: E402
from microt_align import ov_canonicalize_  # noqa: E402
from gpt_consensus import canon_  # noqa: E402
torch.set_grad_enabled(False); dev = "cuda"


def canon_qk_(m):
    for b in m.blocks:
        Wq, Wk = b.attn.q_proj.weight, b.attn.k_proj.weight; hh = Wq.shape[0] // 2
        for p in range(hh):
            q = torch.complex(Wq[p].double(), Wq[p + hh].double()); k = torch.complex(Wk[p].double(), Wk[p + hh].double())
            z = (k.abs().pow(2).sum() / q.abs().pow(2).sum()).pow(0.25) / (q.sum() / q.sum().abs())
            q = q * z; k = k / z.conj()
            Wq.data[p], Wq.data[p + hh] = q.real.float(), q.imag.float(); Wk.data[p], Wk.data[p + hh] = k.real.float(), k.imag.float()


def zcorr(A, B):
    A = A - A.mean(0); B = B - B.mean(0); return F.normalize(A, dim=0).t() @ F.normalize(B, dim=0)


gk = torch.load(sys.argv[1], map_location=dev, weights_only=False); ck = torch.load(sys.argv[2], map_location=dev, weights_only=False)
Tb = int(sys.argv[3]) if len(sys.argv) > 3 else 256
cfg = ck["cfg"]
t = canon_(build(cfg, ck["teacher_sd"], dev), True); s = canon_(build(gk["cfg"], gk["sd"], dev), True)
# MLP neuron matching on pre-activations (teacher stream; guess uses its own stream)
X = torch.randn(64, 64, cfg["d_model"], generator=torch.Generator(device=dev).manual_seed(7), device=dev) * 0.5
cos, sin = rope_cos_sin(64, t.hdim, cfg["theta"], dev)
def preacts(m):
    x = X; Z = []
    for b in m.blocks:
        x = x + b.attn(b.norm1(x), cos, sin); Z.append(b.mlp.fc1(b.norm2(x)).reshape(-1, b.mlp.fc1.weight.shape[0])); x = x + b.mlp(b.norm2(x))
    return Z
Zt, Zs = preacts(t), preacts(s)
P = [(tb.attn.q_proj.weight.double().t() @ tb.attn.k_proj.weight.double(), sb.attn.q_proj.weight.double().t() @ sb.attn.k_proj.weight.double()) for tb, sb in zip(t.blocks, s.blocks)]
for m in (t, s):
    canon_qk_(m); ov_canonicalize_(m)
fast, slow = [0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15]
rows = []
def add(name, shape, n, A, B, note=""):
    d = (B - A).abs(); rows.append((name, shape, n, f"{d.mean():.1e}", f"{d.max():.1e}", f"{((B - A).norm() / A.norm()).item():.2%}", note))
for i, (tb, sb) in enumerate(zip(t.blocks, s.blocks)):
    rows.append((f"**Block {i+1}**", "", "", "", "", "", ""))
    rows.append(("norm 1 gain", "16", 16, "—", "—", "—", "folded into W_q, W_k, W_v"))
    for nm, lab in (("q_proj", "W_q"), ("k_proj", "W_k")):
        A, B = getattr(tb.attn, nm).weight.double(), getattr(sb.attn, nm).weight.double()
        add(f"{lab}, fast rows (pairs 0-3)", "8 x 16", 128, A[fast], B[fast]); add(f"{lab}, slow rows (pairs 4-7)", "8 x 16", 128, A[slow], B[slow])
    a, b = P[i]; add("*W_q^T W_k (product, ignores RoPE)*", "*16 x 16*", "*(derived)*", a, b)
    add("W_v", "16 x 16", 256, tb.attn.v_proj.weight.double(), sb.attn.v_proj.weight.double())
    add("W_o", "16 x 16", 256, tb.attn.o_proj.weight.double(), sb.attn.o_proj.weight.double())
    rows.append(("norm 2 gain", "16", 16, "—", "—", "—", "folded into fc1"))
    C = zcorr(Zt[i], Zs[i]); ri, ci = linear_sum_assignment(-C.abs().cpu().numpy())
    ci = torch.as_tensor(ci, device=dev); ri = torch.as_tensor(ri, device=dev); sg = torch.sign(C[ri, ci]).double()
    W1t, W1s = tb.mlp.fc1.weight.double()[ri], sb.mlp.fc1.weight.double()[ci] * sg[:, None]
    W2t, W2s = tb.mlp.fc2.weight.double()[:, ri], sb.mlp.fc2.weight.double()[:, ci]
    live = W1t.norm(dim=1) > 0.01 * W1t.norm(dim=1).median()
    rel = (W1s - W1t).norm(dim=1) / W1t.norm(dim=1).clamp_min(1e-9)
    cnt = f"{int((rel[live] < 0.001).sum())}/{int((rel[live] < 0.01).sum())}/{int((rel[live] < 0.05).sum())} of {int(live.sum())} live within 0.1/1/5%, flips {int((sg[live] < 0).sum())}"
    add("fc1 (MLP in)", "56 x 16", 896, W1t, W1s, cnt); add("fc2 (MLP out)", "16 x 56", 896, W2t, W2s)
rows.append(("**Output**", "", "", "", "", "", ""))
add("E * diag(g) (output layer)", "256 x 16", "(4,096 + 16)", t.embed.weight.double(), s.embed.weight.double(), "split from g not observable with gaussian queries")
print(f"guess {sys.argv[1].split('/')[-1]} (from {gk['source'].split('/')[-1]}, member m{gk['member']})\n")
print("| Matrix | Shape | Params | Mean abs error | Max abs error | Relative error | Note |\n|---|---|---|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(str(x) for x in r) + " |")
g2 = torch.Generator(device=dev).manual_seed(0); beh = []
for T in (32, 64, 256, 1024):
    Xl = torch.randn(max(1, 8192 // T), T, cfg["d_model"], generator=g2, device=dev) * 0.5
    tb_, sb_ = build(cfg, ck["teacher_sd"], dev), build(gk["cfg"], gk["sd"], dev)
    a, c = tb_(inputs_embeds=Xl), sb_(inputs_embeds=Xl); beh.append(f"T={T}: L1 {(c - a).abs().mean():.3f}, argmax {(c.argmax(-1) == a.argmax(-1)).float().mean():.1%}")
print("\nbehaviour vs teacher: " + " | ".join(beh))
