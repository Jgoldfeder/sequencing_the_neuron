"""Consensus/progress for a wide-student checkpoint (students ffn != teacher ffn).
 (a) per-module FUNCTIONAL error vs teacher: feed the teacher's own residual stream into each student
     module, compare its output with the teacher module's (width-agnostic; gaussian pins the basis)
 (b) teacher with ONE student module swapped in (whole module object, any width) -> logit L1
 (c) member-vs-member weight consensus (all groups, same widths) via gpt_consensus.consensus
 (d) teacher MLP neurons found inside the students: max cosine of each teacher fc1 row (norm gain
     folded) against the student's fc1 rows; >0.99 counts as present
"""
import copy
import sys
import torch
import torch.nn.functional as F

PEEL = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel"
sys.path.insert(0, PEEL)
from microt import build, rope_cos_sin  # noqa: E402
from gpt_consensus import consensus, canon_  # noqa: E402

torch.set_grad_enabled(False)
dev = "cuda" if torch.cuda.is_available() else "cpu"
path = sys.argv[1] if len(sys.argv) > 1 else f"{PEEL}/recon_gpt__gaussian__p8__s0__10K_wide2.pt"
ck = torch.load(path, map_location=dev, weights_only=False)
cfg, scfg = ck["cfg"], ck.get("student_cfg", ck["cfg"])
t = build(cfg, ck["teacher_sd"], dev); ss = [build(scfg, sd, dev) for sd in ck["student_sds"]]
P = len(ss); order = sorted(range(P), key=lambda i: ck["fit"][i])
print(f"{path.split('/')[-1]}: iter {ck['iter']} | student ffn {scfg['ffn']} teacher {cfg['ffn']} | fits {[round(f, 3) for f in ck['fit']]}")

g = torch.Generator(device=dev).manual_seed(3)
X = torch.randn(512, 32, cfg["d_model"], generator=g, device=dev) * 0.5
Yt = t(inputs_embeds=X)


def rel(a, b):
    return ((a - b).pow(2).mean() / b.pow(2).mean()).sqrt().item()


def module_io(m, X):
    """teacher-stream inputs/outputs of every module of m"""
    x = X; cos, sin = rope_cos_sin(x.shape[1], m.hdim, m.cfg["theta"], x.device); io = {}
    for i, b in enumerate(m.blocks):
        a = b.attn(b.norm1(x), cos, sin); io[f"B{i+1}.attn"] = (x, a); x = x + a
        o = b.mlp(b.norm2(x)); io[f"B{i+1}.mlp"] = (x, o); x = x + o
    io["out"] = (x, m.out_norm(x) @ m.embed.weight.t())
    return io


tio = module_io(t, X); cos, sin = rope_cos_sin(32, t.hdim, cfg["theta"], dev)


def run_module(s, name, x):
    if name == "out":
        return s.out_norm(x) @ s.embed.weight.t()
    i = int(name[1]) - 1; b = s.blocks[i]
    return b.attn(b.norm1(x), cos, sin) if name.endswith("attn") else b.mlp(b.norm2(x))


def swap_into_teacher(s, name):
    m = copy.deepcopy(t)
    if name == "out":
        m.embed = copy.deepcopy(s.embed); m.out_norm = copy.deepcopy(s.out_norm)
    else:
        i = int(name[1]) - 1
        if name.endswith("attn"):
            m.blocks[i].norm1 = copy.deepcopy(s.blocks[i].norm1); m.blocks[i].attn = copy.deepcopy(s.blocks[i].attn)
        else:
            m.blocks[i].norm2 = copy.deepcopy(s.blocks[i].norm2); m.blocks[i].mlp = copy.deepcopy(s.blocks[i].mlp)
    return (m(inputs_embeds=X) - Yt).abs().mean().item()


names = list(tio)
print("\n(a) module output rel-error on the teacher's residual stream   (b) teacher + this student module -> logit L1")
print("    member  " + "  ".join(f"{n:>17s}" for n in names))
for i in order:
    row = []
    for n in names:
        xin, yout = tio[n]
        row.append(f"{rel(run_module(ss[i], n, xin), yout):6.3f} / {swap_into_teacher(ss[i], n):6.3f}")
    print(f"    m{i} {ck['fit'][i]:.3f} " + "  ".join(f"{r:>17s}" for r in row))

full, _ = consensus(ss, ss[order[0]], ck["fit"], fold_out=True)
print("\n(c) member-vs-member consensus (eps 0.02, quorum 5/8): "
      + "  ".join(f"{k} {v['cons']}/{v['n']}" for k, v in full.items()))
for q in (0.375,):
    f3, _ = consensus(ss, ss[order[0]], ck["fit"], fold_out=True, quorum_ratio=q)
    print(f"    at quorum {round(q * P)}/{P}: " + "  ".join(f"{k} {v['cons']}/{v['n']}" for k, v in f3.items()))

print("\n(d) teacher MLP neurons present in student (fc1 row cosine > 0.99 / > 0.95), best 3 members")
tc = canon_(t.clone(), True)
for i in order[:3]:
    sc = canon_(ss[i].clone(), True); parts = []
    for bi, (tb, sb) in enumerate(zip(tc.blocks, sc.blocks)):
        C = F.normalize(tb.mlp.fc1.weight, dim=1) @ F.normalize(sb.mlp.fc1.weight, dim=1).t()
        best = C.abs().amax(1)
        parts.append(f"B{bi+1} {int((best > 0.99).sum())}/{len(best)} ({int((best > 0.95).sum())}) median cos {best.median():.3f}")
    print(f"    m{i}: " + "   ".join(parts))
