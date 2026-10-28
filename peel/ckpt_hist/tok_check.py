"""Diagnostic for REAL-TOKEN (--query discrete) runs (uses the teacher; not a teacher-free score).
Probes are random token sequences, so each model reads its OWN embedding; the student's residual stream is
related to the teacher's by a linear map A_k per stage (fitted by least squares on the streams).
Per member: L1 / argmax vs teacher on fresh random tokens; residual R^2 per stage; MLP neuron census
(teacher neurons found by pre-activation |corr| > 0.99); per-frequency QK error with the stage map A undone;
embedding error E_s vs E_t A^T (rows = tokens).  usage: python tok_check.py ckpt [ckpt ...]"""
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel")
from microt import build, rope_cos_sin  # noqa: E402
from gpt_consensus import canon_  # noqa: E402
torch.set_grad_enabled(False); dev = "cuda"


def streams(m, idx):
    x = m.embed(idx); cos, sin = rope_cos_sin(idx.shape[1], m.hdim, m.cfg["theta"], idx.device)
    S = {"in": x}; Z = []
    for i, b in enumerate(m.blocks):
        x = x + b.attn(b.norm1(x), cos, sin); S[f"B{i+1}.attn"] = x
        Z.append(b.mlp.fc1(b.norm2(x)).reshape(-1, b.mlp.fc1.weight.shape[0])); x = x + b.mlp(b.norm2(x)); S[f"B{i+1}.mlp"] = x
    return {k: v.reshape(-1, v.shape[-1]).double() for k, v in S.items()}, Z


def zcorr(A, B):
    A = A - A.mean(0); B = B - B.mean(0); return F.normalize(A, dim=0).t() @ F.normalize(B, dim=0)


def rms(v):
    return v.pow(2).mean(-1, keepdim=True).sqrt()


def pairs(Wq, Wk):
    hh = Wq.shape[0] // 2
    return torch.stack([torch.outer(torch.complex(Wq[p], Wq[p + hh]), torch.complex(Wk[p], Wk[p + hh]).conj()) for p in range(hh)])


for f in sys.argv[1:]:
    ck = torch.load(f, map_location=dev, weights_only=False); cfg = ck["cfg"]; T = 256
    g = torch.Generator(device=dev).manual_seed(5)
    idx = torch.randint(0, cfg["vocab"], (32, T), generator=g, device=dev)
    t = canon_(build(cfg, ck["teacher_sd"], dev), False); St, Zt = streams(t, idx); Yt = t(idx=idx)
    out = []
    for mi in range(len(ck["student_sds"])):
        s = canon_(build(ck.get("student_cfg", cfg), ck["student_sds"][mi], dev), False); Ss, Zs = streams(s, idx); Ys = s(idx=idx)
        A = {k: torch.linalg.lstsq(St[k], Ss[k]).solution.t() for k in St}
        r2 = {k: 1 - ((St[k] @ A[k].t() - Ss[k]).pow(2).sum() / (Ss[k] - Ss[k].mean(0)).pow(2).sum()).item() for k in St}
        found = [int((zcorr(zt.float(), zs.float()).abs().amax(1) > 0.99).sum()) for zt, zs in zip(Zt, Zs)]
        fr = []
        for b, k in ((0, "in"), (1, "B1.mlp")):
            r = (rms(St[k]) / rms(St[k] @ A[k].t())).median().item()
            Pt = pairs(t.blocks[b].attn.q_proj.weight.double(), t.blocks[b].attn.k_proj.weight.double())
            Ps = pairs(s.blocks[b].attn.q_proj.weight.double() @ A[k] * r, s.blocks[b].attn.k_proj.weight.double() @ A[k] * r)
            fr.append([((Ps[p] - Pt[p]).abs().pow(2).sum() / Pt[p].abs().pow(2).sum()).sqrt().item() for p in range(8)])
        Ee = ((s.embed.weight.double() - t.embed.weight.double() @ A["in"].t()).norm() / (t.embed.weight.double() @ A["in"].t()).norm()).item()
        out.append(((Ys - Yt).abs().mean().item(), mi, (Ys.argmax(-1) == Yt.argmax(-1)).float().mean().item(), r2, found, fr, Ee,
                    torch.linalg.svdvals(A["in"]).tolist()))
    out.sort()
    live = [int((b.mlp.fc1.weight.norm(dim=1) > 0.01 * b.mlp.fc1.weight.norm(dim=1).median()).sum()) for b in t.blocks]
    print(f"\n{f.split('/')[-1]} (iter {ck['iter']}, query {ck.get('query')}): best 3 members by L1 on fresh random tokens (T={T})")
    for L, mi, am, r2, found, fr, Ee, sv in out[:3]:
        print(f"  m{mi}: L1 {L:.3f} argmax {am:.1%} | residual R^2 " + " ".join(f"{k} {v:.3f}" for k, v in r2.items())
              + f" | MLP teacher neurons found B1 {found[0]}/{live[0]} B2 {found[1]}/{live[1]}"
              + f"\n       input map A singular values {min(sv):.2f}-{max(sv):.2f} | embedding rel err (after A) {Ee:.2%}"
              + "\n       QK per pair B1 " + " ".join(f"{x:5.1%}" for x in fr[0]) + " | B2 " + " ".join(f"{x:5.1%}" for x in fr[1]))
