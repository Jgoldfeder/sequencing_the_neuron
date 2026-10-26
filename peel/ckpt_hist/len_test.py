"""Does the black-box query LENGTH matter for the slow RoPE pairs?  Fine-tune the frozen teacher-free guess on
oracle data of length T (same token budget for every T), then score the exact per-frequency QK invariant
outer(q_p, conj(k_p)) per pair and the long-context fit.  T=32 is also run 4x longer (precision vs information)."""
import sys
import torch
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel")
from microt import build  # noqa: E402
from gpt_consensus import canon_  # noqa: E402

dev = "cuda"
gk = torch.load("guess_take4.pt", map_location=dev, weights_only=False)
ck = torch.load("10K_wide4_take4_it018.pt", map_location=dev, weights_only=False)
bb = build(ck["cfg"], ck["teacher_sd"], dev)                 # used as oracle for training; teacher weights only for scoring
TOK = 640_000


def pairs(blk):
    Wq, Wk = blk.attn.q_proj.weight.double(), blk.attn.k_proj.weight.double(); hh = Wq.shape[0] // 2
    return torch.stack([torch.outer(torch.complex(Wq[p], Wq[p + hh]), torch.complex(Wk[p], Wk[p + hh]).conj()) for p in range(hh)])


@torch.no_grad()
def score(m):
    t = canon_(bb.clone(), True); s = canon_(m.clone(), True); out = []
    for tb, sb in zip(t.blocks, s.blocks):
        Pt, Ps = pairs(tb), pairs(sb)
        out.append([((Ps[p] - Pt[p]).abs().pow(2).sum() / Pt[p].abs().pow(2).sum()).sqrt().item() for p in range(8)])
    g = torch.Generator(device=dev).manual_seed(0); lc = []
    for T in (32, 256, 1024):
        X = torch.randn(8192 // T, T, 16, generator=g, device=dev) * 0.5
        lc.append(((m(inputs_embeds=X) - bb(inputs_embeds=X)).abs().mean().item(),
                   (m(inputs_embeds=X).argmax(-1) == bb(inputs_embeds=X).argmax(-1)).float().mean().item()))
    return out, lc


def finetune(T, epochs, lr=3e-4, tok_per_step=16384):
    n = TOK // T; bs = max(1, tok_per_step // T)
    g = torch.Generator(device=dev).manual_seed(1)
    X = torch.randn(n, T, 16, generator=g, device=dev) * 0.5
    with torch.no_grad():
        Y = torch.cat([bb(inputs_embeds=X[i:i + bs]) for i in range(0, n, bs)])
    m = build(gk["cfg"], gk["sd"], dev).train(); opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        for i in torch.randperm(n, device=dev).split(bs):
            opt.zero_grad(); (m(inputs_embeds=X[i]) - Y[i]).abs().mean().backward(); opt.step()
        sch.step()
    return m.eval()


def show(tag, m):
    (b1, b2), lc = score(m)
    f = lambda v: " ".join(f"{x:6.1%}" for x in v)  # noqa: E731
    print(f"{tag:18s} B1 pairs {f(b1)}\n{'':18s} B2 pairs {f(b2)}\n{'':18s} L1/argmax at T=32 {lc[0][0]:.4f}/{lc[0][1]:.3f}  "
          f"T=256 {lc[1][0]:.4f}/{lc[1][1]:.3f}  T=1024 {lc[2][0]:.4f}/{lc[2][1]:.3f}", flush=True)


show("guess (no tune)", build(gk["cfg"], gk["sd"], dev).eval())
for T, ep in [(32, 100), (32, 400), (64, 100), (128, 100), (256, 100)]:
    show(f"tune T={T} ep={ep}", finetune(T, ep))
