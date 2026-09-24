"""Re-score recon_gpt__gaussian__p8__s0.pt with the tied-unembed gauge modded out.

Hypothesis: under --query gaussian (inputs_embeds), embed is NEVER used on the
input side; it only appears in logits = out_norm(x) @ embed.T, i.e. as the
product W_e @ diag(g).  The per-column split between embed and out_norm gain is
therefore a genuine unobservable gauge (d dof), and the raw embed/out_norm eps
in the log should collapse once we fold g into embed on BOTH models.
"""
import os
import sys

import torch

PEEL = os.path.expanduser("~/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel")
sys.path.insert(0, PEEL)
from microt import build  # noqa: E402
from microt_align import align_to_, circuit_error, circuit_report, format_report  # noqa: E402

torch.set_grad_enabled(False)


def fold_outnorm_(m):
    g = m.out_norm.weight.data.clone()
    m.embed.weight.data.mul_(g[None, :])
    m.out_norm.weight.data.fill_(1.0)


ck = torch.load(os.path.join(PEEL, "recon_gpt__gaussian__p8__s0.pt"), map_location="cpu")
cfg, bi = ck["cfg"], ck["best_idx"]
teacher = build(cfg, ck["teacher_sd"])
students = [build(cfg, sd) for sd in ck["student_sds"]]
print(f"iter={ck['iter']} query={ck['query']} best_idx={bi} fit(best)={ck['fit'][bi]:.2e}")

for tag, canon in [("raw       ", False), ("gauge-fixed", True)]:
    errs, best_rep = [], None
    for i, s in enumerate(students):
        c, t = s.clone(), teacher.clone()
        if canon:
            fold_outnorm_(c)
            fold_outnorm_(t)
        align_to_(c, t)
        errs.append(circuit_error(c, t))
        if i == bi:
            best_rep = circuit_report(c, t)
    print(f"[{tag}] circuit-eps best {min(errs):.2e} mean {sum(errs)/len(errs):.2e}")
    print("             " + format_report(best_rep))

# practical cost of the wrong split: run REAL TOKENS through best student vs teacher
gen = torch.Generator().manual_seed(0)
X = torch.randint(0, cfg["vocab"], (256, 32), generator=gen)
yt = teacher(idx=X)
ys = students[bi](idx=X)
print(f"[real tokens] logit err: max {(ys-yt).abs().max():.3f} mean {(ys-yt).abs().mean():.3f}"
      f"  | argmax agree {(ys.argmax(-1) == yt.argmax(-1)).float().mean():.1%}")
# and through the gauge-canon'd pair (should be unchanged for teacher; student's
# input-side embed is genuinely different -> this is NOT expected to fix it)
