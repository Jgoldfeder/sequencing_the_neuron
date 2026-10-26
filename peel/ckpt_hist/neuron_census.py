"""Width-agnostic comparison of students to the teacher (gaussian mode, residual basis pinned):
 * residual decodability: R^2 of the best linear map student-residual -> teacher-residual per layer
   (and the reverse); 1.0 = same information, any linear re-encoding allowed
 * neuron census per MLP: each teacher neuron's pre-activation (on the teacher's own stream) vs every
   student neuron's pre-activation (on the student's own stream), same inputs X. best |corr|.
   Also student-side: duplicates (corr > 0.99 with another student neuron), dead (std tiny),
   and how many student neurons map to each teacher neuron.
"""
import sys
import torch
import torch.nn.functional as F

PEEL = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel"
sys.path.insert(0, PEEL)
from microt import build, rope_cos_sin  # noqa: E402

torch.set_grad_enabled(False)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def streams(m, X):
    x = X; cos, sin = rope_cos_sin(x.shape[1], m.hdim, m.cfg["theta"], x.device); res, pre = [], []
    for b in m.blocks:
        x = x + b.attn(b.norm1(x), cos, sin); res.append(x)                 # after attn
        z = b.mlp.fc1(b.norm2(x)); pre.append(z.reshape(-1, z.shape[-1]))
        x = x + b.mlp.fc2(F.gelu(z)); res.append(x)                        # after mlp
    return [r.reshape(-1, r.shape[-1]) for r in res], pre


def r2(A, B):
    """R^2 of least-squares B ~ A W + c"""
    A1 = torch.cat([A, torch.ones(len(A), 1, device=A.device)], 1).double(); B = B.double()
    W = torch.linalg.lstsq(A1, B).solution
    return 1 - ((A1 @ W - B).pow(2).sum() / (B - B.mean(0)).pow(2).sum()).item()


def zcorr(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return (F.normalize(A, dim=0).t() @ F.normalize(B, dim=0))


def census(t, s, X):
    Rt, Zt = streams(t, X); Rs, Zs = streams(s, X)
    labels = [f"B{i // 2 + 1}.{'attn' if i % 2 == 0 else 'mlp'}" for i in range(len(Rt))]
    out = {"r2": [(l, r2(a, b), r2(b, a)) for l, a, b in zip(labels, Rs, Rt)], "mlp": []}
    for zt, zs in zip(Zt, Zs):
        C = zcorr(zt, zs).abs()                         # (teacher n, student n)
        best = C.amax(1)
        S = zcorr(zs, zs).abs(); S.fill_diagonal_(0)
        sstd = zs.std(0)
        owner = C.argmax(0)                             # each student neuron's best teacher neuron
        good = C.amax(0) > 0.99
        per_t = torch.bincount(owner[good], minlength=zt.shape[1])
        out["mlp"].append(dict(n_t=zt.shape[1], n_s=zs.shape[1], found99=int((best > 0.99).sum()),
                               found95=int((best > 0.95).sum()), med=best.median().item(),
                               dup=int((S.amax(1) > 0.99).sum()), dead=int((sstd < 1e-3 * sstd.max()).sum()),
                               s_matched=int(good.sum()), multi=int((per_t > 1).sum())))
    return out


def fmt(c):
    r = "  ".join(f"{l} {a:.3f}/{b:.3f}" for l, a, b in c["r2"])
    m = "  ".join(f"B{i + 1}: teacher found {d['found99']}/{d['n_t']} (>.95 {d['found95']}, med {d['med']:.3f}) | "
                  f"student matched {d['s_matched']}/{d['n_s']} dup {d['dup']} dead {d['dead']} split-teacher {d['multi']}"
                  for i, d in enumerate(c["mlp"]))
    return r, m


if __name__ == "__main__":
    path = sys.argv[1]
    ck = torch.load(path, map_location=dev, weights_only=False)
    t = build(ck["cfg"], ck["teacher_sd"], dev)
    ss = [build(ck.get("student_cfg", ck["cfg"]), sd, dev) for sd in ck["student_sds"]]
    X = torch.randn(256, 32, ck["cfg"]["d_model"], generator=torch.Generator(device=dev).manual_seed(5), device=dev) * 0.5
    print(f"{path.split('/')[-1]} iter {ck['iter']}")
    print("  residual R^2 (student->teacher / teacher->student) per stage, then MLP neuron census by activation")
    for i in sorted(range(len(ss)), key=lambda i: ck["fit"][i]):
        r, m = fmt(census(t, ss[i], X))
        print(f"  m{i} fit {ck['fit'][i]:.3f}\n     R2  {r}\n     MLP {m}")
