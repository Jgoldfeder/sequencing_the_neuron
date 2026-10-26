"""Per-layer recovery of (wide or narrow) MicroT students in the TEACHER's coordinates (gaussian mode).
Residual gauge: at every stage k the student stream h_s ~ A_k h_t (least squares); a module reading stage k
sees norm(A_k h_t) = A_k h_t * r_k / rms(h_t) with r_k = rms(h_t)/rms(A_k h_t)  (gains folded, canon_);
its effective read matrix in teacher coords is W_s A_k r_k; a write into stage k+1 maps back by A_{k+1}^-1.
Per module:  func = rel. error of the module's output (teacher input mapped in, output mapped back)
             weights = rel. Frobenius error of the mapped weights vs the teacher's
MLP neurons: matched by pre-activation |corr| > 0.99; copies merged scale-weighted; flipped neurons use
GELU(-z) = GELU(z) - z (row sign flipped, the linear term is part of the module output, not of the neuron).
Dead teacher neurons (fc1 row norm < 1% of median) are excluded."""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/tmp/claude-1001/-home-judah/d2136c58-1893-4037-b72c-aee248efb5e0/scratchpad")
from neuron_census import zcorr, build, dev  # noqa: E402
from gpt_consensus import canon_  # noqa: E402
from microt import rope_cos_sin  # noqa: E402
from microt_align import qk_circuit, ov_circuit  # noqa: E402

torch.set_grad_enabled(False)


def stages(m, X, cos, sin):
    x = X; S = {"in": x}; io = {}
    for i, b in enumerate(m.blocks):
        a = b.attn(b.norm1(x), cos, sin); io[f"B{i+1}.attn"] = a; x = x + a; S[f"B{i+1}.attn"] = x
        o = b.mlp(b.norm2(x)); io[f"B{i+1}.mlp"] = o; x = x + o; S[f"B{i+1}.mlp"] = x
    return {k: v.reshape(-1, v.shape[-1]).double() for k, v in S.items()}, io


def rms(v):
    return v.pow(2).mean(-1, keepdim=True).sqrt()


def relF(a, b):
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()


def report(ck, mi, X):
    cfg = ck["cfg"]; scfg = ck.get("student_cfg", cfg)
    t = canon_(build(cfg, ck["teacher_sd"], dev), True); s = canon_(build(scfg, ck["student_sds"][mi], dev), True)
    cos, sin = rope_cos_sin(X.shape[1], t.hdim, cfg["theta"], dev)
    St, IOt = stages(t, X, cos, sin); Ss, IOs = stages(s, X, cos, sin)
    keys = list(St); A = {}; r = {}; fitres = {}
    for k in keys:
        A[k] = torch.linalg.lstsq(St[k], Ss[k]).solution.t()
        fitres[k] = relF(St[k] @ A[k].t(), Ss[k])
        r[k] = (rms(St[k]) / rms(St[k] @ A[k].t())).median().item()
    prev = {f"B{i+1}.attn": ("in" if i == 0 else f"B{i}.mlp") for i in range(len(t.blocks))}
    prev.update({f"B{i+1}.mlp": f"B{i+1}.attn" for i in range(len(t.blocks))})
    out = {"gauge": {k: (fitres[k], torch.linalg.svdvals(A[k]).min().item(), torch.linalg.svdvals(A[k]).max().item()) for k in keys}}
    for i in range(len(t.blocks)):
        tb, sb = t.blocks[i], s.blocks[i]
        for kind in ("attn", "mlp"):
            name = f"B{i+1}.{kind}"; kin, kout = prev[name], name
            Ain, Aout, rin = A[kin], A[kout], r[kin]
            # functional: student module's write, mapped back, vs the teacher module's write (same tokens)
            o_s = IOs[name].reshape(-1, IOs[name].shape[-1]).double()
            o_t = IOt[name].reshape(-1, o_s.shape[-1]).double()
            # (student module evaluated on its own stream, which = A_in h_t up to the gauge-fit residual)
            func = relF(o_s @ torch.linalg.inv(Aout).t(), o_t)
            if kind == "attn":
                qk = torch.stack([qk_circuit(tb, h) for h in range(tb.attn.n_head)]).double()
                qks = torch.stack([(rin ** 2) * Ain.t() @ qk_circuit(sb, h).double() @ Ain for h in range(sb.attn.n_head)])
                ov = torch.stack([ov_circuit(tb, h) for h in range(tb.attn.n_head)]).double()
                ovs = torch.stack([rin * torch.linalg.solve(Aout, ov_circuit(sb, h).double()) @ Ain for h in range(sb.attn.n_head)])
                out[name] = dict(func=func, w=f"qk {relF(qks, qk):.3f}  ov {relF(ovs, ov):.3f}")
            else:
                W1t, W2t = tb.mlp.fc1.weight.double(), tb.mlp.fc2.weight.double()
                W1s = sb.mlp.fc1.weight.double() @ Ain * rin; W2s = torch.linalg.solve(Aout, sb.mlp.fc2.weight.double())
                zt = (St[kin] / rms(St[kin])) @ W1t.t(); zs = (St[kin] / rms(St[kin])) @ W1s.t()   # same normalized teacher input
                C = zcorr(zt.float(), zs.float())
                live = W1t.norm(dim=1) > 0.01 * W1t.norm(dim=1).median()
                e_in, e_out, n_ok = [], [], 0
                for u in live.nonzero().flatten().tolist():
                    cl = (C[u].abs() > 0.99).nonzero().flatten()
                    if len(cl) == 0:
                        continue
                    n_ok += 1
                    j = cl[C[u, cl].abs().argmax()]; sg = torch.sign(C[u, j]).double()
                    e_in.append(relF(sg * W1s[j], W1t[u]))
                    # merged output column: least-squares weights of each copy's activation on the teacher's
                    gt = F.gelu(zt[:, u]); G = F.gelu(zs[:, cl])
                    sgn = torch.sign(C[u, cl]).double()
                    # copy j contributes W2s[:,j]*gelu(z_j); flipped copy: gelu(-z) = gelu(z) - z -> coefficient -1 on gelu(z_t)
                    coef = torch.stack([torch.linalg.lstsq(gt[:, None], (G[:, k] if sgn[k] > 0 else F.gelu(-zs[:, cl[k]]))[:, None]).solution[0, 0]
                                        for k in range(len(cl))])
                    col = sum(W2s[:, cl[k]] * coef[k] * sgn[k] for k in range(len(cl)))
                    e_out.append(relF(col, W2t[:, u]))
                e_in, e_out = torch.tensor(e_in), torch.tensor(e_out)
                out[name] = dict(func=func, w=f"neurons {n_ok}/{int(live.sum())} live | in-rows median {e_in.median():.3f} max {e_in.max():.3f} | out-cols median {e_out.median():.3f} max {e_out.max():.3f}")
    kf = keys[-1]
    Et = t.embed.weight.double(); Es = s.embed.weight.double() @ A[kf] * r[kf]
    out["unembed"] = dict(func=relF((Ss[kf] / rms(Ss[kf])) @ s.embed.weight.double().t(), (St[kf] / rms(St[kf])) @ Et.t()),
                          w=f"E {relF(Es, Et):.3f}")
    return out


if __name__ == "__main__":
    ck = torch.load(sys.argv[1], map_location=dev, weights_only=False)
    X = torch.randn(256, 32, ck["cfg"]["d_model"], generator=torch.Generator(device=dev).manual_seed(5), device=dev) * 0.5
    order = sorted(range(len(ck["student_sds"])), key=lambda i: ck["fit"][i])
    print(f"{sys.argv[1].split('/')[-1]}: iter {ck['iter']} | fits {[round(f, 3) for f in ck['fit']]}")
    for mi in order[:int(sys.argv[2]) if len(sys.argv) > 2 else 3]:
        rep = report(ck, mi, X)
        g = rep.pop("gauge")
        print(f"\n m{mi} (fit {ck['fit'][mi]:.3f})   gauge fit residual per stage: " + "  ".join(f"{k} {v[0]:.3f} [sv {v[1]:.2f}-{v[2]:.2f}]" for k, v in g.items()))
        for k, v in rep.items():
            print(f"   {k:9s} func err {v['func']:.3f} | weights {v['w']}")
