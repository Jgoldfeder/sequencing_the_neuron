"""Cross-run certificate: two INDEPENDENT committees (same black box, different
population seeds) -> a unit is certified iff the two runs' best members agree on it
(max-element <= eps on unit vectors, canonical gauge, units paired by activation,
compared over the inputs identified between the two runs). Teacher-free; the teacher
is used only to score. Independent initializations agreeing to 0.02 on a 200-entry
vector is not chance, so this certificate does not need several good members per
run (the within-run quorum does), only one good member per run.

usage: python cross_certificate.py <popA.pt> <popB.pt> [--eps 0.02]
"""
import sys, argparse, torch
from nets import MLP
from align import scale_normalize_, _preacts
from head_refine import _act_perm


def load(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False); dims = ck["dims"]; act = ck.get("act", "leaky_relu")
    t = MLP(dims, act=act).to(dev); t.load_state_dict(ck["teacher_state"])
    pop = []
    for sd in ck["pop_states"]:
        m = MLP(dims, act=act).to(dev); m.load_state_dict(sd); pop.append(m)
    return dims, t, pop


@torch.no_grad()
def cross_certificate(pathA, pathB, dev="cuda:0", eps=0.02, id_corr=0.99, n_probe=8192):
    dims, t, popA = load(pathA, dev); dimsB, tB, popB = load(pathB, dev)
    assert dims == dimsB and all(torch.equal(a, b) for a, b in zip(t.state_dict().values(), tB.state_dict().values()))
    nW = len(dims) - 1; nh = nW - 1
    torch.manual_seed(0); X = (torch.rand(4000, dims[0], device=dev) * 2 - 1)
    yt = t(X); lA = [float((m(X) - yt).abs().mean()) for m in popA]; lB = [float((m(X) - yt).abs().mean()) for m in popB]
    iA, iB = min(range(len(popA)), key=lambda i: lA[i]), min(range(len(popB)), key=lambda i: lB[i])
    A = popA[iA].clone(); B = popB[iB].clone(); T = t.clone()
    for n in (A, B, T):
        scale_normalize_(n)
    probe = (torch.rand(n_probe, dims[0], device=dev) * 2 - 1)
    ZA, ZB, ZT = _preacts(A, probe), _preacts(B, probe), _preacts(T, probe)
    pAB, cAB = zip(*[_act_perm(ZA[l], ZB[l]) for l in range(nh)]); pAT, cAT = zip(*[_act_perm(ZA[l], ZT[l]) for l in range(nh)])
    idAB = [c > id_corr for c in cAB]; idAT = [c > id_corr for c in cAT]
    out = {"bestA": (iA, lA[iA]), "bestB": (iB, lB[iB]), "ident_AB": [int(i.sum()) for i in idAB], "layers": {}}
    for k in range(nW):
        is_head = (k == nW - 1)
        if not is_head:
            m_in = torch.ones(dims[k], dtype=torch.bool, device=dev) if k == 0 else (idAB[k - 1] & idAT[k - 1])
            cols = m_in.nonzero().flatten()
            WA = torch.cat([A.layers[k].weight[:, cols], A.layers[k].bias[:, None]], 1)
            WBf = B.layers[k].weight[pAB[k]]; WBf = WBf[:, pAB[k - 1]] if k > 0 else WBf
            WB = torch.cat([WBf[:, cols], B.layers[k].bias[pAB[k]][:, None]], 1)
            WTf = T.layers[k].weight[pAT[k]]; WTf = WTf[:, pAT[k - 1]] if k > 0 else WTf
            WT = torch.cat([WTf[:, cols], T.layers[k].bias[pAT[k]][:, None]], 1)
            UA, UB, UT = [W / W.norm(dim=1, keepdim=True) for W in (WA, WB, WT)]
            D = (UA - UB).abs(); d = D.amax(1); eA = (UA - UT).abs().amax(1); eB = (UB - UT).abs().amax(1)
        else:
            HA = A.layers[k].weight; HB = B.layers[k].weight[:, pAB[k - 1]]; HT = T.layers[k].weight[:, pAT[k - 1]]
            UA, UB, UT = [H / H.norm(dim=0, keepdim=True) for H in (HA, HB, HT)]
            D = (UA - UB).abs(); d = D.amax(0); eA = (UA - UT).abs().amax(0); eB = (UB - UT).abs().amax(0)
        ok = d <= eps
        out["layers"][k] = dict(is_head=is_head, ok=ok, entries_agree=float((D <= eps).float().mean()),
                                err_ok=torch.maximum(eA, eB)[ok], eA=eA, eB=eB)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--eps", type=float, default=0.02); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    o = cross_certificate(a.a, a.b, a.device, a.eps)
    print(f"best members: A m{o['bestA'][0]} (MAE {o['bestA'][1]:.4f}), B m{o['bestB'][0]} (MAE {o['bestB'][1]:.4f}); units identified A<->B per hidden layer {o['ident_AB']}")
    print(f"{'matrix':>8} | {'units certified':>15} | {'entries agree':>13} | {'error of certified units vs teacher mean / max':>46} | disagreeing units: err A / err B")
    for k, v in o["layers"].items():
        ok = v["ok"]; dis = (~ok).nonzero().flatten().tolist(); e = v["err_ok"]
        print(f"{'W%d%s' % (k + 1, '=head' if v['is_head'] else ''):>8} | {int(ok.sum()):>11}/{len(ok)} | {v['entries_agree']:>12.1%} | "
              f"{(f'{e.mean():.1e} / {e.max():.1e}' if e.numel() else 'n/a'):>46} | " + (", ".join(f"{v['eA'][i]:.2f}/{v['eB'][i]:.2f}" for i in dis[:8]) + (" ..." if len(dis) > 8 else "") if dis else "none"))
