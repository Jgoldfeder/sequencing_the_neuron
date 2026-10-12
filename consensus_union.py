"""Cross-run consensus: do two INDEPENDENT committees (same black box, different
population seeds) certify the same units, and does their union cover more?

Teacher-free core: run B's reference member is paired to run A's reference member
by activation matching, layer by layer, so B's consensus masks and values can be
expressed in A's frame. The teacher is used only to SCORE the result.

usage: python consensus_union.py <popA.pt> <popB.pt>
"""
import sys, types, torch
from nets import MLP
from align import _preacts, param_errors_actmatch, pairing_preacts
from head_refine import _act_perm
from consensus_eps import entry_coverage


def load(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False); dims = ck["dims"]; act = ck.get("act", "leaky_relu")
    t = MLP(dims, act=act).to(dev); t.load_state_dict(ck["teacher_state"])
    pop = []
    for sd in ck["pop_states"]:
        m = MLP(dims, act=act).to(dev); m.load_state_dict(sd); pop.append(m)
    return ck, dims, t, pop


@torch.no_grad()
def cross_run(pathA, pathB, dev="cuda:0", n_probe=8192):
    ckA, dims, t, popA = load(pathA, dev); ckB, dimsB, tB, popB = load(pathB, dev)
    assert dims == dimsB
    assert all(torch.equal(a, b) for a, b in zip(t.state_dict().values(), tB.state_dict().values())), "different black boxes"
    nW = len(dims) - 1; nh = nW - 1
    torch.manual_seed(0); X = (torch.rand(4000, dims[0], device=dev) * 2 - 1)
    with torch.no_grad():
        yt = t(X); lossA = [float((m(X) - yt).abs().mean()) for m in popA]; lossB = [float((m(X) - yt).abs().mean()) for m in popB]
    cfg = types.SimpleNamespace(reverse_eps=0.02, reverse_thresh=1.0, cluster_quorum=0.625, qg_init="uniform", qg_range=1.0)
    torch.manual_seed(0); EA = entry_coverage(popA, t, lossA, cfg, dev)
    torch.manual_seed(0); EB = entry_coverage(popB, t, lossB, cfg, dev)
    refA = min(range(len(popA)), key=lambda i: lossA[i]); refB = min(range(len(popB)), key=lambda i: lossB[i])
    # teacher-free pairing of B's frame to A's frame (hidden layers), on a fresh probe
    probe = (torch.rand(n_probe, dims[0], device=dev) * 2 - 1)
    ZA = pairing_preacts(popA[refA], probe); ZB = pairing_preacts(popB[refB], probe)
    pBA, cBA = zip(*[_act_perm(ZA[l], ZB[l]) for l in range(nh)])      # A unit i <-> B unit pBA[l][i]
    out = {}
    for k in range(nW):
        is_head = (k == nW - 1)
        a, b = EA[k], EB[k]
        # remap B's masks/values into A's frame: rows (if hidden) by pBA[k], cols (if k>0) by pBA[k-1]
        def toA(Mb):
            M = Mb
            if not is_head:
                M = M[pBA[k]]
            if k > 0:
                M = M[:, pBA[k - 1]]
            return M
        rowA = a["row_cov"] | a["col_cov"]; rowB = toA(b["row_cov"] | b["col_cov"])
        VA, VB = a["V"], toA(b["V"])
        both = rowA & rowB; onlyA = rowA & ~rowB; onlyB = rowB & ~rowA; neither = ~(rowA | rowB)
        # union value: A where A covers, else B
        V = torch.where(rowA, VA, torch.where(rowB, VB, a["M_ref"]))
        # teacher in A's frame for scoring: reuse A's error map => E_u for A-covered, else B's remapped error
        Wt_err_B = toA(b["E_u"])
        E = torch.where(rowA, a["E_u"], torch.where(rowB, Wt_err_B, a["E_ref"]))
        # unit-level: a unit (row unit for hidden, col unit for head) certified if ANY of its entries covered
        if not is_head:
            uA = a["row_cov"].any(1); uB = toA(b["row_cov"]).any(1) if True else None
            uB = (b["row_cov"][pBA[k]]).any(1)
        else:
            uA = a["col_cov"].any(0); uB = b["col_cov"][:, pBA[k - 1]].any(0)
        out[k] = dict(both=both, onlyA=onlyA, onlyB=onlyB, neither=neither, E=E, V=V, uA=uA, uB=uB,
                      pair_corr=(float(cBA[k - 1].min()) if (is_head and k > 0) else (float(cBA[k].min()) if not is_head else 1.0)),
                      is_head=is_head)
    return out, dims, lossA, lossB


def fmt(x):
    return f"mean {x.mean():.1e} max {x.max():.1e}" if x.numel() else "n/a"


if __name__ == "__main__":
    pa, pb = sys.argv[1], sys.argv[2]
    out, dims, lossA, lossB = cross_run(pa, pb)
    print(f"A = {pa.split('/')[-1]}  (best MAE {min(lossA):.3f})\nB = {pb.split('/')[-1]}  (best MAE {min(lossB):.3f})")
    print("units certified (rows for hidden layers, columns for the head) and ENTRY coverage, B mapped into A's frame by activation pairing:")
    for k, v in out.items():
        nm = f"W{k + 1}{'=head' if v['is_head'] else ''}"
        uA, uB = v["uA"], v["uB"]; n = len(uA)
        print(f"{nm}: units A {int(uA.sum())}/{n}  B {int(uB.sum())}/{n}  both {int((uA & uB).sum())}  only-A {int((uA & ~uB).sum())}  only-B {int((~uA & uB).sum())}  neither {int((~uA & ~uB).sum())}  | min pairing corr A<->B {v['pair_corr']:.3f}")
        tot = v["both"].numel()
        print(f"      entries: A {float((v['both'] | v['onlyA']).float().mean()):.1%}  B {float((v['both'] | v['onlyB']).float().mean()):.1%}  union {float((~v['neither']).float().mean()):.1%}  "
              f"| err both {fmt(v['E'][v['both']])} | only-A {fmt(v['E'][v['onlyA']])} | only-B {fmt(v['E'][v['onlyB']])} | neither {fmt(v['E'][v['neither']])}")
