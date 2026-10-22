"""Gauge / isomorphism test for the LeNet-5 (1998) ConvNet (act="lenet5").

Builds a random net, generates functionally IDENTICAL copies by applying the network's symmetries
  (a) a C3-table automorphism on C1/S2 + C3/S4 (12 of them),
  (b) free permutations of C5 and F6,
  (c) independent sign flips of every conv/fc unit and of every subsampling unit (odd squashing),
then checks: same function; canonicalise + align recovers the reference parameters exactly; the CNN quorum consensus
reports every unit in consensus; the isolated two-sided report pairs every unit.  Then repeats with small noise.
Run:  python test_lenet5_isomorphisms.py
"""
import random
import sys
import torch
import align
from nets import ConvNet
import cnn_iso

torch.manual_seed(0); random.seed(0)
# usage: python test_lenet5_isomorphisms.py [lenet5|tanh]   (tanh = the modern LeNet-5 with tanh: 28x28, full C3, linear head)
ARCH = sys.argv[1] if len(sys.argv) > 1 else "lenet5"
CFG = (((1, 32, 32), [(1, 6, 5, 1, 0, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)], (84,), 10, "lenet5") if ARCH == "lenet5" else
       ((1, 28, 28), [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)], (84,), 10, ARCH))
D_IN = 1024 if ARCH == "lenet5" else 784


def make():
    n = ConvNet(*CFG).double()
    with torch.no_grad():
        for p in n.parameters():
            if p.requires_grad:
                p.add_(0.3 * torch.randn_like(p))
        for i in n.masked:
            n.layers[i].weight.mul_(getattr(n, f"conn_mask{i}"))
    return n


@torch.no_grad()
def scramble(n, gen):
    m = n.clone()
    if m.lenet5:                                     # C1/C3 only via the table's automorphisms
        autos = align.lenet5_c3_automorphisms(); p1, p3 = autos[int(torch.randint(len(autos), (1,), generator=gen))]
        align._permute_unit_block_(m, 0, torch.tensor(p1)); align._permute_unit_block_(m, 1, torch.tensor(p3))
    free = (2, 3) if m.lenet5 else range(len(m.layers) - 1)
    for i in free:
        align._permute_unit_block_(m, i, torch.randperm(m.layers[i].weight.shape[0], generator=gen))
    for i in range(len(m.layers) - 1):
        for k in range(m.layers[i].weight.shape[0]):
            if torch.rand(1, generator=gen) < 0.5:
                align._lenet5_flip_(m, i, k, -1.0)
            if i in m.sub_of and torch.rand(1, generator=gen) < 0.5:     # flip the subsampling unit itself
                j = m.sub_of[i]; m.sub_w[j][k].mul_(-1); m.sub_b[j][k].mul_(-1); align._push_unit_(m, i, k, -1.0)
    return m


def params(n):
    return torch.cat([p.detach().reshape(-1) for p in n.parameters()])


def main():
    g = torch.Generator().manual_seed(1)
    ref = make(); X = torch.randn(256, D_IN, dtype=torch.double)
    ok = True
    copies = [scramble(ref, g) for _ in range(8)]
    fdiff = max(float((c(X) - ref(X)).abs().max()) for c in copies)
    pdiff_raw = min(float((params(c) - params(ref)).abs().max()) for c in copies)
    print(f"[1] scrambled copies: max |f - f_ref| = {fdiff:.2e} (function preserved), params differ by >= {pdiff_raw:.2f} (they ARE scrambled)")
    ok &= fdiff < 1e-10 and pdiff_raw > 0.1
    rc = ref.clone(); align.cnn_canonicalize_(rc)
    worst = 0.0
    for c in copies:
        cc = c.clone(); align.cnn_canonicalize_(cc); align.cnn_align_to_(cc, rc)
        worst = max(worst, float((params(cc) - params(rc)).abs().max()))
        assert float((cc(X) - ref(X)).abs().max()) < 1e-10
    print(f"[2] canonicalise + align: max parameter difference to the canonical reference = {worst:.2e}")
    ok &= worst < 1e-10
    import method
    pop = [c.float() for c in copies]
    _, per_layer, masks = method._cnn_consensus(pop, 0, 0.02, 5)
    print(f"[3] quorum consensus (eps 0.02, 5/8) on exact copies: {per_layer}  head {int(masks[-1].sum())}/{masks[-1].numel()}")
    ok &= all(nc == nt for nt, nc in per_layer)
    rep = cnn_iso.cnn_two_sided(pop, X[:128].float(), 0, 0.02, 5)
    print("[4] isolated report on exact copies:", cnn_iso.fmt(rep))
    ok &= all(v["paired"] == v["n"] for v in rep.values())
    noisy = []
    for c in copies:
        d = c.clone()
        with torch.no_grad():
            for p in d.parameters():
                if p.requires_grad:
                    p.add_(1e-3 * torch.randn_like(p))
            for i in d.masked:
                d.layers[i].weight.mul_(getattr(d, f"conn_mask{i}"))
        noisy.append(d.float())
    _, per_layer_n, masks_n = method._cnn_consensus(noisy, 0, 0.02, 5)
    print(f"[5] consensus on copies + 1e-3 noise: {per_layer_n}  head {int(masks_n[-1].sum())}/{masks_n[-1].numel()}")
    ok &= all(nc == nt for nt, nc in per_layer_n)
    rep_n = cnn_iso.cnn_two_sided(noisy, X[:128].float(), 0, 0.02, 5)
    print("[6] isolated report on noisy copies:", cnn_iso.fmt(rep_n))
    print("ALL ISOMORPHISM TESTS PASSED" if ok else "SOME ISOMORPHISM TESTS FAILED")


if __name__ == "__main__":
    main()
