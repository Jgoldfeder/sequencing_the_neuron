"""Offline census of a LeNet committee snapshot (recon/_pop__*cnn*.pt: best member + pins)
against the cached teacher. Answers, per layer:
  * scale state: median [w|b] row norm (student vs teacher), pre-activation RMS, dead and
    always-on channel counts on cube and Gaussian probe batches  (is W1 inflating? dying?)
  * accuracy: aligned per-channel max error (median / #<1e-2 / #<1e-3 / max), sign flips,
    duplicate channels (two student channels matching the same teacher channel)
  * identifiability: kink-crossing balance min(P(z>0), P(z<0)) per channel for teacher and
    student -- channels that (almost) never cross their kink are affine on the query
    distribution and invisible to gradient descent
  * pins: partial / frozen masks per layer and the stored fp64 solved-row error
Teacher weights are used only for scoring.

Usage: python diag_lenet_snapshot.py recon/_pop__mergedbest_lenetpeel_base_cnn__*.pt [--device cuda:1]
"""
import argparse, glob, json, sys
import torch
from nets import ConvNet
from data import make_teacher_cnn
from align import cnn_canonicalize_, cnn_align_to_
import cnn_balance as cb


def rows(layer):
    W = layer.weight; k = W.shape[0]
    return torch.cat([W.reshape(k, -1), layer.bias.view(-1, 1)], 1)


@torch.no_grad()
def census(net, x, name):
    st = cb.layer_stats(net, x)
    print(f"  {name:8s} pre-act RMS " + " ".join(f"{r['rms']:6.2f}" for r in st[:-1])
          + " | row norm " + " ".join(f"{r['row_norm']:6.2f}" for r in st)
          + " | dead " + " ".join(f"{r['dead']:3d}" for r in st[:-1])
          + " | always-on " + " ".join(f"{r['always_on']:3d}" for r in st[:-1]))
    return st


@torch.no_grad()
def crossing(net, x):
    """per hidden layer: min(P(z>0), P(z<0)) per channel over batch x positions."""
    out = []
    for z in cb.hidden_preacts(net, x):
        zc = z.transpose(0, 1).reshape(z.shape[1], -1)
        p = (zc > 0).double().mean(1)
        out.append(torch.minimum(p, 1 - p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap", nargs="+")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    dev = a.device
    allres = {}
    for pat in a.snap:
        for path in sorted(glob.glob(pat)):
            ck = torch.load(path, map_location=dev, weights_only=False)
            shape, cfgs, fc, od, act = ck["input_shape"], ck["conv_cfgs"], ck["fc_dims"], ck["out_dim"], ck["act"]
            teacher = make_teacher_cnn(shape, cfgs, fc, od, epochs=25, seed=0, device=dev, act=act).double()
            stu = ConvNet(shape, cfgs, fc, od, act).to(dev).double()
            stu.load_state_dict({k: v.double() for k, v in ck["state_dict"].items()})
            print(f"=== {path}\n  iter {ck.get('iter')}  peel queries {ck.get('peel_refinement_queries')}")
            H = len(stu.layers) - 1
            pins = {}
            for l in range(H):
                pm = ck.get("partial_mask", {}).get(l); fz = ck.get("frozen", {}).get(l)
                npin = int(pm.sum()) if pm is not None else 0
                nfz = int(fz[2].sum()) if fz is not None else 0
                pins[l] = (npin, nfz)
            print("  pins (partial, frozen) per hidden layer:", pins)
            d = shape[0] * shape[1] * shape[2]
            g = torch.Generator(device=dev).manual_seed(0)
            xc = torch.rand(a.n, d, generator=g, device=dev, dtype=torch.double) * 2 - 1
            xg = torch.randn(a.n, d, generator=g, device=dev, dtype=torch.double) * 0.5
            print("  -- scale census (raw, unaligned):")
            st_s = census(stu, xc, "student"); census(teacher, xc, "teacher")
            print("  -- aligned per-channel error (max |dw|,|db| after cnn canonicalisation + alignment):")
            t = teacher.clone(); r = stu.clone()
            cnn_canonicalize_(t); cnn_canonicalize_(r); cnn_align_to_(r, t)
            res = {"iter": ck.get("iter"), "pins": pins, "layers": []}
            cr_t = crossing(teacher, xc); cr_s = crossing(stu, xc); cr_tg = crossing(teacher, xg)
            for l in range(H):
                A, B = rows(t.layers[l]), rows(r.layers[l])
                e = (A - B).abs().max(1).values
                cos = torch.nn.functional.cosine_similarity(A, B, dim=1)
                # duplicates: student channel whose best teacher match is already claimed better by another
                An = A / A.norm(dim=1, keepdim=True); Bn = B / B.norm(dim=1, keepdim=True)
                C = (Bn @ An.T).abs(); best = C.argmax(1)
                dup = int(len(best) - len(torch.unique(best)))
                one_sided_t = int((cr_t[l] < 0.01).sum()); one_sided_s = int((cr_s[l] < 0.01).sum())
                one_sided_tg = int((cr_tg[l] < 0.01).sum())
                bad = (e > 1e-2).nonzero().flatten().tolist()
                print(f"  L{l + 1} n={len(e):3d} med {float(e.median()):.1e} max {float(e.max()):.1e} "
                      f"<1e-2:{int((e < 1e-2).sum()):3d} <1e-3:{int((e < 1e-3).sum()):3d} <1e-6:{int((e < 1e-6).sum()):3d} "
                      f"flips {int((cos < 0).sum()):3d} dups {dup:2d} | one-sided(<1%% cross) teacher cube {one_sided_t} gauss {one_sided_tg} / student {one_sided_s}"
                      + (f" | bad ch (first 12): {bad[:12]}" if bad and len(bad) < len(e) else ""))
                if bad and len(bad) < len(e):
                    # for the bad channels: were they one-sided in the teacher?
                    bt = cr_t[l][bad]
                    print(f"      bad channels' teacher crossing balance: med {float(bt.median()):.3f}, "
                          f"{int((bt < 0.01).sum())}/{len(bad)} one-sided; good channels' med {float(cr_t[l][e <= 1e-2].median()) if int((e <= 1e-2).sum()) else float('nan'):.3f}")
                res["layers"].append(dict(n=len(e), med=float(e.median()), n_1e2=int((e < 1e-2).sum()),
                                          n_1e3=int((e < 1e-3).sum()), flips=int((cos < 0).sum()), dups=dup,
                                          one_sided_teacher=one_sided_t, one_sided_student=one_sided_s,
                                          student_row_norm=st_s[l]["row_norm"], dead=st_s[l]["dead"],
                                          always_on=st_s[l]["always_on"], rms=st_s[l]["rms"]))
            Ah, Bh = rows(t.layers[-1]), rows(r.layers[-1])
            print(f"  head: med row err {float((Ah - Bh).abs().max(1).values.median()):.2e}, student head norm {st_s[-1]['row_norm']:.2f}")
            allres[path] = res
    if a.json:
        json.dump(allres, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
