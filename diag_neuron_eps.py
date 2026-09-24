"""Per-neuron weight-error histogram of a CNN student checkpoint vs its teacher.

Canonicalizes + channel-aligns the student into the teacher frame (function-
preserving), then reports, per layer, how many neurons are at low [W|b] inf-norm
error -- i.e. how many are actually recovered / refiner-ready -- not just the
layer mean the run log prints.

Usage: python diag_neuron_eps.py <checkpoint.pt> [--epochs 25]
"""
import argparse

import torch

from align import cnn_canonicalize_, cnn_align_to_
from data import make_teacher_cnn
from nets import ConvNet

THRESHOLDS = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ishape = tuple(ck["input_shape"]); cfgs = [tuple(c) for c in ck["conv_cfgs"]]
    fc = tuple(ck["fc_dims"]); out = ck["out_dim"]; act = ck["act"]
    print(f"checkpoint iter {ck.get('iter', '?')}  {ishape} conv{cfgs} fc{fc} out{out}")

    teacher = make_teacher_cnn(ishape, cfgs, fc, out, epochs=args.epochs,
                               seed=args.seed, device=dev, act=act)
    student = ConvNet(ishape, cfgs, fc, out, act).to(dev)
    student.load_state_dict(ck["state_dict"])

    t, r = teacher.clone(), student.clone()
    cnn_canonicalize_(t); cnn_canonicalize_(r); cnn_align_to_(r, t)

    nlay = len(t.layers)
    print(f"\n{'layer':10s} {'n':>5s} {'med':>9s} {'min':>9s} | " +
          " ".join(f"<{x:g}".rjust(8) for x in THRESHOLDS))
    for li in range(nlay):
        Wt, bt = t.layers[li].weight, t.layers[li].bias
        Wr, br = r.layers[li].weight, r.layers[li].bias
        Dt = torch.cat([Wt.reshape(Wt.shape[0], -1), bt[:, None]], 1)
        Dr = torch.cat([Wr.reshape(Wr.shape[0], -1), br[:, None]], 1)
        if li < nlay - 1:                         # hidden: unit-norm rows (canonical)
            Dt = Dt / Dt.norm(dim=1, keepdim=True).clamp_min(1e-12)
            Dr = Dr / Dr.norm(dim=1, keepdim=True).clamp_min(1e-12)
        e = (Dt - Dr).abs().max(1).values
        counts = " ".join(f"{int((e < x).sum()):8d}" for x in THRESHOLDS)
        name = "head" if li == nlay - 1 else f"L{li+1}"
        print(f"{name:10s} {len(e):5d} {e.median():9.2e} {e.min():9.2e} | {counts}")
    print("\n(hidden layers scored on unit-[W|b] rows; head raw. counts are "
          "cumulative: '<1e-2' includes '<1e-3'.)")


if __name__ == "__main__":
    main()
