"""Closed-form least-squares re-solve of a CNN's LINEAR head with the trunk frozen (black-box outputs only).
    python head_lsq_cnn.py recon/_refined_....pt [--points 20000] [--out recon/_..._headlsq.pt]"""
import argparse, copy, torch
from data import make_teacher_cnn
from nets import ConvNet
from align import cnn_param_errors

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--points", type=int, default=20000)
ap.add_argument("--x-std", type=float, default=0.5); ap.add_argument("--out", default=""); a = ap.parse_args()
dev = "cuda"; d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
shape, cfgs, fc, out, act = tuple(d["input_shape"]), d["conv_cfgs"], tuple(d["fc_dims"]), d["out_dim"], d["act"]
t64 = make_teacher_cnn(shape, cfgs, fc, out, epochs=25, seed=0, device=dev, act=act, verbose=False).clone().double()
net = ConvNet(shape, cfgs, fc, out, act).to(dev).double(); net.load_state_dict({k: v.double() for k, v in d["state_dict"].items()})
before = cnn_param_errors(copy.deepcopy(net), t64)
g = torch.Generator(device=dev).manual_seed(7)
X = torch.randn(a.points, int(torch.tensor(shape).prod()), device=dev, dtype=torch.float64, generator=g) * a.x_std
with torch.no_grad():
    Y = t64(X)                                                     # black-box outputs (the only teacher use)
    feats = {}
    h = net.layers[-1].register_forward_hook(lambda m, i, o: feats.__setitem__("x", i[0]))
    net(X); h.remove()
    H = torch.cat([feats["x"], torch.ones(len(X), 1, device=dev, dtype=torch.float64)], 1)
    sol = torch.linalg.lstsq(H, Y).solution                        # (85, 10)
    net.layers[-1].weight.copy_(sol[:-1].T); net.layers[-1].bias.copy_(sol[-1])
after = cnn_param_errors(copy.deepcopy(net), t64)
fmt = lambda e: " ".join(f"{v:.2e}" for v in e["max_eps_per_matrix"])
print(f"[head-lsq] {a.points:,} black-box queries | per-matrix max eps before: {fmt(before)}")
print(f"[head-lsq]                         after : {fmt(after)}   (overall max {after['max_eps']:.2e})")
if a.out:
    d2 = dict(d); d2["state_dict"] = net.state_dict(); torch.save(d2, a.out); print("[head-lsq] saved", a.out)
