"""Empirical basin of the CNN kink refiner: perturb the TEACHER's own L1/L2
rows by a controlled angle (unit-[W|b] rotation toward a random orthogonal
direction), run _cnn_refine_layer, and report solve rate + refined eps per
noise level. Sets the cheat-peel threshold from measurement instead of
guesswork: the gate should fire as soon as the layer's guess is inside the
angle basin found here."""
import math

import torch

from data import make_teacher_cnn
from method import _cnn_refine_layer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFGS = [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)]


def perturb_layer(net, li, angle_deg, gen):
    """Rotate each unit's [W|b] row by angle_deg toward a random orthogonal
    direction (in the unit-row sphere), preserving row norm."""
    W, b = net.layers[li].weight, net.layers[li].bias
    D = torch.cat([W.view(W.shape[0], -1), b[:, None]], 1)
    th = math.radians(angle_deg)
    for c in range(D.shape[0]):
        v = D[c]
        n = v.norm()
        u = torch.randn(v.shape, generator=gen, device=v.device)
        u = u - (u @ v) * v / n**2
        u = u / u.norm().clamp_min(1e-12)
        D[c] = (math.cos(th) * (v / n) + math.sin(th) * u) * n
    W.data.copy_(D[:, :-1].view_as(W))
    b.data.copy_(D[:, -1])


def eps_stats(net, teacher, li):
    Dt = torch.cat([teacher.layers[li].weight.view(-1, 1).view(
        teacher.layers[li].weight.shape[0], -1), teacher.layers[li].bias[:, None]], 1)
    Dn = torch.cat([net.layers[li].weight.view(net.layers[li].weight.shape[0], -1),
                    net.layers[li].bias[:, None]], 1)
    Dt = Dt / Dt.norm(dim=1, keepdim=True)
    Dn = Dn / Dn.norm(dim=1, keepdim=True)
    d = (Dt - Dn).abs()
    return d.max(1).values, d.mean(1)


def main():
    teacher = make_teacher_cnn((1, 28, 28), CFGS, (84,), 10, epochs=25, seed=0,
                               device=DEV, act="leaky_relu")
    gen = torch.Generator(device=DEV).manual_seed(0)
    for li, nm, angles in [(0, "L1(d=26)", [1, 2, 3, 4, 6, 8]),
                           (1, "L2(d=151)", [1, 2, 3, 4])]:
        print(f"\n== {nm} refine basin ==", flush=True)
        for a in angles:
            guess = teacher.clone()
            perturb_layer(guess, li, a, gen)
            gm, gmean = eps_stats(guess, teacher, li)
            Wr, br, mask = _cnn_refine_layer(teacher, guess, li, (1, 28, 28),
                                             DEV, "leaky_relu")
            n = mask.shape[0] if mask is not None else 0
            ok = int(mask.sum()) if mask is not None else 0
            post = ""
            if ok:
                sol = guess.clone()
                idx = mask.to(DEV).nonzero(as_tuple=True)[0]
                sol.layers[li].weight.data[idx] = Wr[idx].to(sol.layers[li].weight.dtype)
                sol.layers[li].bias.data[idx] = br[idx].to(sol.layers[li].bias.dtype)
                pm, _ = eps_stats(sol, teacher, li)
                post = f" | refined eps max {pm[idx].max():.1e}"
            print(f"  {a}deg: guess eps max {gm.max():.2e} mean {gmean.mean():.2e}"
                  f" -> solved {ok}/{n}{post}", flush=True)


if __name__ == "__main__":
    main()
