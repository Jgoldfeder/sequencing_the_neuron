"""ORACLE DIAGNOSTIC (cheat): focused disagreement objective for the frontier's UNSOLVED channels (2026-10-20).

Objective = || f_T(x) - f_T^{mask}(x) ||, f_T^{mask} = the teacher with its not-yet-pinned frontier channels forced to 0.
That difference is exactly the unsolved channels' contribution to the output (their 0.01 leak while silent, full size once
a kink is crossed); pinned channels cancel exactly, the students' junk rows and F6/head fit errors do not enter.  Oracle
ascent test: 16/16 silent C5 channels reached in 3 pin-and-reascend rounds.  Needs teacher internals -> cheat only.

Which teacher channels are "pinned" is read from the committee: the best member's pinned frontier rows are mapped to
teacher channels through the activation-free weight permutations of the (exact) layers below."""
import copy
import torch
import torch.nn as nn


def _rows(layer):
    W = layer.weight.detach().double(); return torch.cat([W.reshape(W.shape[0], -1), layer.bias.detach().double()[:, None]], 1)


def _unit(R):
    return R / R.norm(dim=1, keepdim=True).clamp_min(1e-300)


@torch.no_grad()
def teacher_channel_map(teacher, member, upto):
    """for layers 0..upto: perm[l][member channel] = teacher channel (cosine match after permuting inputs)."""
    perms = []; prev = None
    for l in range(upto + 1):
        Wm = member.layers[l].weight.detach().double()
        if prev is not None:
            inv = torch.argsort(prev)                      # teacher order of the member's input channels
            if Wm.dim() == 4:
                Wm = Wm[:, inv]
            else:
                cin = len(prev); Wm = Wm.reshape(Wm.shape[0], cin, -1)[:, inv].reshape(Wm.shape[0], -1)
        Rm = torch.cat([Wm.reshape(Wm.shape[0], -1), member.layers[l].bias.detach().double()[:, None]], 1)
        C = _unit(Rm) @ _unit(_rows(teacher.layers[l])).T
        prev = C.argmax(1); perms.append((prev, C.max(1).values))
    return perms


class MaskedTeacher(nn.Module):
    """teacher with the given frontier channels' pre-activations forced to 0 (act(0) = 0 -> no contribution)."""
    def __init__(self, teacher, layer, zero_channels):
        super().__init__()
        self.t = teacher; self.layer = layer
        self.register_buffer("keep", torch.ones(teacher.layers[layer].weight.shape[0], device=teacher.layers[layer].weight.device))
        self.keep[zero_channels] = 0.0

    def forward(self, x):
        h = self.t.layers[self.layer].register_forward_hook(
            lambda m, i, o: o * (self.keep.view(1, -1, *([1] * (o.dim() - 2)))).to(o.dtype))
        try:
            return self.t(x)
        finally:
            h.remove()


class DeadRowTeacher(nn.Module):
    """teacher copy whose unsolved frontier rows are replaced by RANDOM rows (teacher-row norm) with a strongly negative
    bias: the oracle test's 'dead student' reference.  Offline from the 103-row record this reached 17/17 unsolved C5
    channels in 3 ascend-pin rounds, where the zeroed-channel reference (MaskedTeacher) stalled at 6/17: the junk rows'
    occasional firing at large |x| drives the ascent far out in assorted directions, where silent teacher channels fire."""
    def __init__(self, teacher, layer, channels, bias=-5.0, seed=0):
        super().__init__()
        self.t = copy.deepcopy(teacher)
        W = self.t.layers[layer].weight.data; b = self.t.layers[layer].bias.data
        g = torch.Generator(device=W.device).manual_seed(seed)
        for j in channels:
            W[j] = torch.randn(W[j].shape, generator=g, device=W.device, dtype=W.dtype) * W[j].norm() / W[j].numel() ** 0.5
            b[j] = bias
        for p_ in self.t.parameters():
            p_.requires_grad_(False)

    def forward(self, x):
        return self.t(x)


class Padded(nn.Module):
    """append a large constant output so the committee's per-sample L1 normalisation is ~ a constant rescale (keeps
    the raw size of the leak gap, which the normalisation would otherwise divide out)."""
    def __init__(self, net, c=1e3):
        super().__init__(); self.net = net; self.c = c

    def forward(self, x):
        y = self.net(x); return torch.cat([y, torch.full_like(y[:, :1], self.c)], 1)


def focus_pair(teacher, member, frontier, pinned_mask, verbose=True, mode="dead", seed=0):
    """returns [Padded(masked teacher), Padded(teacher)] for gen_queries(..., ref_last=True), and the unsolved teacher set."""
    perms = teacher_channel_map(teacher, member, frontier)
    pm, cos = perms[frontier]
    pinned_t = set(pm[pinned_mask.to(pm.device).bool() & (cos > 0.9)].tolist())   # match only up to the per-channel gauge of the layers below (cos ~0.99)
    C = teacher.layers[frontier].weight.shape[0]
    unsolved = sorted(set(range(C)) - pinned_t)
    if verbose:
        print(f"  [cheat-focus] frontier L{frontier + 1}: {len(pinned_t)} teacher channels pinned, {len(unsolved)} unsolved "
              f"(zeroed in the reference): {unsolved}", flush=True)
    ref = DeadRowTeacher(teacher, frontier, unsolved, seed=seed) if mode == "dead" else MaskedTeacher(teacher, frontier, unsolved)
    return [Padded(ref), Padded(teacher)], unsolved
