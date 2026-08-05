"""Split the consensus by cluster size: units where ALL members agree vs units
where only a quorum (most) agree, and report max_eps within each group.

Mirrors build_consensus() but tags each reconstructed hidden unit with how many
committee members formed its cluster, then aligns to the teacher and reports the
per-unit parameter error broken down by agreement level. Read-only."""
import argparse
import math
from collections import defaultdict

import torch

from nets import MLP
from align import scale_normalize_, greedy_perm, permute_layer_


def load_net(dims, state, dev):
    m = MLP(dims).to(dev); m.load_state_dict(state); return m


@torch.no_grad()
def build_consensus_tracked(pop, dims, quorum_ratio=0.625, eps=0.02):
    """build_consensus + per-unit cluster size. Returns (net, sizes) or (None,None)."""
    P, H = len(pop), dims[1]
    quorum = math.ceil(quorum_ratio * P)
    mem = [m.clone() for m in pop]
    for r in mem:
        scale_normalize_(r)
    F = [torch.cat([r.layers[0].weight, r.layers[0].bias[:, None]], 1) for r in mem]
    parent = list(range(P * H))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for a in range(P):
        for b in range(a + 1, P):
            D = torch.cdist(F[a].float(), F[b].float(), p=float("inf"))
            nn_ab, nn_ba = D.argmin(1), D.argmin(0)
            for i in range(H):
                j = int(nn_ab[i])
                if int(nn_ba[j]) == i and D[i, j] < eps:
                    ra, rb = find(a * H + i), find(b * H + j)
                    if ra != rb:
                        parent[ra] = rb
    clusters = defaultdict(list)
    for node in range(P * H):
        clusters[find(node)].append(node)
    good = sorted((v for v in clusters.values() if len(v) >= quorum),
                  key=len, reverse=True)[:H]
    if len(good) < H:
        return None, None
    net = mem[0].clone()
    sizes = []
    for k, comp in enumerate(good):
        rows = torch.stack([mem[n // H].layers[0].weight[n % H] for n in comp])
        bias = torch.stack([mem[n // H].layers[0].bias[n % H] for n in comp])
        outs = torch.stack([mem[n // H].layers[1].weight[:, n % H] for n in comp])
        net.layers[0].weight.data[k] = rows.mean(0)
        net.layers[0].bias.data[k] = bias.mean(0)
        net.layers[1].weight.data[:, k] = outs.mean(0)
        sizes.append(len(comp))
    net.layers[1].bias.data = mem[0].layers[1].bias.clone()
    return net, sizes


@torch.no_grad()
def per_unit_errors(net, sizes, teacher):
    """Align consensus to teacher (hidden layer) and return (sizes_aligned,
    per-unit max param error, output-bias error)."""
    t = teacher.clone(); r = net.clone()
    scale_normalize_(t); scale_normalize_(r)
    A0 = torch.cat([t.layers[0].weight, t.layers[0].bias[:, None]], 1)
    B0 = torch.cat([r.layers[0].weight, r.layers[0].bias[:, None]], 1)
    perm = greedy_perm(A0, B0)
    permute_layer_(r, 0, perm)                       # reorders r units + w1 cols
    sizes_al = [sizes[perm[i]] for i in range(len(perm))]
    H = len(perm)
    errs = []
    for i in range(H):
        e = (t.layers[0].weight[i] - r.layers[0].weight[i]).abs().max().item()
        e = max(e, (t.layers[0].bias[i] - r.layers[0].bias[i]).abs().item())
        e = max(e, (t.layers[1].weight[:, i] - r.layers[1].weight[:, i]).abs().max().item())
        errs.append(e)
    outbias = (t.layers[1].bias - r.layers[1].bias).abs().max().item()
    return sizes_al, errs, outbias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quorum", type=float, default=0.625)
    args = ap.parse_args()
    dev = args.device

    print(f"[load] {args.dump} ...", flush=True)
    ck = torch.load(args.dump, map_location="cpu", weights_only=False)
    dims = ck["dims"]
    teacher = load_net(dims, ck["teacher_state"], dev)
    pop = [load_net(dims, s, dev) for s in ck["pop_states"]]
    P = len(pop)
    print(f"[load] dims={dims} members={P} quorum={math.ceil(args.quorum*P)}/{P}",
          flush=True)

    net, sizes = build_consensus_tracked(pop, dims, quorum_ratio=args.quorum)
    if net is None:
        print("[consensus] n/a at this quorum"); return
    sizes_al, errs, outbias = per_unit_errors(net, sizes, teacher)

    # histogram of agreement levels
    hist = defaultdict(int)
    for s in sizes_al:
        hist[s] += 1
    print("\n[agreement histogram] cluster size -> #units:", flush=True)
    for s in sorted(hist, reverse=True):
        tag = "ALL agree" if s == P else "most agree"
        print(f"   {s}/{P} ({tag}): {hist[s]} units", flush=True)

    # split: all-agree (size==P) vs most-agree (quorum<=size<P)
    all_e = [e for e, s in zip(errs, sizes_al) if s == P]
    most_e = [e for e, s in zip(errs, sizes_al) if s < P]

    def stat(name, xs):
        if not xs:
            print(f"   {name:24s}: (none)"); return
        xs_sorted = sorted(xs, reverse=True)
        print(f"   {name:24s}: n={len(xs):4d}  max={xs_sorted[0]:.3e}  "
              f"p99={xs_sorted[max(0,len(xs)//100)]:.3e}  "
              f"median={sorted(xs)[len(xs)//2]:.3e}", flush=True)

    print("\n[per-unit max_eps by agreement level]:", flush=True)
    stat("ALL agree (8/8)", all_e)
    stat("most agree (5-7/8)", most_e)
    print(f"   output-bias error       : {outbias:.3e}", flush=True)
    print(f"\n[overall] consensus max_eps = {max(max(errs), outbias):.3e}  "
          f"(location: "
          f"{'output-bias' if outbias >= max(errs) else ('ALL-agree unit' if sizes_al[max(range(len(errs)), key=lambda i: errs[i])] == P else 'most-agree unit')})",
          flush=True)


if __name__ == "__main__":
    main()
