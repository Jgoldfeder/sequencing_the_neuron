"""Which FREE (training-time) per-unit signal flags the under-identified unit?
Per hidden unit we compute statistics that cost nothing during training (they
use pre-activations already produced on the forward pass), plus committee cluster
size, and report how each ranks the actually-worst unit. Signals are measured on
the consensus (attacker's view); teacher recon-error is ground truth for ranking."""
import argparse
import math
from collections import defaultdict
import torch
from nets import MLP
from align import scale_normalize_, greedy_perm, permute_layer_


def load_net(dims, state, dev):
    m = MLP(dims).to(dev); m.load_state_dict(state); return m


@torch.no_grad()
def consensus_with_sizes(pop, dims, quorum_ratio=0.625, eps=0.02):
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
            ab, ba = D.argmin(1), D.argmin(0)
            for i in range(H):
                j = int(ab[i])
                if int(ba[j]) == i and D[i, j] < eps:
                    ra, rb = find(a * H + i), find(b * H + j)
                    if ra != rb:
                        parent[ra] = rb
    cl = defaultdict(list)
    for n in range(P * H):
        cl[find(n)].append(n)
    good = sorted((v for v in cl.values() if len(v) >= quorum), key=len,
                  reverse=True)[:H]
    net = mem[0].clone(); sizes = []
    for k, comp in enumerate(good):
        rows = torch.stack([mem[n // H].layers[0].weight[n % H] for n in comp])
        bias = torch.stack([mem[n // H].layers[0].bias[n % H] for n in comp])
        outs = torch.stack([mem[n // H].layers[1].weight[:, n % H] for n in comp])
        net.layers[0].weight.data[k] = rows.mean(0)
        net.layers[0].bias.data[k] = bias.mean(0)
        net.layers[1].weight.data[:, k] = outs.mean(0)
        sizes.append(len(comp))
    net.layers[1].bias.data = mem[0].layers[1].bias.clone()
    return net, torch.tensor(sizes, dtype=torch.float32)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=100000)
    ap.add_argument("--delta", type=float, default=0.25, help="kink band |z|<delta")
    args = ap.parse_args()
    dev = args.device

    print(f"[load] {args.dump}", flush=True)
    ck = torch.load(args.dump, map_location="cpu", weights_only=False)
    dims = ck["dims"]; X = ck["X"]; N = len(X); H = dims[1]
    teacher = load_net(dims, ck["teacher_state"], dev)
    pop = [load_net(dims, s, dev) for s in ck["pop_states"]]
    cons, sizes = consensus_with_sizes(pop, dims)
    cons = cons.to(dev); sizes = sizes.to(dev)

    # align consensus -> teacher; carry signals along the permutation
    t = teacher.clone(); r = cons.clone()
    scale_normalize_(t); scale_normalize_(r)
    A0 = torch.cat([t.layers[0].weight, t.layers[0].bias[:, None]], 1)
    B0 = torch.cat([r.layers[0].weight, r.layers[0].bias[:, None]], 1)
    perm = greedy_perm(A0, B0)
    permute_layer_(r, 0, perm)
    sizes = sizes[torch.tensor(perm, device=dev)]
    err = torch.tensor([
        max((t.layers[0].weight[i] - r.layers[0].weight[i]).abs().max().item(),
            (t.layers[0].bias[i] - r.layers[0].bias[i]).abs().item(),
            (t.layers[1].weight[:, i] - r.layers[1].weight[:, i]).abs().max().item())
        for i in range(H)], device=dev)

    # free training-time signals from the (aligned) consensus pre-activations
    W, b = r.layers[0].weight, r.layers[0].bias
    act = torch.zeros(H, device=dev)
    kink = torch.zeros(H, device=dev)
    for i in range(0, N, args.chunk):
        Z = X[i:i + args.chunk].to(dev) @ W.T + b
        act += (Z > 0).float().sum(0)
        kink += (Z.abs() < args.delta).float().sum(0)
    act /= N; kink /= N
    woutn = r.layers[1].weight.norm(dim=0)
    thr = woutn * kink                     # threshold-identifiability score
    thr2 = woutn.pow(2) * kink

    wk = int(err.argmax())
    print(f"\n[worst unit = pos {wk}]  recon_err={err[wk]:.3e}", flush=True)

    def rank_of(sig):                      # rank 1 = most suspicious (lowest value)
        order = sig.argsort()
        return int((order == wk).nonzero()[0].item()) + 1

    def corr(sig):
        e, s = err, sig
        return (((e - e.mean()) * (s - s.mean())).mean() /
                (e.std() * s.std() + 1e-12)).item()

    sigs = [
        ("activation_rate", act),
        (f"kink_frac |z|<{args.delta}", kink),
        ("||w_out||", woutn),
        ("||w_out||*kink", thr),
        ("||w_out||^2*kink", thr2),
        ("cluster_size", sizes),
    ]
    print(f"\n{'signal (low=suspect)':24s} {'worst rank':>12s} {'value':>11s} "
          f"{'median':>11s} {'corr w/ err':>12s}", flush=True)
    for name, sig in sigs:
        print(f"{name:24s} {rank_of(sig):>7d}/{H:<4d} {sig[wk].item():>11.3e} "
              f"{sig.median().item():>11.3e} {corr(sig):>+12.3f}", flush=True)
    print("\n(rank 1 => that free signal puts the bad unit at the very top of the "
          "suspect list; corr more negative => better global predictor)", flush=True)

    # how CLEANLY does kink_frac separate the bad unit? show the bottom of the
    # suspect list with true errors (are the other low-kink units bad or innocent?)
    order = kink.argsort()
    print(f"\n[bottom-15 by kink_frac]   pos | kink_frac | recon_err | is_bad(>0.1)",
          flush=True)
    for j in range(15):
        p = int(order[j])
        print(f"   #{j+1:2d}: pos {p:4d} | {kink[p]:.4e} | {err[p]:.4e} | "
              f"{'BAD' if err[p] > 0.1 else '.'}", flush=True)
    r1, r2 = int(order[0]), int(order[1])
    print(f"\n[separation] rank1 pos {r1} kink={kink[r1]:.3e} (err {err[r1]:.2e}) vs "
          f"rank2 pos {r2} kink={kink[r2]:.3e} (err {err[r2]:.2e});  "
          f"gap={kink[r2]/kink[r1].clamp_min(1e-12):.2f}x", flush=True)
    # false-alarm check: of the 15 lowest-kink units, how many are actually bad?
    low15 = order[:15]
    nbad = int((err[low15] > 0.1).sum())
    print(f"[false-alarm] of the 15 lowest-kink units, {nbad}/15 are truly bad "
          f"(err>0.1)", flush=True)


if __name__ == "__main__":
    main()
