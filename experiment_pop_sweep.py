"""Population-size sweep: is consensus accuracy variance-limited (improves with
more members) or bias-limited (plateaus)?

Controlled design: ONE fixed query set, M independent members trained on it, then
consensus_neuron_stats over growing P and several quorum ratios. If mean_eps on
consensus neurons keeps dropping with P -> variance-limited (more members help,
can trade for queries). If it plateaus -> bias-limited (query-set identifiability
floor; more members can't help). Queries here are fixed random (controls the P
effect); the real method uses adaptive disagreement queries.
"""
import argparse
import time
import torch
from data import make_teacher
from nets import MLP
from method import consensus_neuron_stats


def train_member(dims, X, Y, dev, epochs, lr, batch, seed):
    torch.manual_seed(seed)
    net = MLP(dims).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(X)
    decay = {int(0.6 * epochs), int(0.85 * epochs)}
    for ep in range(epochs):
        if ep in decay:
            for g in opt.param_groups:
                g["lr"] /= 10
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = ((net(X[idx]) - Y[idx]) ** 2).mean()   # MSE: descends fast
            loss.backward()
            opt.step()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="3072,512,100")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--queries", type=int, default=60000)
    ap.add_argument("--members", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--qstd", type=float, default=1.0, help="query gaussian std")
    args = ap.parse_args()
    dev = args.device
    dims = [int(x) for x in args.arch.split(",")]

    teacher = make_teacher(dims, epochs=25, seed=0, device=dev, verbose=False)
    torch.manual_seed(0)
    X = torch.randn(args.queries, dims[0], device=dev) * args.qstd
    with torch.no_grad():
        Y = teacher(X)
    print(f"[setup] arch={dims} queries={args.queries} members={args.members} "
          f"epochs={args.epochs} device={dev}", flush=True)

    members = []
    t0 = time.time()
    for s in range(args.members):
        members.append(train_member(dims, X, Y, dev, args.epochs, args.lr,
                                     args.batch, seed=1000 + s))
        if (s + 1) % 8 == 0:
            with torch.no_grad():
                l = (members[-1](X) - Y).abs().mean().item()
            print(f"  trained {s+1}/{args.members} members  "
                  f"(last L1={l:.2e}, {time.time()-t0:.0f}s)", flush=True)

    Pvals = [p for p in (8, 16, 32, 48, 64) if p <= args.members]
    QRs = [0.25, 0.5, 0.625]
    print(f"\n[sweep] P vs quorum-ratio  (n_consensus / {dims[1]}  |  "
          f"mean_eps  |  max_eps)", flush=True)
    header = "  P\\qr | " + " | ".join(f"{qr:>22.3f}" for qr in QRs)
    print(header, flush=True)
    for P in Pvals:
        cells = []
        for qr in QRs:
            s = consensus_neuron_stats(members[:P], teacher, dims, quorum_ratio=qr)
            if s is None or s["max_eps"] is None:
                cells.append(f"{s['n_consensus'] if s else 0:>4d}/{dims[1]}  "
                             f"{'--':>8} {'--':>8}")
            else:
                cells.append(f"{s['n_consensus']:>4d}/{dims[1]}  "
                             f"{s['mean_eps']:.2e} {s['max_eps']:.2e}")
        print(f"  {P:>4d} | " + " | ".join(cells), flush=True)
    print("\n[read] down a column (fixed quorum ratio): does mean_eps drop as P "
          "grows? -> variance-limited. flat -> bias-limited.", flush=True)


if __name__ == "__main__":
    main()
