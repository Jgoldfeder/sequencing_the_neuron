"""Why is one unit unrecoverable? Test the under-excitation hypothesis: align the
consensus to the teacher, get each hidden unit's reconstruction error, and
correlate it with that teacher unit's ACTIVATION RATE on the query set
(fraction of queries with pre-activation z>0, i.e. in LeakyReLU's steep region).
If the worst-recovered unit is the least-activated, its input row is
under-excited by these queries and no solver can recover it. Read-only."""
import argparse
import torch
from nets import MLP
from method import build_consensus
from align import scale_normalize_, greedy_perm, permute_layer_


def load_net(dims, state, dev):
    m = MLP(dims).to(dev); m.load_state_dict(state); return m


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=100000)
    args = ap.parse_args()
    dev = args.device

    print(f"[load] {args.dump}", flush=True)
    ck = torch.load(args.dump, map_location="cpu", weights_only=False)
    dims = ck["dims"]; X = ck["X"]; N = len(X)
    teacher = load_net(dims, ck["teacher_state"], dev)
    pop = [load_net(dims, s, dev) for s in ck["pop_states"]]
    cons = build_consensus(pop, dims, quorum_ratio=0.625).to(dev)

    # align consensus -> teacher (hidden layer); position i == teacher unit i
    t = teacher.clone(); r = cons.clone()
    scale_normalize_(t); scale_normalize_(r)
    A0 = torch.cat([t.layers[0].weight, t.layers[0].bias[:, None]], 1)
    B0 = torch.cat([r.layers[0].weight, r.layers[0].bias[:, None]], 1)
    perm = greedy_perm(A0, B0)
    permute_layer_(r, 0, perm)
    H = dims[1]
    err = torch.tensor([
        max((t.layers[0].weight[i] - r.layers[0].weight[i]).abs().max().item(),
            (t.layers[0].bias[i] - r.layers[0].bias[i]).abs().item(),
            (t.layers[1].weight[:, i] - r.layers[1].weight[:, i]).abs().max().item())
        for i in range(H)])

    # teacher activation rate on the queries (raw teacher, chunked)
    Wt = teacher.layers[0].weight        # (H, D)
    bt = teacher.layers[0].bias
    act = torch.zeros(H, device=dev)
    zmean = torch.zeros(H, device=dev)
    for i in range(0, N, args.chunk):
        Xc = X[i:i + args.chunk].to(dev)
        Z = Xc @ Wt.T + bt               # (Nc, H)
        act += (Z > 0).float().sum(0)
        zmean += Z.sum(0)
    act /= N; zmean /= N
    woutnorm = teacher.layers[1].weight.norm(dim=0).cpu()  # teacher influence

    act = act.cpu(); zmean = zmean.cpu()
    # correlation between recon error and activation rate
    e = err; a = act
    corr = (((e - e.mean()) * (a - a.mean())).mean() /
            (e.std() * a.std() + 1e-12)).item()

    wk = int(err.argmax())
    print(f"\n[worst unit {wk}] recon_err={err[wk]:.3e}  "
          f"activation_rate={act[wk]:.4f}  mean_z={zmean[wk]:+.3e}  "
          f"||w_out||={woutnorm[wk]:.3e}", flush=True)
    print(f"[population] median activation_rate={act.median():.4f}  "
          f"median recon_err={err.median():.3e}", flush=True)
    print(f"[corr] recon_err vs activation_rate: r={corr:+.3f}  "
          f"(negative => less-active units are worse-recovered)", flush=True)

    order = err.argsort(descending=True)[:8]
    print("\n[8 worst-recovered units]  err | act_rate | mean_z | ||w_out||",
          flush=True)
    for i in order.tolist():
        print(f"   unit {i:4d}: {err[i]:.3e} | {act[i]:.4f} | {zmean[i]:+.3e} "
              f"| {woutnorm[i]:.3e}", flush=True)
    # how rare is the worst unit's activation among all units?
    pct = (act < act[wk]).float().mean().item() * 100
    print(f"\n[rank] worst unit's activation_rate is at the {pct:.1f}th percentile "
          f"of all {H} units", flush=True)


if __name__ == "__main__":
    main()
