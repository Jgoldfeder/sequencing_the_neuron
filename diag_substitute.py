"""Decisive recoverability test for the worst unit: substitute the TEACHER's
actual row/bias/output-col for that unit into the (gauge-matched) consensus and
measure the query loss. If loss drops, the unit is recoverable and the solvers
merely missed it; if loss does NOT drop, the query set cannot distinguish the
teacher's row from the consensus's wrong one -> genuinely non-identifiable on
these queries. Teacher weights are used only to construct the test, not to solve."""
import argparse
import torch
from nets import MLP
from method import build_consensus
from align import scale_normalize_, greedy_perm, permute_layer_


def load_net(dims, state, dev):
    m = MLP(dims).to(dev); m.load_state_dict(state); return m


@torch.no_grad()
def l1(net, X, Y, dev, bs=100000):
    tot, n = 0.0, 0
    for i in range(0, len(X), bs):
        xb, yb = X[i:i+bs].to(dev), Y[i:i+bs].to(dev)
        tot += (net(xb) - yb).abs().sum().item(); n += yb.numel()
    return tot / n


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval", type=int, default=300000)
    args = ap.parse_args()
    dev = args.device

    print(f"[load] {args.dump}", flush=True)
    ck = torch.load(args.dump, map_location="cpu", weights_only=False)
    dims = ck["dims"]; X, Y = ck["X"], ck["Y"]; N = len(X)
    Xe, Ye = X[:args.eval], Y[:args.eval]
    teacher = load_net(dims, ck["teacher_state"], dev)
    pop = [load_net(dims, s, dev) for s in ck["pop_states"]]
    cons = build_consensus(pop, dims, quorum_ratio=0.625).to(dev)

    # gauge-match: scale-normalize both, permute consensus onto teacher
    t = teacher.clone(); r = cons.clone()
    scale_normalize_(t); scale_normalize_(r)
    A0 = torch.cat([t.layers[0].weight, t.layers[0].bias[:, None]], 1)
    B0 = torch.cat([r.layers[0].weight, r.layers[0].bias[:, None]], 1)
    permute_layer_(r, 0, greedy_perm(A0, B0))
    H = dims[1]
    err = torch.tensor([
        max((t.layers[0].weight[i] - r.layers[0].weight[i]).abs().max().item(),
            (t.layers[0].bias[i] - r.layers[0].bias[i]).abs().item(),
            (t.layers[1].weight[:, i] - r.layers[1].weight[:, i]).abs().max().item())
        for i in range(H)])
    wk = int(err.argmax())

    l_cons = l1(cons, Xe, Ye, dev)
    l_r = l1(r, Xe, Ye, dev)                 # sanity: == l_cons (norm preserves fn)
    l_teacher = l1(teacher, Xe, Ye, dev)     # == 0 (Y is teacher(X))

    # substitute teacher's unit wk into the gauge-matched consensus
    r_fix = r.clone()
    r_fix.layers[0].weight.data[wk] = t.layers[0].weight[wk]
    r_fix.layers[0].bias.data[wk] = t.layers[0].bias[wk]
    r_fix.layers[1].weight.data[:, wk] = t.layers[1].weight[:, wk]
    l_fix = l1(r_fix, Xe, Ye, dev)

    print(f"\n[unit {wk}] recon_err={err[wk]:.3e}", flush=True)
    print(f"  loss(consensus)            = {l_cons:.4e}", flush=True)
    print(f"  loss(consensus, normed)    = {l_r:.4e}   (sanity ~= consensus)", flush=True)
    print(f"  loss(teacher)              = {l_teacher:.4e}   (== 0, Y=teacher(X))", flush=True)
    print(f"  loss(consensus w/ TRUE u{wk}) = {l_fix:.4e}", flush=True)
    drop = l_r - l_fix
    print(f"\n[verdict] substituting the teacher's true unit {wk} changes loss by "
          f"{drop:+.3e}", flush=True)
    if drop > 1e-6:
        print("  -> loss DROPS: the row IS recoverable from these queries; "
              "the solvers missed it.", flush=True)
    else:
        print("  -> loss does NOT drop: the queries cannot distinguish the true "
              "row from the consensus's -> NON-IDENTIFIABLE on this query set.",
              flush=True)


if __name__ == "__main__":
    main()
