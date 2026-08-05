"""Diagnostic: per-member loss + max_eps spread from a --fast dump, and how the
consensus changes when high-loss members are gated out. Read-only."""
import argparse
import torch
from nets import MLP
from method import build_consensus, l1_on
from align import param_errors, scale_normalize_, greedy_perm, permute_layer_


def load_net(dims, state, dev):
    m = MLP(dims).to(dev); m.load_state_dict(state); return m


@torch.no_grad()
def inter_member_maxeps(pop):
    """Align every member to member 0 and report max pairwise param distance."""
    anchor = pop[0].clone(); scale_normalize_(anchor)
    outs = []
    for i in range(1, len(pop)):
        m = pop[i].clone(); scale_normalize_(m)
        for l in range(len(anchor.layers) - 1):
            A = torch.cat([anchor.layers[l].weight, anchor.layers[l].bias[:, None]], 1)
            B = torch.cat([m.layers[l].weight, m.layers[l].bias[:, None]], 1)
            permute_layer_(m, l, greedy_perm(A, B))
        d = max((pa - pb).abs().max().item()
                for la, lb in zip(anchor.layers, m.layers)
                for pa, pb in zip(la.parameters(), lb.parameters()))
        outs.append(d)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--loss-samples", type=int, default=100000)
    args = ap.parse_args()
    dev = args.device

    print(f"[load] {args.dump} ...", flush=True)
    ck = torch.load(args.dump, map_location="cpu", weights_only=False)
    dims = ck["dims"]; X, Y = ck["X"], ck["Y"]; N = len(X)
    teacher = load_net(dims, ck["teacher_state"], dev)
    pop = [load_net(dims, s, dev) for s in ck["pop_states"]]
    print(f"[load] dims={dims} iter={ck['iter']} queries={N} members={len(pop)}",
          flush=True)

    g = torch.Generator().manual_seed(0)
    li = torch.randperm(N, generator=g)[:min(args.loss_samples, N)]
    Xl, Yl = X[li], Y[li]
    losses = l1_on(pop, Xl, Yl)
    bl = min(losses)

    print("\n[members] per-member train_L1 and max_eps (vs teacher):", flush=True)
    for i in range(len(pop)):
        me = param_errors(pop[i], teacher)["max_eps"]
        print(f"   member {i}: L1={losses[i]:.3e}  ({losses[i]/bl:5.1f}x best)  "
              f"max_eps={me:.3e}", flush=True)

    print("\n[scatter] inter-member max param distance (aligned to member 0):",
          flush=True)
    sc = inter_member_maxeps(pop)
    print("   " + "  ".join(f"m0-m{i+1}={d:.3e}" for i, d in enumerate(sc)),
          flush=True)

    print("\n[consensus] max_eps under different gating:", flush=True)
    variants = [
        ("ungated (what --fast uses)", dict(quorum_ratio=0.625)),
        ("gated k=10", dict(losses=losses, quorum_ratio=0.625, gate_kappa=10.0)),
        ("gated k=3",  dict(losses=losses, quorum_ratio=0.625, gate_kappa=3.0)),
        ("gated k=1.5", dict(losses=losses, quorum_ratio=0.625, gate_kappa=1.5)),
    ]
    for tag, kw in variants:
        c = build_consensus(pop, dims, **kw)
        if c is None:
            print(f"   {tag:28s}: n/a", flush=True)
        else:
            print(f"   {tag:28s}: max_eps={param_errors(c, teacher)['max_eps']:.3e}",
                  flush=True)


if __name__ == "__main__":
    main()
