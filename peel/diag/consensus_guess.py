"""Trunk guess for the post-committee solver: the quorum-CONSENSUS net instead of the best member.

Why: a single member can carry a few units that are wrong by O(1) (9L demo: three units of W6/W7/W8 at
0.45-0.72) while the 5-of-8 quorum has them right (5e-3). The jet solver is local, so such units never
recover; starting from the consensus trunk they are simply not there.

How: method._partial_consensus builds the consensus in the most-agreeing member's frame (consensus rows =
quorum mean, stragglers = that member's own rows); it is then aligned (function-preserving, sign-aware)
into the BEST member's unit order so that everything downstream that is indexed by unit (layer-1
isolation shards, gap-fill pairing, orphan complement) stays consistent. The whole aligned net is used
(its own layer 1 + trunk): mixing the best member's layer 1 with the consensus trunk is wrong, because
the consensus is in a sign-canonical gauge and its layer-1 magnitudes differ (W2 came out at 0.86).
Only gate: every layer-1 row of the aligned consensus must match the best member's row (|cos| >= 0.99);
otherwise the best member is kept. Teacher-based numbers are printed for scoring only.

    from peel.diag.consensus_guess import consensus_guess
    sd, note = consensus_guess(pop_ckpt, best_state_dict)      # sd is None -> keep the best member
"""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from nets import MLP
from method import _partial_consensus, align_member_to
from align import param_errors


@torch.no_grad()
def consensus_guess(pop_ckpt, best_state, eps=0.02, quorum=0.625, act="tanh", min_cos=0.99, max_mae_ratio=3.0, bb=None, verbose=True):
    """pop_ckpt: dict with dims / pop_states / teacher_state (a _pop__ snapshot). best_state: the best member's
    state_dict (the final checkpoint). bb: black-box callable X -> Y for the sanity gate (default: the teacher in
    pop_ckpt, i.e. the same oracle the run queried; costs 4096 queries). Returns (state_dict or None, note)."""
    dims = pop_ckpt["dims"]
    pop = []
    for sd in pop_ckpt["pop_states"]:
        m = MLP(dims, act=act); m.load_state_dict(sd); m.eval(); pop.append(m)
    best = MLP(dims, act=act); best.load_state_dict(best_state); best.eval()
    if len(pop) < 2:
        return None, "consensus guess: population too small"
    if bb is None and "teacher_state" in pop_ckpt:
        _T = MLP(dims, act=act); _T.load_state_dict(pop_ckpt["teacher_state"]); _T.eval(); bb = _T
    cnet, masks = _partial_consensus(pop, dims, eps, quorum)
    ref = best.clone(); align_member_to(ref, cnet)            # cnet -> best member's unit order (ref = scratch)
    cos = torch.nn.functional.cosine_similarity(best.layers[0].weight, cnet.layers[0].weight, dim=1).abs()
    n_bad = int((cos < min_cos).sum())
    note = (f"consensus guess: quorum {quorum:.3f} eps {eps:g} | units in consensus per hidden layer "
            f"{[int(mk.sum()) for mk in masks]} | layer-1 order check |cos| min {float(cos.min()):.4f} ({n_bad} rows < {min_cos})")
    if "teacher_state" in pop_ckpt:                           # scoring only
        T = MLP(dims, act=act); T.load_state_dict(pop_ckpt["teacher_state"]); T.eval()
        pb = param_errors(best, T)["max_eps_per_matrix"]; pc = param_errors(cnet, T)["max_eps_per_matrix"]
        note += ("\n  best member per-matrix max (teacher, scoring only): " + " ".join(f"{float(x):.1e}" for x in pb)
                 + "\n  consensus   per-matrix max (teacher, scoring only): " + " ".join(f"{float(x):.1e}" for x in pc))
    if n_bad:
        note += "\n  -> order check FAILED: keeping the best member"
        if verbose: print(note, flush=True)
        return None, note
    # black-box sanity gate (4096 fresh queries): a consensus whose non-quorum rows come from a poor reference
    # member can be a worse FUNCTION than the best member; allow the usual layer-1-magnitude gap (9L: 1.7x) but
    # not a broken net
    if bb is not None:
        g = torch.Generator().manual_seed(0); X = torch.randn(4096, dims[0], generator=g)
        Y = bb(X); mb = float((best(X) - Y).abs().mean()); mc = float((cnet(X) - Y).abs().mean())
        note += f"\n  black-box MAE on 4096 fresh queries: best member {mb:.3e} consensus {mc:.3e}"
        if mc > max_mae_ratio * mb:
            note += f"\n  -> consensus is a much worse function (> {max_mae_ratio:g}x): keeping the best member"
            if verbose: print(note, flush=True)
            return None, note
    note += "\n  -> using the aligned consensus net as the solver's starting net"
    if verbose: print(note, flush=True)
    return {k: v.detach().clone() for k, v in cnet.state_dict().items()}, note


if __name__ == "__main__":
    pop_path, final_path, out_path = sys.argv[1:4]
    pop = torch.load(pop_path, map_location="cpu", weights_only=False)
    fin = torch.load(final_path, map_location="cpu", weights_only=False)
    sd, note = consensus_guess(pop, fin["state_dict"])
    torch.save({"dims": pop["dims"], "act": "tanh", "teacher_state": pop["teacher_state"],
                "state_dict": sd if sd is not None else fin["state_dict"],
                "note": ("aligned quorum consensus" if sd is not None else "best member") + f" of {final_path}"}, out_path)
    print(f"wrote {out_path} ({'consensus' if sd is not None else 'best member'})")
