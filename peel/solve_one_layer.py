"""
solve_one_layer.py -- exact black-box solve of ONE layer from a consensus guess.

Reads:
  * --consensus PATH : a checkpoint holding the consensus guess. Accepts either a
                       `consensus_state` dict (from run.py --fast --stop-outer, or
                       the _consensus.pt artifact) or a plain `state_dict` (the
                       _final.pt). The guess for the target layer is read from it.
  * --layer N        : which hidden layer the consensus is for (0-indexed; L1 = 0).
  * the BLACK BOX    : the teacher. Rebuilt via make_teacher (same arch/seed/epochs
                       as run.py, so it hits the cache) unless --teacher-ckpt is given.

Runs the sigmoid rank certificate (solve_layer) to refine that layer's weights, then
saves {layer, W, b, dims, act} to --out.

For a layer n>0 you must also supply the already-solved earlier layers with repeated
--recovered FILE (each an output of this script), so the net can be sealed at layer
n-1's activations. For the OUTERMOST layer (n=0) none are needed.

  python peel/solve_one_layer.py \
      --consensus recon/mergedbest512_sigmoid__784x128x80x40x32x10__s0_consensus.pt \
      --layer 0 --out recon/solved_L1.pt
"""
import argparse, sys
import torch
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from scipy.optimize import linear_sum_assignment
from nets import MLP
from data import make_teacher
from solve_layer import solve_layer


def load_consensus(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        return {}, ck
    for key in ("consensus_state", "state_dict"):
        if ck.get(key) is not None:
            return ck, ck[key]
    return ck, ck  # the file may itself be a bare state_dict


def align_errors(W, b, teacher, n):
    """Hungarian + sign alignment to the teacher, then return
    (rel_mean, w_mean, w_max, b_mean, b_max): the mean per-neuron RELATIVE weight
    error, and the mean / max ABSOLUTE element-wise error for the weights W and
    (with the same neuron permutation + sign) the biases b."""
    Wt = teacher.layers[n].weight.detach(); nt = Wt.norm(dim=1)
    bt = teacher.layers[n].bias.detach()
    W = W.detach(); b = b.detach()
    Cp = torch.cdist(W, Wt); Cm = torch.cdist(-W, Wt)
    C = torch.minimum(Cp, Cm)
    Cn = C.cpu().numpy(); ri, ci = linear_sum_assignment(Cn)
    rel = float((torch.tensor([Cn[ri[k], ci[k]] for k in range(len(ri))])
                 / nt[ci].cpu()).mean())
    Wal = torch.zeros_like(Wt); bal = torch.zeros_like(bt)   # aligned into teacher order+sign
    for k in range(len(ri)):
        s = -1.0 if float(Cm[ri[k], ci[k]]) < float(Cp[ri[k], ci[k]]) else 1.0
        Wal[ci[k]] = W[ri[k]] * s
        bal[ci[k]] = b[ri[k]] * s
    dW = (Wal - Wt).abs(); dB = (bal - bt).abs()
    return rel, float(dW.mean()), float(dW.max()), float(dB.mean()), float(dB.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus", required=True, help="consensus checkpoint path")
    ap.add_argument("--resume", default="",
                    help="continue from a previously-saved solved-layer file (uses its W,b as "
                         "the init instead of the consensus guess). Same rowspace is recovered, "
                         "so it picks up exactly where that solve stopped.")
    ap.add_argument("--layer", type=int, required=True, help="hidden layer index (L1 = 0)")
    ap.add_argument("--recovered", action="append", default=[],
                    help="already-solved earlier layer file(s), in order 0..n-1 "
                         "(repeat the flag). Required iff --layer > 0.")
    ap.add_argument("--teacher-ckpt", default="",
                    help="load teacher_state from a file (e.g. peel_committee.pt); "
                         "else rebuild the black box via make_teacher.")
    ap.add_argument("--arch", default="", help="comma arch, if not stored in --consensus")
    ap.add_argument("--act", default="sigmoid")
    ap.add_argument("--teacher-seed", type=int, default=0)
    ap.add_argument("--teacher-epochs", type=int, default=25)
    ap.add_argument("--iters", type=int, default=2000,
                    help="max solver iterations (Gauss-Newton/LM steps).")
    ap.add_argument("--log-every", type=int, default=250,
                    help="print werr/|dw| every k iters (iteration 0 always prints).")
    ap.add_argument("--fp32", action="store_true",
                    help="mixed precision: iterate in fp32 (~2-3x faster) then finish "
                         "the last --fp64-finish iters in fp64 to reach the true floor.")
    ap.add_argument("--fp64-finish", type=int, default=300,
                    help="with --fp32, how many final iters to run in fp64 (default 300).")
    ap.add_argument("--retarget-every", type=int, default=0,
                    help="automatic targeted probing: every k iters, flag under-excited "
                         "neurons (teacher-free) and add probes on their knees so the "
                         "certificate can pin them. 0 = off. Adds black-box samples.")
    ap.add_argument("--max-mult", type=int, default=4,
                    help="adaptive probe count: the most-saturated rows get up to this many "
                         "x probes_per_neuron probes (near-threshold rows get 1x).")
    ap.add_argument("--refresh-every", type=int, default=8,
                    help="every k retargets, re-mint ALL targeted rows onto their current "
                         "knees (helps excited-but-slow rows whose knee drifted). 0 = off. "
                         "Costs black-box samples each refresh.")
    ap.add_argument("--bias-steps", type=int, default=1,
                    help="alternating bias refinement: frozen-N Gauss-Newton (LM) steps on bn "
                         "alone against ||K.N||^2 each outer iter (same trick as the weights). "
                         "0 = weights only. Each step costs an extra SVD, so 1 is usually enough.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64); dev = args.device
    ck, cstate = load_consensus(args.consensus)
    dims = ck.get("dims") or ck.get("arch")
    if dims is None and args.arch:
        dims = [int(x) for x in args.arch.split(",")]
    if dims is None:
        raise SystemExit("could not determine arch; pass --arch")
    act = ck.get("act", args.act)
    n = args.layer

    # --- the black box ---
    if args.teacher_ckpt:
        tk = torch.load(args.teacher_ckpt, map_location=dev, weights_only=False)
        dims = tk.get("dims", dims)
        teacher = MLP(dims, act=act).to(dev).double()
        teacher.load_state_dict({k: v.double() for k, v in tk["teacher_state"].items()})
        print(f"[solve] black box loaded from {args.teacher_ckpt}", flush=True)
    else:
        teacher = make_teacher(dims, epochs=args.teacher_epochs, seed=args.teacher_seed,
                               device=dev, act=act).to(dev).double()
    teacher.eval()
    class CountingBB:                 # every input ROW pushed through the black box = 1 sample
        def __init__(self, t): self.t = t; self.n = 0
        @torch.no_grad()
        def __call__(self, x): self.n += int(x.shape[0]); return self.t(x)
    BB = CountingBB(teacher)

    # --- init: resume from a saved solve, else the consensus guess ---
    if args.resume:
        rk = torch.load(args.resume, map_location=dev, weights_only=False)
        Wn_g = rk["W"].to(dev).double(); bn_g = rk["b"].to(dev).double()
        print(f"[solve] RESUMING layer {n} from {args.resume} "
              f"(saved werr {rk.get('werr', float('nan'))*100:.4f}%)", flush=True)
    else:
        Wn_g = cstate[f"layers.{n}.weight"].to(dev).double()
        bn_g = cstate[f"layers.{n}.bias"].to(dev).double()

    # --- recovered earlier layers (to seal, when n>0) ---
    W_early, b_early = [], []
    for rp in args.recovered:
        rk = torch.load(rp, map_location=dev, weights_only=False)
        W_early.append(rk["W"].to(dev).double()); b_early.append(rk["b"].to(dev).double())
    if len(W_early) != n:
        raise SystemExit(f"--layer {n} needs {n} --recovered file(s); got {len(W_early)}")

    def score(W, b):
        rel, wm, wmax, bm, bmax = align_errors(W, b, teacher, n)
        return (f"werr {rel*100:.4f}%  W[mean {wm:.3e} max {wmax:.3e}]  "
                f"b[mean {bm:.3e} max {bmax:.3e}]")
    print(f"[solve] layer n={n}: {dims[n]}->{dims[n+1]}, next {dims[n+1]}->{dims[n+2]} "
          f"| consensus guess {score(Wn_g, bn_g)}", flush=True)
    Wn, bn, info = solve_layer(BB, W_early, b_early, Wn_g, bn_g, dims, n,
                               iters=args.iters, dev=dev, score=score,
                               log_every=args.log_every,
                               fp32=args.fp32, fp64_finish=args.fp64_finish,
                               bias_steps=args.bias_steps, retarget_every=args.retarget_every,
                               max_mult=args.max_mult, refresh_every=args.refresh_every)
    print(f"[solve] DONE layer {n}: final {score(Wn, bn)}", flush=True)
    print(f"[solve] black-box samples used: {BB.n:,} (all at setup: rowspace + "
          f"certificate probes; the LM iterations query nothing)", flush=True)
    rel, wm, wmax, bm, bmax = align_errors(Wn, bn, teacher, n)
    torch.save({"layer": n, "dims": dims, "act": act,
                "W": Wn.detach().cpu(), "b": bn.detach().cpu(),
                "werr": rel, "w_mean": wm, "w_max": wmax,
                "b_mean": bm, "b_max": bmax, "samples": BB.n}, args.out)
    print(f"[solve] saved solved layer {n} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
