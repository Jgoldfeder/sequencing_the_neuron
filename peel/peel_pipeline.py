"""
peel_pipeline.py -- integrated layer-by-layer black-box extraction.

Wires the four existing pieces into one peeling loop. For each weight matrix n
(outer -> inner):

  1. SEAL the teacher at layer n-1's activations using the layers already
     recovered EXACTLY (solve_layer.build_seal). n=0 -> the raw teacher.
  2. POPULATION + CONSENSUS + FAST on the SEALED subnet dims[n:] to get an
     approximate guess for layer n (method.reconstruct with --fast-style
     early-stop, build_consensus, solver_polish_). Queries are generated
     unbounded then squashed into the activation domain via input_transform.
  3. EXACT refine of layer n from that guess:
        sigmoid hidden layer -> solve_layer (rank certificate)
        relu    hidden layer -> verify_layer1.extract_neuron_exact (kink)
        final linear layer   -> closed-form least squares
  4. PEEL: append the exact (W_n, b_n) and move inward.

Because each layer is made exact before it is used to seal the next, the seal
stays exact and the ingredient-quality floor that limits a single certificate
solve does not accumulate.

The population stage SCORES its guess against the true sub-weights (oracle,
logging only, exactly as run.py scores with param_errors); the SOLVE itself
only ever queries the sealed black box. Run it yourself, e.g.:

  python peel_pipeline.py --arch 784,128,80,40,32,10 --act sigmoid \
      --variant mergedbest512 --pop 8 --outer 60 --q 8000 --window 60 \
      --combine --exact-iters 2000 --device cuda
"""
import argparse, os, sys, time, tempfile
from dataclasses import replace
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from scipy.optimize import linear_sum_assignment
from method import Cfg, reconstruct, build_consensus, solver_polish_
from nets import MLP
from data import make_teacher
from solve_layer import solve_layer, build_seal
import verify_layer1 as vl1


# ---- activation-domain squashers (unbounded query -> valid activation h) ----
def squash_fn(act):
    if act == "sigmoid":
        return lambda z: torch.sigmoid(z).clamp(2e-2, 1 - 2e-2)     # -> (0,1)
    return lambda z: F.softplus(z) + 1e-3                            # relu: -> (0, inf)


# ---- sealed teacher: true sub-weights for SCORING, sealed query for SOLVING ----
class SealedTeacher(MLP):
    """An MLP over dims[n:] loaded with the teacher's TRUE layers n.. (so
    param_errors/agreement can score the guess -- oracle, logging only), but with
    forward() OVERRIDDEN to the sealed black-box query BB_n(h)=teacher(x_of_h(h)),
    which uses only the recovered early layers + the real black box."""
    def __init__(self, sub_dims, act, seal_fn):
        super().__init__(sub_dims, act=act)
        object.__setattr__(self, "_seal_fn", seal_fn)
    def forward(self, x):
        return self._seal_fn(x)


def build_true_subnet_state(teacher, dims, n, act, device):
    sub = MLP(dims[n:], act=act).to(device)
    with torch.no_grad():
        for k in range(len(sub.layers)):
            sub.layers[k].weight.copy_(teacher.layers[n + k].weight)
            sub.layers[k].bias.copy_(teacher.layers[n + k].bias)
    return sub.state_dict()


# ---- oracle scorer (LOGGING ONLY): mean per-neuron rel weight error, Hungarian+sign ----
def hungarian_werr(Wn, bn, teacher, n):
    Wn = Wn.detach()
    if not torch.isfinite(Wn).all():
        return float("nan")
    Wt = teacher.layers[n].weight.detach(); nt = Wt.norm(dim=1)
    Cp = torch.cdist(Wn, Wt); Cm = torch.cdist(-Wn, Wt)
    C = torch.minimum(Cp, Cm).cpu().numpy(); ri, ci = linear_sum_assignment(C)
    return float((torch.tensor([C[ri[i], ci[i]] for i in range(len(ri))]) / nt[ci].cpu()).mean())


# ============================ STAGE 2: population/consensus/fast ============================
def fast_solve(sealed_teacher, sub_dims, cfg, device, seed, eval_pts, input_transform):
    """population + consensus + staged solve on the sealed subnet -> guess MLP(sub_dims)."""
    tmp = tempfile.mktemp(suffix="_peelfast.pt")
    cfg = replace(cfg, stop_on_consensus=True, dump_path=tmp)
    best, _, _ = reconstruct(sealed_teacher, sub_dims, cfg, device, eval_pts, seed,
                             input_transform=input_transform)
    if os.path.exists(tmp):
        ck = torch.load(tmp, map_location="cpu", weights_only=False)
        Xf, Yf = ck["X"], ck["Y"]
        pop = []
        for s in ck["pop_states"]:
            m = MLP(sub_dims, act=cfg.act).to(device); m.load_state_dict(s); pop.append(m)
        cons = build_consensus(pop, sub_dims, quorum_ratio=cfg.cluster_quorum)
        os.remove(tmp)
        if cons is not None:
            cons = cons.to(device)
            solver_polish_(cons, Xf, Yf, mse_steps=40, mae_steps=40, tag=" peelfast")
            return cons
        print("    [fast] no consensus; falling back to reconstruct's best", flush=True)
    return best


# ============================ STAGE 3: exact per-layer solvers ============================
def exact_sigmoid(teacher, W_early, b_early, Wn_g, bn_g, dims, n, iters, device):
    return solve_layer(teacher, W_early, b_early, Wn_g, bn_g, dims, n,
                       iters=iters, dev=device,
                       score=lambda W, b: f"werr {hungarian_werr(W, b, teacher, n)*100:.4f}%")

def exact_relu(teacher, W_early, b_early, Wn_g, bn_g, dims, n, act, device):
    """Kink extraction per neuron on the sealed oracle (verify_layer1). NOTE: recovers
    each neuron's hyperplane (unit direction + bias); the per-neuron POSITIVE SCALE is a
    genuine ReLU gauge freedom the kink method leaves open (see relu_vs_sigmoid note)."""
    BBh, _ = build_seal(teacher, W_early, b_early, n, act)
    class Orc:
        def __init__(s): s.n = 0
        @torch.no_grad()
        def __call__(s, X): s.n += int(X.shape[0]); return BBh(X)
    orc = Orc(); d_n, d_prev = dims[n + 1], dims[n]
    g = torch.Generator(device=device).manual_seed(7); rows = []; biases = []
    for j in range(d_n):
        base = F.softplus(torch.randn(d_prev, generator=g, device=device)) + 1e-3
        res = vl1.extract_neuron_exact(orc, Wn_g[j], float(bn_g[j]), base)
        if isinstance(res, dict) and res.get("ok"):
            rows.append(res["w"].to(device).double()); biases.append(res["b"])
        else:                                    # extractor rejected -> keep the guess row
            rows.append(Wn_g[j] / Wn_g[j].norm()); biases.append(float(bn_g[j]))
    return torch.stack(rows), torch.tensor(biases, device=device)

def exact_last_linear(teacher, W_early, b_early, dims, n, act, device):
    """Final layer feeds the LINEAR output: seal at a_{n-1}, then W_n,b_n by exact LSQ."""
    BBh, _ = build_seal(teacher, W_early, b_early, n, act)
    g = torch.Generator(device=device).manual_seed(123)
    Z = torch.randn(4000, dims[n], generator=g, device=device); H = squash_fn(act)(Z)
    with torch.no_grad(): Y = BBh(H)
    A = torch.cat([H, torch.ones(H.shape[0], 1, device=device)], 1)
    sol = torch.linalg.lstsq(A, Y).solution
    return sol[:-1].t().contiguous(), sol[-1].contiguous()


# ============================ THE PEEL LOOP ============================
def peel(teacher, dims, cfg, device, seed, act, exact_iters):
    n_layers = len(dims) - 1                       # number of weight matrices
    recovered = []                                 # exact [(W_i, b_i)], i < n
    for n in range(n_layers):
        d_prev, d_n = dims[n], dims[n + 1]
        is_last = (n == n_layers - 1)              # feeds the linear output
        print(f"\n===== PEEL layer n={n}: {d_prev}->{d_n}"
              f"{' (linear output)' if is_last else f', next {d_n}->{dims[n+2]}'} =====", flush=True)
        W_early = [W for W, _ in recovered]; b_early = [b for _, b in recovered]

        # ---- STAGE 1: seal ----
        if n == 0:
            sealed, transform, eval_pts = teacher, None, torch.randn(512, dims[0], device=device)
        else:
            seal_fn, _ = build_seal(teacher, W_early, b_early, n, act)
            sealed = SealedTeacher(dims[n:], act, seal_fn).to(device)
            sealed.load_state_dict(build_true_subnet_state(teacher, dims, n, act, device))
            transform = squash_fn(act)
            eval_pts = transform(torch.randn(512, dims[n], device=device))

        # ---- STAGE 2: population + consensus + fast (skip for the trivial linear layer) ----
        if not is_last:
            guess = fast_solve(sealed, dims[n:], cfg, device, seed, eval_pts, transform)
            Wn_g = guess.layers[0].weight.detach().double()
            bn_g = guess.layers[0].bias.detach().double()
            print(f"  [guess] layer {n} werr {hungarian_werr(Wn_g, bn_g, teacher, n)*100:.3f}%", flush=True)

        # ---- STAGE 3: exact refine (guarded: a diverging layer falls back to its
        #      guess and the peel continues rather than crashing the whole run) ----
        if is_last:
            Wn, bn = exact_last_linear(teacher, W_early, b_early, dims, n, act, device)
        else:
            try:
                if act == "sigmoid":
                    Wn, bn, _ = exact_sigmoid(teacher, W_early, b_early, Wn_g, bn_g, dims, n, exact_iters, device)
                else:
                    Wn, bn = exact_relu(teacher, W_early, b_early, Wn_g, bn_g, dims, n, act, device)
            except Exception as e:
                print(f"  [EXACT] layer {n} solve FAILED ({type(e).__name__}: {e}); "
                      f"falling back to the guess and continuing", flush=True)
                Wn, bn = Wn_g, bn_g

        # ---- STAGE 4: peel ----
        recovered.append((Wn.detach(), bn.detach()))
        print(f"  [EXACT] layer {n} werr {hungarian_werr(Wn, bn, teacher, n)*100:.4f}%", flush=True)
    return recovered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="784,128,80,40,32,10")
    ap.add_argument("--act", default="sigmoid", choices=["sigmoid", "leaky_relu"])
    ap.add_argument("--variant", default="mergedbest512")
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--outer", type=int, default=60)
    ap.add_argument("--q", type=int, default=8000)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--exact-iters", type=int, default=2000)
    ap.add_argument("--teacher-epochs", type=int, default=10)
    ap.add_argument("--teacher-ckpt", default="",
                    help="load teacher_state from a checkpoint (e.g. peel_committee.pt) "
                         "instead of make_teacher; uses its 'dims' if present")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)         # whole pipeline runs in float64
    from run import VARIANTS
    dims = [int(x) for x in args.arch.split(",")]
    device = args.device
    if args.teacher_ckpt:
        ck = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
        dims = ck.get("dims", dims)
        teacher = MLP(dims, act=args.act).to(device).double()
        teacher.load_state_dict({k: v.double() for k, v in ck["teacher_state"].items()})
        print(f"[peel] loaded teacher from {args.teacher_ckpt}", flush=True)
    else:
        teacher = make_teacher(dims, epochs=args.teacher_epochs, seed=args.seed,
                               device=device, act=args.act).to(device).double()
    teacher.eval()

    vd = dict(VARIANTS[args.variant])
    if args.window is not None: vd["window"] = args.window
    cfg = Cfg(p=args.pop, q=args.q, outer=args.outer, act=args.act, **vd)
    cfg = replace(cfg, combine=args.combine, log_every=args.log_every)

    t0 = time.time()
    print(f"[peel] arch {dims} act {args.act} variant {args.variant} "
          f"pop {args.pop} outer {args.outer} q {args.q}", flush=True)
    recovered = peel(teacher, dims, cfg, device, args.seed, args.act, args.exact_iters)

    werrs = [hungarian_werr(W, b, teacher, i) for i, (W, b) in enumerate(recovered)]
    print(f"\n[peel] DONE in {time.time()-t0:.0f}s. per-layer werr: "
          + "  ".join(f"L{i}:{e*100:.3f}%" for i, e in enumerate(werrs)), flush=True)
    if args.out:
        torch.save({"dims": dims, "act": args.act,
                    "recovered": [(W.cpu(), b.cpu()) for W, b in recovered],
                    "werrs": werrs}, args.out)
        print(f"[peel] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
