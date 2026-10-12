"""Layer-1 saturation isolation on a deep tanh MLP, from a committee guess.
For each layer-1 neuron i: walk far along v (in row(W1), v . w_hat_i = 0); all other
layer-1 neurons saturate, so the black-box Jacobian is rank one with row = w_i. Rounds at
growing t; accept the direction common to all M Jacobian row spaces (eigenvalue == M).
Downstream widths are irrelevant (no narrowing needed). Usage:
  python peel/isolate_l1_deep.py <pop_or_net.pt> <guess_net.pt> <neurons: a-b or comma list> <out.pt> [gpu]
Prints the truth-based direction error for SCORING only."""
import sys, os, time, math, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nets import MLP
from tanh_solve import Oracle, measure_jets
from align import sign_align_clone, _preacts

teacher_path, guess_path, spec, out_path = sys.argv[1:5]
dev = f"cuda:{sys.argv[5]}" if len(sys.argv) > 5 else "cuda:0"
torch.manual_seed(0)
ck = torch.load(teacher_path, map_location=dev, weights_only=False); dims = ck["dims"]
teacher = MLP(dims, act="tanh").to(dev); teacher.load_state_dict(ck["teacher_state"])
teacher_f = teacher.clone()                                    # fp32 copy for the (scoring-only) alignment
teacher = teacher.double()
g = torch.load(guess_path, map_location=dev, weights_only=False)
guess = MLP(dims, act="tanh").to(dev); guess.load_state_dict(g["state_dict"])
bb = Oracle(lambda x: teacher(x.double()))
d = dims[0]
# pair guess units to teacher units (scoring only) -- the guess net is the input of the method
with torch.no_grad():
    Xp = torch.rand(8192, d, device=dev) * 2 - 1
    Zg = _preacts(guess, Xp); T_al, tperm, _ = sign_align_clone(teacher_f, Zg, Xp)
W1_true = T_al.layers[0].weight.double()[tperm[0]]            # teacher rows in the GUESS's unit order
guess = guess.double()
neurons = (list(range(int(spec.split("-")[0]), int(spec.split("-")[1]) + 1)) if "-" in spec else [int(a) for a in spec.split(",")])

def dir_err(u, i):
    u = u / u.norm(); wt = W1_true[i] / W1_true[i].norm()
    return float(torch.minimum((u - wt).norm(), (u + wt).norm()))

# ---- row space of W1 from black-box Jacobians at random points (shared by all neurons) ----
t0 = time.time()
X = torch.randn(40, d, dtype=torch.float64, device=dev)
_, J = measure_jets(bb, X, None, h=1e-3)                      # (40, out, d)
S, Vh = torch.linalg.svd(J.reshape(-1, d), full_matrices=False)[1:]
r = int((S > 1e-8 * S[0]).sum()); Vsub = Vh[:r]
print(f"row space of W1: numerical rank {r} (sv[r-1]/sv0 {float(S[r-1]/S[0]):.1e}, sv[r]/sv0 {float(S[min(r, len(S)-1)]/S[0]):.1e}) | {bb.n} queries, {time.time()-t0:.0f}s", flush=True)

def rand_perp(u):
    gg = torch.randn(d, dtype=torch.float64, device=dev)
    gg = Vsub.T @ (Vsub @ gg); gg = gg - (gg @ u) * u / (u @ u)
    return gg / gg.norm()

def centre(x0, u, S_win, n=801, h=1e-3):
    s = torch.linspace(-S_win, S_win, n, dtype=torch.float64, device=dev)
    P = x0[None] + s[:, None] * u[None]
    gq = (bb(P + h * u) - bb(P - h * u)) / (2 * h)
    return float(s[gq.norm(dim=1).argmax()])

results = {}
for i in neurons:
    q0 = bb.n; t_i = time.time()
    w_hat = guess.layers[0].weight[i].detach().clone()
    w_hat = Vsub.T @ (Vsub @ w_hat)
    e0 = dir_err(w_hat, i); hist = [e0]; accepted = 0
    for rnd, (t, M) in enumerate([(30, 6), (100, 6), (300, 6), (1000, 6), (3000, 6)]):
        u = w_hat / w_hat.norm()
        Psum = torch.zeros(d, d, dtype=torch.float64, device=dev); used = 0
        for m in range(M):
            v = rand_perp(u)
            S_win = 3.0 + 0.05 * t if rnd == 0 else 5.0
            s_star = centre(t * v, u, S_win)
            _, Jx = measure_jets(bb, (t * v + s_star * u)[None], None, h=1e-3)
            Sg, Vh2 = torch.linalg.svd(Jx[0], full_matrices=False)[1:]
            if float(Sg[0]) < 1e-9:
                continue
            k = int((Sg > 1e-10 * Sg[0]).sum())
            Psum += Vh2[:k].T @ Vh2[:k]; used += 1
        if used == 0:
            continue
        ev, evec = torch.linalg.eigh(Psum)
        # the subspace COMMON to all M Jacobian row spaces (eigenvalue == used) can be more than
        # one-dimensional: a small-norm neuron that is unsaturated at every probe point sits in it
        # too, and taking the top eigenvector then returned THAT neuron's row (38/133 failures,
        # dir err ~1.4 from round 1). Pick the common-subspace vector nearest the guess instead,
        # and accept only if the guess lies mostly inside that subspace.
        common = evec[:, ev > used - 1e-3]
        if common.shape[1] == 0:
            hist.append(dir_err(w_hat, i)); continue
        proj = common @ (common.T @ u)
        if float(proj.norm()) > 0.9:
            cand = proj / proj.norm()
            w_hat = cand * w_hat.norm(); accepted += 1
        hist.append(dir_err(w_hat, i))
    results[i] = dict(direction=(w_hat / w_hat.norm()).cpu(), err_hist=hist, queries=bb.n - q0, accepted=accepted)
    print(f"neuron {i:3d}: dir err {e0:.1e} -> {hist[-1]:.1e} (rounds accepted {accepted}/5; per round {' '.join(f'{h:.0e}' for h in hist[1:])}) | {bb.n - q0} queries, {time.time()-t_i:.0f}s", flush=True)
torch.save(dict(results=results, dims=dims, guess_path=guess_path, total_queries=bb.n), out_path)
errs = [results[i]["err_hist"][-1] for i in results]
print(f"DONE {len(results)} neurons | final dir err median {sorted(errs)[len(errs)//2]:.1e} max {max(errs):.1e} | within 1e-8: {sum(e < 1e-8 for e in errs)} | total queries {bb.n} | {time.time()-t0:.0f}s", flush=True)
