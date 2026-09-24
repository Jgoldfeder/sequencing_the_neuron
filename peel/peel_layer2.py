"""SIGMOID peel test: freeze the SOLVED first layer (from our jet solver, NOT any ReLU
kink extractor) shared across the current committee, then run the extraction's own
activation-agnostic disagreement sampler + committee training and watch whether LAYER 2
reaches consensus.

Config via env: L1SRC={solved|true|consensus}  ITERS=N  Q=n  MEASURE_EVERY=k
Reuses method.gen_queries / build_consensus / _install_freeze / l1_on (all sigmoid-capable).
"""
import os, sys, math, time, torch, numpy as np
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from scipy.optimize import linear_sum_assignment
from nets import MLP
from method import (Cfg, gen_queries, build_consensus, _install_freeze, l1_on,
                    param_errors, consensus_neuron_stats, agreement, disagreement)
TARGET = os.environ.get("TARGET", "out")        # "out" = orig sampler; "L2"/"corner" = target layer-2
FREEZE_L1_BIAS = int(os.environ.get("FREEZE_L1_BIAS", "1"))   # 0 = freeze L1 weights only, leave L1 bias trainable
LOSS_SCALE = float(os.environ.get("LOSS_SCALE", "1"))         # multiply the training loss (Adam ~invariant except its eps)
EPS_CONS = float(os.environ.get("EPS_CONS", "0.0"))           # >0: also print committee consensus at this looser L-inf eps
DUMP_POP = os.environ.get("DUMP_POP", "")                     # path: save committee state_dicts at end for offline threshold sweeps
DUMP_EVERY = int(os.environ.get("DUMP_EVERY", "0"))           # >0: also dump the committee every N iters (versioned _itN.pt)

L1SRC = os.environ.get("L1SRC", "solved")
ITERS = int(os.environ.get("ITERS", "40"))
Q = int(os.environ.get("Q", "8000"))
MEASURE_EVERY = int(os.environ.get("MEASURE_EVERY", "5"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
CD = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
CACHE = "/tmp/claude-1001/-home-judah/480010f4-8e92-4a50-8319-7f2f544ea513/scratchpad/solved_l1_cache.pt"

# ---------- load teacher + population ----------
pop_ck = torch.load(CD + "_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt", map_location=dev, weights_only=False)
merged = torch.load(CD + "mergedbest512_sigmoid__784x128x80x40x32x10__s0_final.pt", map_location=dev, weights_only=False)
dims = pop_ck["dims"]; d, k = dims[0], dims[1]
teacher = MLP(dims, act="sigmoid").to(dev); teacher.load_state_dict(pop_ck["teacher_state"]); teacher.eval()
for p in teacher.parameters(): p.requires_grad_(False)
W1t = teacher.layers[0].weight.detach().float(); b1t = teacher.layers[0].bias.detach().float()
W2t = teacher.layers[1].weight.detach().float()
pop = [MLP(dims, act="sigmoid").to(dev) for _ in range(len(pop_ck["pop_states"]))]
for m, sd in zip(pop, pop_ck["pop_states"]): m.load_state_dict(sd)

# ---------- solved first layer, expressed in TRUE order (canonical) ----------
def match_sign(A, B):
    Cp = torch.cdist(A, B); Cm = torch.cdist(-A, B); C = torch.minimum(Cp, Cm)
    r, c = linear_sum_assignment(C.detach().cpu().numpy())
    sgn = torch.where(Cm[r, c] < Cp[r, c], -1.0, 1.0).to(A.device)
    return torch.tensor(r, device=A.device), torch.tensor(c, device=A.device), sgn

def recover_solved_l1():
    if os.path.exists(CACHE):
        z = torch.load(CACHE, map_location=dev); return z["W1"].to(dev), z["b1"].to(dev)
    print("[recover] jet-solving first layer (dir SVD + multi-harmonic mag)...", flush=True)
    td = MLP(dims, act="sigmoid").to(dev).double(); td.load_state_dict(pop_ck["teacher_state"]); td.eval()
    Wg = merged["state_dict"]["layers.0.weight"].to(dev).double().clone()
    bg = merged["state_dict"]["layers.0.bias"].to(dev).double().clone()
    Wgpinv = Wg.t() @ torch.linalg.inv(Wg @ Wg.t()); g = torch.Generator(device=dev).manual_seed(1)
    @torch.no_grad()
    def J_at(x, fd=5e-5):
        E = torch.eye(d, device=dev, dtype=torch.float64)
        return ((td(x.unsqueeze(0) + fd * E) - td(x.unsqueeze(0) - fd * E)) / (2 * fd)).t()
    N = torch.empty_like(Wg)
    for j in range(k):
        t = torch.full((k,), 20.0, device=dev, dtype=torch.float64); t[j] = 0.0
        x0 = Wgpinv @ (t - bg); U, S, Vh = torch.linalg.svd(J_at(x0), full_matrices=False)
        nv = Vh[0]; N[j] = nv if float(nv @ Wg[j]) > 0 else -nv
    V = N.t() @ torch.linalg.inv(N @ N.t()); Wdir = N * Wg.norm(dim=1, keepdim=True)
    W1rp = Wdir.t() @ torch.linalg.inv(Wdir @ Wdir.t()); O = dims[-1]
    def fit_mag(tails, a_g):
        ell = np.arange(7)[:, None]
        def res(a):
            tot = 0.0
            for ts, gs, side in tails:
                A = np.exp(-side * a * ell * ts[None, :]).T
                cf, _, _, _ = np.linalg.lstsq(A, gs, rcond=None); tot += float(((A @ cf - gs) ** 2).sum())
            return tot
        lo, hi = 0.8 * a_g, 1.2 * a_g
        for _ in range(70):
            m1 = hi - (hi - lo) * .618; m2 = lo + (hi - lo) * .618
            hi, lo = (m2, lo) if res(m1) < res(m2) else (hi, m1)
        return .5 * (lo + hi)
    a = torch.zeros(k, device=dev, dtype=torch.float64)
    for jj in range(k):
        a_g = float(Wg[jj].norm()); vj = V[:, jj]
        TT = (2 * torch.rand(40, k, generator=g, device=dev, dtype=torch.float64) - 1) * 2; TT[:, jj] = 0
        X0 = (TT - bg) @ W1rp.t()
        with torch.no_grad():
            sw = (td(X0 + (6 / a_g) * vj) - td(X0 - (6 / a_g) * vj)).norm(dim=1)
        top = torch.topk(sw, 8).indices
        tp = torch.linspace(2.5 / a_g, 7 / a_g, 40, device=dev, dtype=torch.float64)
        tm = torch.linspace(-7 / a_g, -2.5 / a_g, 40, device=dev, dtype=torch.float64)
        tails = []
        for mi in top.tolist():
            x0 = X0[mi]
            with torch.no_grad():
                Fp = td(x0.unsqueeze(0) + tp.unsqueeze(1) * vj.unsqueeze(0)).cpu().numpy()
                Fm = td(x0.unsqueeze(0) + tm.unsqueeze(1) * vj.unsqueeze(0)).cpu().numpy()
            for r in range(O):
                tails.append((tp.cpu().numpy(), Fp[:, r], 1)); tails.append((tm.cpu().numpy(), Fm[:, r], -1))
        a[jj] = fit_mag(tails, a_g)
    W1s = (a[:, None] * N).float(); b1s = bg.float()               # solved W (dir*mag) + consensus bias
    torch.save({"W1": W1s.cpu(), "b1": b1s.cpu()}, CACHE)
    return W1s, b1s

if L1SRC == "true":
    W1s, b1s = W1t.clone(), b1t.clone()
elif L1SRC == "consensus":
    W1s = merged["state_dict"]["layers.0.weight"].to(dev).float(); b1s = merged["state_dict"]["layers.0.bias"].to(dev).float()
else:
    W1s, b1s = recover_solved_l1()
# put solved L1 in TRUE order (canonical) so members share one ordering
r, c, s = match_sign(W1s, W1t); inv = torch.argsort(c)
W1s = (s[inv][:, None] * W1s[inv]).contiguous(); b1s = (s[inv] * b1s[inv]).contiguous()
W1s_pinv = W1s.t() @ torch.linalg.inv(W1s @ W1s.t())         # (784,128): W1s @ W1s_pinv = I -> query in z1-space
print(f"[L1={L1SRC}] solved-vs-true weight rel: mean {float(((W1s-W1t).norm(dim=1)/W1t.norm(dim=1)).mean()):.2e} "
      f"max {float(((W1s-W1t).norm(dim=1)/W1t.norm(dim=1)).max()):.2e}", flush=True)
def gen_queries_L2(pop, q, steps, lr, gen):
    """Queries that TARGET layer 2: parameterize by z1 (so members see s=sigma(z1) exactly),
    push toward the cube corners, and maximize disagreement in the members' LAYER-2 activations."""
    z1 = (torch.randn(q, k, generator=gen, device=dev) * 2.5).requires_grad_(True)
    opt = torch.optim.Adam([z1], lr=lr)
    W2 = [m.layers[1].weight for m in pop]; b2 = [m.layers[1].bias for m in pop]
    for step in range(steps):
        s = torch.sigmoid(z1)
        h2 = torch.stack([torch.sigmoid(s @ W2[i].t() + b2[i]) for i in range(len(pop))])  # (P,q,80)
        loss = disagreement(h2, cfg.disagree, cfg.qg_dist)   # negative L2 disagreement -> minimize
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return ((z1.detach() - b1s) @ W1s_pinv.t())          # x with members' s = sigma(z1)
def gen_queries_corner(pop, q, gen, G=6.0, sweep=2.5, eta=0.5):
    """Per-neuron corner targeting: for each L2 neuron j (member-0 estimate), drive z1 along
    sign(W2_j)*G and sweep the amplitude m so s runs corner -> transition -> corner, strongly
    exciting neuron j. Budget split across all 80 neurons (non-collapsing)."""
    W2 = pop[0].layers[1].weight.detach(); H = W2.shape[0]; per = max(1, q // H)
    d = torch.sign(W2)                                        # (H,128) aligned corner directions
    zs = []
    for j in range(H):
        m = (2 * torch.rand(per, generator=gen, device=dev) - 1) * sweep        # sweep across transition
        n = torch.randn(per, k, generator=gen, device=dev) * eta               # coverage on other dims
        zs.append(m[:, None] * d[j][None, :] * G + n)
    z1 = torch.cat(zs)[:q]
    return ((z1 - b1s) @ W1s_pinv.t()).detach()

# ---------- align each member's L2 to canonical (true) order, then freeze solved L1 ----------
mask_all = torch.ones(k, dtype=torch.bool)
for m in pop:
    Wm = m.layers[0].weight.detach()
    rr, cc, ss = match_sign(Wm, W1t)                       # member neuron i -> true cc[i], sign ss[i]
    W2 = m.layers[1].weight.detach().clone(); b2 = m.layers[1].bias.detach().clone()
    W2new = torch.zeros_like(W2); W2new[:, cc] = W2[:, rr] * ss[None, :]     # cols -> true order + sign
    comp = ss < 0; b2new = b2 + (W2[:, rr][:, comp]).sum(1)                 # complement bias shift
    with torch.no_grad():
        m.layers[1].weight.copy_(W2new); m.layers[1].bias.copy_(b2new)
    if FREEZE_L1_BIAS:
        _install_freeze(m, {0: (W1s, b1s, mask_all)})      # freeze L1 weights AND bias
    else:                                                  # freeze L1 weights only; leave bias trainable
        with torch.no_grad():
            m.layers[0].weight.copy_(W1s); m.layers[0].bias.copy_(b1s)
        m.layers[0].weight.register_hook(lambda gr: gr * 0.0)   # zero weight grad; bias grad flows
# optional: reinit layers 2+ (fresh diversity for consensus) while keeping frozen L1
if int(os.environ.get("REINIT", "0")):
    for m in pop:
        fresh = MLP(dims, act="sigmoid").to(dev)
        with torch.no_grad():
            for l in range(1, len(dims) - 1):
                m.layers[l].weight.copy_(fresh.layers[l].weight); m.layers[l].bias.copy_(fresh.layers[l].bias)
    print("[reinit] layers 2+ reinitialized fresh (L1 frozen shared)", flush=True)
opts = [torch.optim.Adam(m.parameters(), lr=float(os.environ.get("LR", "1e-3"))) for m in pop]

# ---------- L2 consensus measurement (L1 shared => L2 cols already in true order) ----------
def report(it, X, Y):                                  # identical format to run.py / reconstruct()
    losses = l1_on(pop, X, Y)
    cnet = build_consensus(pop, dims, quorum_ratio=cfg.cluster_quorum)
    cc = param_errors(cnet, teacher)["max_eps"] if cnet is not None else None
    cstats = consensus_neuron_stats(pop, teacher, dims, quorum_ratio=cfg.cluster_quorum)
    bi = min(range(len(pop)), key=lambda i: losses[i]); best = pop[bi]
    errs = param_errors(best, teacher)
    mean_eps = sum(errs["mean_eps_per_matrix"]) / len(errs["mean_eps_per_matrix"])
    agree = agreement(best, teacher, eval_pts); wall = round(time.time() - t0, 1)
    cc_str = f"{cc:.2e}" if cc is not None else "  n/a  "
    nmat = len(dims) - 1; pmax = errs.get("max_eps_per_matrix"); pmean = errs.get("mean_eps_per_matrix")
    if cstats is not None and "layers" in cstats:
        parts = " ".join(f"L{li+1}:{x['n_cons']}/{x['n_tot']}" for li, x in enumerate(cstats["layers"]))
        e = (f"max {cstats['max_eps']:.2e} mean {cstats['mean_eps']:.2e}"
             if cstats.get("max_eps") is not None else "n/a")
        cons_str = f"{cstats['n_consensus']}/{cstats['n_total']} [{parts}] {e}"
    else:
        cons_str = f"0/{dims[1]}"
    print(f"  it {it:3d} | q {it*cfg.q:6d} | loss {losses[bi]:.2e} | max_eps {errs['max_eps']:.2e} | "
          f"mean_eps {mean_eps:.2e} | cluster {cc_str} | cons {cons_str} | agree {agree:.4f} | {wall}s", flush=True)
    if EPS_CONS > 0:
        c2 = consensus_neuron_stats(pop, teacher, dims, eps=EPS_CONS, quorum_ratio=cfg.cluster_quorum)
        if c2 is not None and "layers" in c2:
            p2 = " ".join(f"L{li+1}:{x['n_cons']}/{x['n_tot']}" for li, x in enumerate(c2["layers"]))
            L2c = c2["layers"][1]
            acc = (f"  L2 cons-err mean {L2c['cons_mean']:.2e} max {L2c['cons_max']:.2e}"
                   if L2c.get("cons_mean") is not None else "")
            print(f"        cons@eps={EPS_CONS:g}: {c2['n_consensus']}/{c2['n_total']} [{p2}]{acc}", flush=True)
    # --- L2 consensus formed on WEIGHT ONLY (bias excluded from the criterion); W & B scored apart ---
    W2t = teacher.layers[1].weight.detach(); b2t = teacher.layers[1].bias.detach(); ntt = W2t.norm(dim=1)
    AW, AB = [], []
    for m in pop:
        W2m = m.layers[1].weight.detach(); b2m = m.layers[1].bias.detach()
        Cp = torch.cdist(W2m, W2t); Cm = torch.cdist(-W2m, W2t); C = torch.minimum(Cp, Cm)
        rr, cc = linear_sum_assignment(C.cpu().numpy()); rr = torch.tensor(rr); cc = torch.tensor(cc)
        sgn = torch.where(Cm[rr, cc] < Cp[rr, cc], -1.0, 1.0)
        W2a = torch.zeros_like(W2m); b2a = torch.zeros_like(b2m)
        W2a[cc] = W2m[rr] * sgn[:, None]; b2a[cc] = b2m[rr] * sgn
        AW.append(W2a); AB.append(b2a)
    AW = torch.stack(AW); AB = torch.stack(AB); quo = math.ceil(0.625 * len(pop)); Hn = W2t.shape[0]
    for we in (0.02, 0.2):
        We, Be = [], []
        for j in range(Hn):
            medW = AW[:, j].median(0).values; cl = ((AW[:, j] - medW).norm(dim=1) / ntt[j]) < we
            if int(cl.sum()) >= quo:
                We.append(float((AW[cl, j].mean(0) - W2t[j]).norm() / ntt[j]))
                Be.append(float((AB[cl, j].mean() - b2t[j]).abs()))
        if We:
            We = np.array(We); Be = np.array(Be)
            print(f"        W-cons@{we:g}: {len(We)}/{Hn} | W-err mean {We.mean():.2e} max {We.max():.2e} | "
                  f"B-err mean {Be.mean():.2e} max {Be.max():.2e}", flush=True)
        else:
            print(f"        W-cons@{we:g}: 0/{Hn}", flush=True)
    if pmax:
        print("        eps/layer (all):  " + "  ".join(
            f"L{i+1}[max {max(pmax[2*i], pmax[2*i+1]):.2e} mean {pmean[2*i]:.2e}]" for i in range(nmat)), flush=True)
    if cstats is not None and "layers" in cstats:
        def _clyr(i, x):
            if x.get("cons_max") is not None:
                return f"L{i+1}[max {x['cons_max']:.2e} mean {x['cons_mean']:.2e} ({x['n_cons']}/{x['n_tot']})]"
            return f"L{i+1}[n/a ({x['n_cons']}/{x['n_tot']})]"
        print("        eps/layer (cons): " + "  ".join(_clyr(i, x) for i, x in enumerate(cstats["layers"])), flush=True)

# ---------- cfg (mergedbest512, sigmoid) ----------
WINDOW = int(os.environ.get("WINDOW", "0"))     # keep last WINDOW*Q queries (0 = all)
cfg = Cfg(qg_steps=30, qg_lr=float(os.environ.get("QG_LR", "0.1")), qg_dist="l1", qg_init="uniform", qg_range=1.0,
          disagree="median_pair", fit_loss="l1", batch=512, window=WINDOW, warmstart_iters=5,
          gate_kappa=0.0, act="sigmoid", p=len(pop), q=Q, outer=ITERS,
          epochs=int(os.environ.get("EPOCHS", "10")))

# ---------- peel loop ----------
t0 = time.time()
eval_pts = torch.randn(4000, d, device=dev)
gen = torch.Generator(device=dev).manual_seed(0)
X = torch.empty(0, d); Y = torch.empty(0, dims[-1])
print(f"[setup] teacher {dims} epochs=25 act=sigmoid device={dev}", flush=True)
print(f"[run] mergedbest512 (PEEL: L1={L1SRC} {'W+b frozen' if FREEZE_L1_BIAS else 'W frozen, BIAS trainable'}"
      f"{', L2+ reinit' if int(os.environ.get('REINIT','0')) else ''}, TARGET={TARGET}) "
      f"seed=0 outer={ITERS} q={Q} window={WINDOW}", flush=True)
_Xi = torch.randn(2048, d, device=dev) * 0.5
report(0, _Xi.cpu(), teacher(_Xi).detach().cpu())
for t in range(ITERS):
    if t < cfg.warmstart_iters:
        I = torch.randn(cfg.q, d, generator=gen, device=dev) * cfg.qg_init_std
    elif TARGET == "L2":
        I = gen_queries_L2(pop, cfg.q, cfg.qg_steps, cfg.qg_lr, gen)
    elif TARGET == "corner":
        I = gen_queries_corner(pop, cfg.q, gen)
    else:
        I = gen_queries(pop, cfg, d, dev, gen, losses=None)
    with torch.no_grad():
        T = teacher(I)
    X = torch.cat([X, I.detach().cpu()]); Y = torch.cat([Y, T.detach().cpu()])
    if cfg.window > 0:
        keep = cfg.window * cfg.q; X, Y = X[-keep:], Y[-keep:]
    n = len(X)
    for ep in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen, device=dev)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch].cpu(); xb = X[idx].to(dev); yb = Y[idx].to(dev)
            for net, opt in zip(pop, opts):
                opt.zero_grad()
                loss = LOSS_SCALE * ((net(xb) - yb).abs().mean() if cfg.fit_loss == "l1" else ((net(xb) - yb) ** 2).mean())
                loss.backward(); opt.step()
    if (t + 1) % MEASURE_EVERY == 0 or t == ITERS - 1:
        report(t + 1, X.to(dev), Y.to(dev))
    if DUMP_POP and DUMP_EVERY > 0 and (t + 1) % DUMP_EVERY == 0:
        pth = DUMP_POP[:-3] + f"_it{t+1}.pt" if DUMP_POP.endswith(".pt") else DUMP_POP + f"_it{t+1}"
        torch.save({"dims": dims, "act": "sigmoid", "pop_states": [m.state_dict() for m in pop],
                    "teacher_state": teacher.state_dict()}, pth)
        print(f"[dump] committee saved to {pth}", flush=True)
if DUMP_POP:
    torch.save({"dims": dims, "act": "sigmoid", "pop_states": [m.state_dict() for m in pop],
                "teacher_state": teacher.state_dict()}, DUMP_POP)
    print(f"[dump] committee ({len(pop)} members) saved to {DUMP_POP}", flush=True)
