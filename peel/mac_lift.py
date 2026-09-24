"""mac_lift.py -- Method-of-Auxiliary-Coordinates (lifted / AL) training for
first-layer recovery, per the design in the conversation.

Question: does lifting the hidden PRE-activations Z (augmented-Lagrangian
block-coordinate training) let independently-initialised students recover /
agree on the TRUE first layer, where ordinary joint sigmoid backprop drifts?

Net (exactly two hidden layers, mirrors 3072->1024->512->100):
    x -> (W1,b1) -> sigma -> (W2,b2) -> sigma -> (W3,b3) -> yhat

Lifted variables (per example, persistent across outer iters):
    Z1 ~ X W1^T + b1   (R1 = Z1 - (X W1^T + b1))
    Z2 ~ sigma(Z1) W2^T + b2   (R2 = Z2 - (sigma(Z1) W2^T + b2))

Objective (all residuals per-element normalised so alpha = rho2/rho1 is meaningful):
    L_teacher(yhat, Y) + <Lam1,R1> + (rho1/2) mean(R1^2)
                       + <Lam2,R2> + (rho2/2) mean(R2^2)

Solvers:
    baseline : plain Adam through the whole sigmoid stack.
    mac1     : lift Z1 only. Z2 computed exactly; teacher path L->W3->Z2->W2->Z1
               stays intact (no per-example Z2 shortcut). W1 = linear solve from
               Z1; (W2,b2,W3,b3) = Adam on the fixed-H1 tail.
    mac2     : lift Z1 and Z2. W1,W2 = linear solves; W3 = linear solve (MSE);
               Z1,Z2 = Adam. Here Z2 CAN shortcut the output, so rho2 is the
               information bridge to L1 -- swept via alpha.

W-solves use the dual-shifted normal equations
    (Xt X + lam I) A = Xt (Z + Lam/rho)
with a cached Cholesky of the L1 Gram matrix (input design is fixed).

Metrics per run: aligned L1 error (rel/mean/max, sigmoid gauge), cross-member L1
consensus error, teacher MSE, ||R1||/||Z1||, ||R2||/||Z2||.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nets import MLP                                  # noqa: E402
from align import sign_canonicalize_, greedy_perm     # noqa: E402


# --------------------------------------------------------------------------- #
# teacher + queries
# --------------------------------------------------------------------------- #
def build_teacher(dims, seed, wscale, device):
    """Synthetic sigmoid teacher with weight scale tuned for mild saturation."""
    torch.manual_seed(seed)
    net = MLP(dims, act="sigmoid").to(device)
    with torch.no_grad():
        for i, l in enumerate(net.layers):
            fan_in = l.weight.shape[1]
            # hidden layers scaled to give preact std ~ wscale; output modest
            s = wscale / (fan_in ** 0.5) if i < len(net.layers) - 1 else 1.0 / (fan_in ** 0.5)
            l.weight.normal_(0, s)
            l.bias.normal_(0, 0.3 if i < len(net.layers) - 1 else 0.0)
    return net


def gen_queries(teacher, n, device, seed, xscale=1.0):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, teacher.dims[0], generator=g, device=device) * xscale
    with torch.no_grad():
        y = teacher(x)
    return x, y


# --------------------------------------------------------------------------- #
# alignment / metrics (sigmoid gauge; scoring only, truth used only here)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def l1_aligned(student, teacher):
    """Return student W1,b1 aligned to teacher gauge (sign-canon + greedy perm
    on layer-0 [W|b] rows). Truth used for the permutation -> scoring only."""
    t = teacher.clone(); sign_canonicalize_(t)
    r = student.clone(); sign_canonicalize_(r)
    A = torch.cat([t.layers[0].weight, t.layers[0].bias[:, None]], 1)
    B = torch.cat([r.layers[0].weight, r.layers[0].bias[:, None]], 1)
    perm = greedy_perm(A, B)
    idx = torch.tensor(perm, device=A.device)
    return r.layers[0].weight[idx].clone(), r.layers[0].bias[idx].clone(), \
        t.layers[0].weight.clone(), t.layers[0].bias.clone()


@torch.no_grad()
def l1_error(student, teacher):
    W, b, Wt, bt = l1_aligned(student, teacher)
    rel = (W - Wt).norm() / Wt.norm()
    return dict(rel=rel.item(),
                mean=(W - Wt).abs().mean().item(),
                max=(W - Wt).abs().max().item())


@torch.no_grad()
def l1_consensus(students, teacher):
    """Align every member to teacher gauge, average their W1, report the
    consensus net's L1 rel error + the cross-member spread."""
    Ws = []
    for s in students:
        W, b, Wt, bt = l1_aligned(s, teacher)
        Ws.append(W)
    Wt = l1_aligned(students[0], teacher)[2]
    Wstack = torch.stack(Ws)                       # (P, h1, din)
    cons = Wstack.mean(0)
    rel = (cons - Wt).norm() / Wt.norm()
    spread = Wstack.std(0).mean().item()           # mean per-weight std across members
    return dict(cons_rel=rel.item(), spread=spread,
                member_rel_mean=sum((W - Wt).norm().item() / Wt.norm().item()
                                    for W in Ws) / len(Ws))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def aug(x):
    return torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], 1)


def chol_factor(Xa, ridge):
    G = Xa.T @ Xa
    G += ridge * torch.eye(G.shape[0], device=G.device)
    return torch.linalg.cholesky(G)


def solve_layer(Xa, target, chol):
    """(Xa^T Xa + ridge I) A = Xa^T target ; returns (W, b) with A rows [W^T; b]."""
    rhs = Xa.T @ target
    A = torch.cholesky_solve(rhs, chol)            # (in+1, out)
    return A[:-1].T.contiguous(), A[-1].contiguous()


def new_student(dims, seed, device, teacher=None, warm=-1.0):
    torch.manual_seed(seed)
    s = MLP(dims, act="sigmoid").to(device)
    if warm >= 0 and teacher is not None:
        with torch.no_grad():
            Wt, bt = teacher.layers[0].weight, teacher.layers[0].bias
            eta = torch.randn_like(Wt); eta *= warm * Wt.norm() / eta.norm()
            etab = torch.randn_like(bt); etab *= warm * bt.norm() / etab.norm().clamp_min(1e-9)
            s.layers[0].weight.copy_(Wt + eta)
            s.layers[0].bias.copy_(bt + etab)
    return s


# --------------------------------------------------------------------------- #
# solvers
# --------------------------------------------------------------------------- #
def train_baseline(student, X, Y, steps, lr, log=None, teacher=None):
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    for t in range(steps):
        opt.zero_grad()
        loss = ((student(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
        if log is not None and t % max(1, steps // 10) == 0:
            log.append((t, loss.item(), l1_error(student, teacher)["rel"]))
    return student


def train_mac(student, X, Y, lift2, rho1, rho2, outer, z_steps, z_lr,
              tail_steps, tail_lr, ridge, anneal=None, log=None, teacher=None):
    """lift2=False -> MAC-1 (Z1 only); lift2=True -> MAC-2 (Z1,Z2)."""
    dev = X.device
    Xa = aug(X)
    cholX = chol_factor(Xa, ridge)                 # cached L1 Gram (input fixed)

    W1, b1 = student.layers[0].weight, student.layers[0].bias
    W2, b2 = student.layers[1].weight, student.layers[1].bias
    W3, b3 = student.layers[2].weight, student.layers[2].bias

    with torch.no_grad():
        Z1 = X @ W1.T + b1                          # persistent lifted vars
        Z2 = torch.sigmoid(Z1) @ W2.T + b2
    Lam1 = torch.zeros_like(Z1)
    Lam2 = torch.zeros_like(Z2)

    def resid_norms():
        with torch.no_grad():
            r1 = Z1 - (X @ W1.T + b1)
            r2 = Z2 - (torch.sigmoid(Z1) @ W2.T + b2)
            return (r1.norm() / Z1.norm().clamp_min(1e-9)).item(), \
                   (r2.norm() / Z2.norm().clamp_min(1e-9)).item()

    for it in range(outer):
        if anneal is not None:
            rho2 = anneal(it, rho1)

        # ---- 1. Z-optimisation (persistent, warm-started) -------------------
        Z1.requires_grad_(True)
        if lift2:
            Z2.requires_grad_(True)
            zopt = torch.optim.Adam([Z1, Z2], lr=z_lr)
        else:
            zopt = torch.optim.Adam([Z1], lr=z_lr)
        for _ in range(z_steps):
            zopt.zero_grad()
            H1 = torch.sigmoid(Z1)
            if lift2:
                yhat = torch.sigmoid(Z2) @ W3.T.detach() + b3.detach()
                R2 = Z2 - (H1 @ W2.T.detach() + b2.detach())
            else:
                Z2c = H1 @ W2.T.detach() + b2.detach()
                yhat = torch.sigmoid(Z2c) @ W3.T.detach() + b3.detach()
                R2 = torch.zeros(1, device=dev)
            R1 = Z1 - (X @ W1.T.detach() + b1.detach())
            loss = ((yhat - Y) ** 2).mean()
            loss = loss + (Lam1 * R1).mean() + 0.5 * rho1 * (R1 ** 2).mean()
            if lift2:
                loss = loss + (Lam2 * R2).mean() + 0.5 * rho2 * (R2 ** 2).mean()
            loss.backward()
            zopt.step()
        Z1 = Z1.detach()
        if lift2:
            Z2 = Z2.detach()

        # ---- 2. linear solve for L1 from Z1 (dual-shifted target) -----------
        with torch.no_grad():
            tgt1 = Z1 + Lam1 / rho1
            W1n, b1n = solve_layer(Xa, tgt1, cholX)
            W1.copy_(W1n); b1.copy_(b1n)

        # ---- 3. L2 update --------------------------------------------------
        H1 = torch.sigmoid(Z1)
        if lift2:
            with torch.no_grad():
                Ha = aug(H1)
                cholH = chol_factor(Ha, ridge)
                tgt2 = Z2 + Lam2 / rho2
                W2n, b2n = solve_layer(Ha, tgt2, cholH)
                W2.copy_(W2n); b2.copy_(b2n)
            # ---- 4. output layer: linear solve on sigma(Z2) -----------------
            with torch.no_grad():
                F2 = aug(torch.sigmoid(Z2))
                cholF = chol_factor(F2, ridge)
                W3n, b3n = solve_layer(F2, Y, cholF)
                W3.copy_(W3n); b3.copy_(b3n)
        else:
            # MAC-1: Z1 fixed -> refine (W2,b2,W3,b3) on the exact tail by Adam
            opt = torch.optim.Adam([W2, b2, W3, b3], lr=tail_lr)
            for _ in range(tail_steps):
                opt.zero_grad()
                yhat = torch.sigmoid(torch.sigmoid(Z1.detach()) @ W2.T + b2) @ W3.T + b3
                ((yhat - Y) ** 2).mean().backward()
                opt.step()
            with torch.no_grad():
                Z2 = torch.sigmoid(Z1) @ W2.T + b2

        # ---- 5. AL dual updates --------------------------------------------
        with torch.no_grad():
            R1 = Z1 - (X @ W1.T + b1)
            Lam1 += rho1 * R1
            if lift2:
                R2 = Z2 - (torch.sigmoid(Z1) @ W2.T + b2)
                Lam2 += rho2 * R2

        if log is not None and it % max(1, outer // 10) == 0:
            with torch.no_grad():
                tl = ((student(X) - Y) ** 2).mean().item()
            r1n, r2n = resid_norms()
            log.append((it, tl, l1_error(student, teacher)["rel"], r1n, r2n, rho2))
    return student


# --------------------------------------------------------------------------- #
# experiment driver
# --------------------------------------------------------------------------- #
def run_config(teacher, X, Y, dims, solver, P, alpha, rho1, args, device):
    students, logs = [], []
    anneal = None
    if args.anneal and solver == "mac2":
        sched = [0.03, 0.1, 0.3, 1.0]

        def anneal(it, r1):
            frac = it / max(1, args.outer - 1)
            a = sched[min(len(sched) - 1, int(frac * len(sched)))]
            return a * r1
    for m in range(P):
        s = new_student(dims, seed=1000 + m, device=device,
                        teacher=teacher, warm=args.warm)
        log = []
        if solver == "baseline":
            train_baseline(s, X, Y, args.baseline_steps, args.lr,
                           log=log, teacher=teacher)
        elif solver == "mac1":
            train_mac(s, X, Y, lift2=False, rho1=rho1, rho2=rho1,
                      outer=args.outer, z_steps=args.z_steps, z_lr=args.z_lr,
                      tail_steps=args.tail_steps, tail_lr=args.tail_lr,
                      ridge=args.ridge, log=log, teacher=teacher)
        elif solver == "mac2":
            train_mac(s, X, Y, lift2=True, rho1=rho1, rho2=alpha * rho1,
                      outer=args.outer, z_steps=args.z_steps, z_lr=args.z_lr,
                      tail_steps=args.tail_steps, tail_lr=args.tail_lr,
                      ridge=args.ridge, anneal=anneal, log=log, teacher=teacher)
        students.append(s); logs.append(log)
    # metrics
    with torch.no_grad():
        tl = sum(((s(X) - Y) ** 2).mean().item() for s in students) / P
    errs = [l1_error(s, teacher) for s in students]
    cons = l1_consensus(students, teacher)
    l1_rel_mean = sum(e["rel"] for e in errs) / P
    r1n = r2n = float("nan")
    if logs[0] and len(logs[0][-1]) >= 5:
        r1n, r2n = logs[0][-1][3], logs[0][-1][4]
    return dict(solver=solver, alpha=alpha, teacher_mse=tl,
                l1_rel_mean=l1_rel_mean, cons_rel=cons["cons_rel"],
                spread=cons["spread"], r1n=r1n, r2n=r2n,
                l1_start=logs[0][0][2] if logs[0] else float("nan"),
                l1_end=errs[0]["rel"], logs=logs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="256,128,64,20")
    ap.add_argument("--N", type=int, default=20000)
    ap.add_argument("--P", type=int, default=5)
    ap.add_argument("--wscale", type=float, default=2.5)
    ap.add_argument("--warm", type=float, default=-1.0,
                    help=">=0: init L1 = teacher + warm*rel noise (drift test)")
    ap.add_argument("--rho1", type=float, default=10.0)
    ap.add_argument("--alphas", default="0.03,0.1,0.3,1.0")
    ap.add_argument("--outer", type=int, default=60)
    ap.add_argument("--z_steps", type=int, default=60)
    ap.add_argument("--z_lr", type=float, default=0.05)
    ap.add_argument("--tail_steps", type=int, default=60)
    ap.add_argument("--tail_lr", type=float, default=3e-3)
    ap.add_argument("--baseline_steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--anneal", action="store_true")
    ap.add_argument("--solvers", default="baseline,mac1,mac2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    dims = [int(x) for x in args.dims.split(",")]
    assert len(dims) == 4, "this harness is the 2-hidden-layer case (4 dims)"

    teacher = build_teacher(dims, args.seed, args.wscale, dev)
    X, Y = gen_queries(teacher, args.N, dev, args.seed, xscale=1.0)
    # report teacher saturation
    with torch.no_grad():
        z1 = X @ teacher.layers[0].weight.T + teacher.layers[0].bias
        z2 = torch.sigmoid(z1) @ teacher.layers[1].weight.T + teacher.layers[1].bias
    print(f"[teacher] dims={dims} wscale={args.wscale}  "
          f"preact std z1={z1.std():.2f} z2={z2.std():.2f}  "
          f"|Y| std={Y.std():.3f}  warm={args.warm}")
    print(f"[queries] N={args.N}  device={dev}\n")

    solvers = args.solvers.split(",")
    alphas = [float(a) for a in args.alphas.split(",")]
    rows = []
    for solver in solvers:
        if solver == "mac2":
            for a in alphas:
                rows.append(run_config(teacher, X, Y, dims, solver, args.P, a,
                                       args.rho1, args, dev))
        else:
            rows.append(run_config(teacher, X, Y, dims, solver, args.P, 1.0,
                                   args.rho1, args, dev))

    hdr = f"{'solver':<10}{'alpha':>7}{'tMSE':>10}{'L1rel(mem)':>12}" \
          f"{'L1cons':>10}{'spread':>9}{'R1/Z1':>9}{'R2/Z2':>9}{'L1 st->end':>14}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        tag = f"{r['solver']}" + (f"+anneal" if (args.anneal and r['solver'] == 'mac2') else "")
        print(f"{tag:<10}{r['alpha']:>7.2f}{r['teacher_mse']:>10.2e}"
              f"{r['l1_rel_mean']*100:>11.2f}%{r['cons_rel']*100:>9.2f}%"
              f"{r['spread']:>9.3f}{r['r1n']:>9.1e}{r['r2n']:>9.1e}"
              f"  {r['l1_start']*100:>5.1f}->{r['l1_end']*100:<5.1f}")


if __name__ == "__main__":
    main()
