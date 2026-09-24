"""Does whitening C5's input coordinates fix the weight-recovery stall?

Setup (mirrors LeNet peel phase 3): exact frozen prefix (teacher L1,L2), exact
TRAINABLE-shaped tail, C5 trainable, cheat-mined + random queries. C5 is fixed
to teacher's exact tail here so its weights are UNIQUELY pinned (no scale gauge)
-> direct eps vs teacher. Two arms differ only in C5's coordinate system:

  raw      : train C5 weights directly (Adam)
  whitened : train C5 in coords where its input covariance is ~isotropic
             (Tikhonov Sigma^-1/2), exactly reparameterized & mapped back

If the stall is conditioning (70:1 input spectrum), whitened recovers C5 far
below raw. If not, they match and the hypothesis dies.
"""
import torch
import torch.nn.functional as F

from data import make_teacher_cnn
from method import Cfg, gen_queries, _cnn_frontier_input, _cnn_tail_forward
from nets import ConvNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFGS = [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)]
FRONTIER = 2                                    # C5


def c5_rows(A, c):
    return torch.cat([A.view(120, -1), c.view(-1, 1)], 1)


def main():
    t = make_teacher_cnn((1, 28, 28), CFGS, (84,), 10, epochs=25, seed=0,
                         device=DEV, act="leaky_relu")
    for p in t.parameters():
        p.requires_grad_(False)
    At = t.layers[FRONTIER].weight.detach().reshape(120, -1)   # (120,400)
    ct = t.layers[FRONTIER].bias.detach()
    Dt = c5_rows(At, ct)

    # --- covariance of C5 input under probe queries through the exact prefix ---
    g = torch.Generator(device=DEV).manual_seed(0)
    Xp = torch.rand(16384, 784, generator=g, device=DEV) * 2 - 1
    with torch.no_grad():
        Hp = _cnn_frontier_input(t, Xp, FRONTIER).double()     # (N,400)
    mu = Hp.mean(0)
    Hc = Hp - mu
    Sig = (Hc.T @ Hc) / len(Hc)
    lam, U = torch.linalg.eigh(Sig)                            # ascending
    lam = lam.clamp_min(0)
    eps_floor = 1e-2 * lam.max()                              # Tikhonov floor
    inv_sqrt = (lam + eps_floor).rsqrt()
    sqrt = (lam + eps_floor).sqrt()
    M = (U * inv_sqrt) @ U.T                                   # Sigma^-1/2 (reg)
    Minv = (U * sqrt) @ U.T
    print(f"C5 input spectrum: sqrt(lam) {lam.max().sqrt():.3f}..{lam.min().sqrt():.2e}"
          f"  eff-rank {(lam.sum()**2/(lam**2).sum()):.1f}/400  "
          f"cond after whiten {(inv_sqrt.max()/inv_sqrt.min()):.1f}")
    M, Minv, mu = M.float(), Minv.float(), mu.float()

    # --- accumulate a large cheat-mined + random query buffer -------------
    cfgm = Cfg(p=1, q=8192, qg_steps=30, qg_lr=0.1, qg_dist="l1",
               qg_init="uniform", qg_range=1.0, disagree="median_pair")
    Xs = []
    for _ in range(6):
        s_tmp = ConvNet((1, 28, 28), CFGS, (84,), 10, "leaky_relu").to(DEV)
        Xs.append(gen_queries([s_tmp, t], cfgm, 784, DEV, g))
        Xs.append(torch.rand(8192, 784, generator=g, device=DEV) * 2 - 1)
    X = torch.cat(Xs)
    with torch.no_grad():
        Y = t(X)
    print(f"buffer: {len(X)} queries")

    def run(whiten):
        torch.manual_seed(0)
        # trainable C5 (whitened or raw params) + fresh trainable tail
        A0 = (At + 0.3 * torch.randn_like(At)).clone()        # perturbed start
        c0 = (ct + 0.3 * torch.randn_like(ct)).clone()
        if whiten:
            # A_w = A Minv, c_w = c + A mu  (so z = A_w h_w + c_w, h_w=M(h-mu))
            Aw = (A0 @ Minv).detach().requires_grad_(True)
            cw = (c0 + A0 @ mu).detach().requires_grad_(True)
        else:
            Aw = A0.detach().requires_grad_(True)
            cw = c0.detach().requires_grad_(True)
        # tail = teacher's EXACT tail, FROZEN -> C5 scale gauge is broken, so
        # A,c are uniquely pinned and compare directly to teacher (no align).
        tail = t
        opt = torch.optim.Adam([Aw, cw], lr=1e-3)

        def c5_apply(h):
            if whiten:
                hw = (h - mu) @ M.T
                return hw @ Aw.T + cw
            return h @ Aw.T + cw

        def recover():
            if whiten:
                A = Aw.detach() @ M
                c = cw.detach() - (Aw.detach() @ M) @ mu
            else:
                A, c = Aw.detach(), cw.detach()
            return A, c

        def forward(xb):
            h = _cnn_frontier_input(t, xb, FRONTIER)          # exact prefix
            z = c5_apply(h)
            a = F.leaky_relu(z, 0.01)
            return _cnn_tail_forward(tail, a, FRONTIER + 1)    # trainable tail

        n = len(X)
        for step in range(6000):
            if step in (3000, 4800):
                for grp in opt.param_groups:
                    grp["lr"] /= 10
            idx = torch.randint(0, n, (512,), generator=g, device=DEV)
            opt.zero_grad()
            (forward(X[idx]) - Y[idx]).abs().mean().backward()
            opt.step()
            if (step + 1) % 1500 == 0:
                A, c = recover()
                e = (Dt - c5_rows(A, c)).abs().max(1).values
                with torch.no_grad():
                    fl = (forward(X[:8192]) - Y[:8192]).abs().mean().item()
                print(f"    step {step+1:5d}: fit {fl:.4f} | C5 eps "
                      f"med {e.median():.3e} max {e.max():.3e} "
                      f">0.05:{(e>0.05).sum().item():3d}/120")
        return

    print("\n== RAW C5 coordinates ==")
    run(False)
    print("\n== WHITENED C5 coordinates ==")
    run(True)


if __name__ == "__main__":
    main()
