"""Activation-based alignment that quotients the extra isomorphisms of SPARSE
(leaky-)ReLU MLPs, for scoring a reconstruction against its teacher.

Plain alignment (align.align_clone_to: unit rows + greedy weight matching, then
max |dW| over every entry) assumes every hidden unit is identifiable. A sparse
teacher has units that are not, and each one inflates the error with entries no
attacker could ever pin down:

  dead-in    zero incoming row: the unit is the constant act(b), folded into the
             next bias. Its row/bias/column are gauge.
  dead-out   zero outgoing column (or feeding only irrelevant units, recursively):
             the unit never reaches the output; its row is arbitrary.
  linear     its pre-activation never changes sign on the probe domain X, so on X
             it is the affine map s*(w.x + b) (s = 1 or the leaky slope). Only the
             SUM over all such units, W_next[:, lin] diag(s) [W | b][lin], is
             identifiable from X -- individually they are a continuous gauge. (On
             all of R^d any nonzero row kinks somewhere; X = where the attacker
             actually queries.)
  duplicate  two kinking units with parallel [w | b] (same hyperplane, same
             orientation): act(a z) + act(c z) = (a + c) act(z) for a, c > 0, so
             only the summed column is identifiable. Grouped, scored as one unit.

Units that remain ('identifiable') are paired teacher -> recon by pre-activation
correlation on X (Hungarian, like head_refine._act_perm), which is invariant to the
rescaling gauge and does not depend on weight-space closeness the way greedy
matching of sparse rows does. Scores:
  unit angle      angle between teacher and paired recon w (WITHOUT bias: when
                  |b| >> ||w|| a [w | b] angle only measures the bias ratio)
  bias err        |b_r/||w_r|| - b_t/||w_t||| (kink offset in row-norm units)
  unit eps        max |d| of the unit-normalised w
  col rel err     ||c_r - c_t|| / ||c_t|| of the outgoing column after pushing the
                  row norm into it (duplicate groups: summed columns)
  affine rel err  relative error of the affine part on X: sum over linear units of
                  col * s * [w | b], plus dead-in constants and the output bias
                  (teacher's linear units vs the recon's unpaired linear units)
  fn rel err      ||f_r(X) - f_t(X)|| / ||f_t(X) - mean||
One hidden layer only (the 12288x1024x200 study): for deeper nets the rows of
layer l live in layer l-1's unit frame, which would have to be aligned first.
"""
import torch
from scipy.optimize import linear_sum_assignment

DEAD_TOL = 0.0          # exact zeros (prox-L1 teachers); raise for penalty-L1 chatter
MIN_FRAC = 0.01         # a unit 'kinks on X' iff >= MIN_FRAC of the probes lie on each side
DUP_COS = 1 - 1e-9      # [w|b] cosine above which two kinking units are one hyperplane


@torch.no_grad()
def _forward_preacts(net, X):
    zs, h = [], X
    for layer in net.layers[:-1]:
        z = layer(h)
        zs.append(z)
        h = net.act(z)
    return zs, net.layers[-1](h)


@torch.no_grad()
def classify_units(net, X, dead_tol=DEAD_TOL, min_frac=MIN_FRAC, dup_cos=DUP_COS):
    """Per hidden layer: dict of bool masks dead_in, dead_out, linear, dup (non-
    representative members of a duplicate group), ident; plus 'groups' = list of
    index lists (duplicate groups among the kinking units, representative first)
    and 'side' = +1/-1 for linear units (which side of the kink X lives on)."""
    zs, _ = _forward_preacts(net, X)
    Ls = len(net.layers) - 1
    info = [None] * Ls
    relevant_next = torch.ones(net.layers[-1].weight.shape[0], dtype=torch.bool, device=X.device)
    for l in reversed(range(Ls)):
        W = net.layers[l].weight
        Wn = net.layers[l + 1].weight
        dead_in = W.abs().amax(1) <= dead_tol
        dead_out = (Wn[relevant_next].abs() <= dead_tol).all(0)
        z = zs[l]
        pos, neg = (z > 0).double().mean(0), (z < 0).double().mean(0)
        linear = ~((pos >= min_frac) & (neg >= min_frac)) & ~dead_in & ~dead_out
        kink = ~dead_in & ~dead_out & ~linear
        side = torch.where(pos >= neg, 1, -1)
        # duplicate hyperplanes among kinking units
        idx = kink.nonzero(as_tuple=True)[0]
        R = torch.cat([W[idx], net.layers[l].bias[idx, None]], 1).double()
        R = R / R.norm(dim=1, keepdim=True).clamp_min(1e-300)
        dup = torch.zeros_like(kink)
        groups = []
        if len(idx):
            C = R @ R.T
            seen = torch.zeros(len(idx), dtype=torch.bool, device=X.device)
            for a in range(len(idx)):
                if seen[a]:
                    continue
                g = (C[a] > dup_cos).nonzero(as_tuple=True)[0]
                seen[g] = True
                groups.append([int(idx[j]) for j in g])
                if len(g) > 1:
                    dup[idx[g[1:]]] = True
        ident = kink & ~dup
        info[l] = dict(dead_in=dead_in, dead_out=dead_out, linear=linear, dup=dup,
                       ident=ident, groups=groups, side=side)
        relevant_next = ~dead_in & ~dead_out
    return info


@torch.no_grad()
def _corr_pair(Zt, Zr):
    """Hungarian on pre-activation correlation: perm[i] = recon unit for teacher col i."""
    A = Zt - Zt.mean(0)
    B = Zr - Zr.mean(0)
    A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)
    B = B / B.norm(dim=0, keepdim=True).clamp_min(1e-12)
    C = (A.T @ B).double()
    rows, cols = linear_sum_assignment((-C).cpu().numpy())
    return torch.as_tensor(cols, device=Zt.device), C[rows, cols]


@torch.no_grad()
def sparse_align_errors(recon, teacher, X, **cls_kw):
    """Score recon vs teacher on probe X with the sparse-net quotient (module doc).
    Returns {'layers': [per hidden layer dict], 'fn_rel_err': float}."""
    assert len(teacher.layers) == 2, "sparse_align: one hidden layer only (deeper layers need the input frame aligned too)"
    t = teacher.clone().double()
    r = recon.clone().double()
    slope = float(getattr(t.act, "negative_slope", 0.01))
    X = X.double()
    info_t = classify_units(t, X, **cls_kw)
    info_r = classify_units(r, X, **cls_kw)
    zt, ft = _forward_preacts(t, X)
    zr, fr = _forward_preacts(r, X)
    out = []
    for l, it in enumerate(info_t):
        Wt, bt = t.layers[l].weight, t.layers[l].bias
        Wr, br = r.layers[l].weight, r.layers[l].bias
        Nt, Nr = t.layers[l + 1].weight, r.layers[l + 1].weight
        ti = it["ident"].nonzero(as_tuple=True)[0]
        rec = {k: int(it[k].sum()) for k in ("dead_in", "dead_out", "linear", "dup", "ident")}
        rec["n"] = Wt.shape[0]
        if len(ti):
            perm, corr = _corr_pair(zt[l][:, ti], zr[l])
            # direction on w ONLY: with |b| >> ||w|| (shrunk rows) a [w | b] angle
            # measures the bias ratio, not the row
            nt, nr = Wt[ti].norm(dim=1), Wr[perm].norm(dim=1).clamp_min(1e-300)
            Ut, Ur = Wt[ti] / nt[:, None], Wr[perm] / nr[:, None]
            ang = torch.rad2deg(torch.arccos((Ut * Ur).sum(1).clamp(-1, 1)))
            bias_err = (br[perm] / nr - bt[ti] / nt).abs()      # offset in units of ||w||
            # outgoing columns with the row scale pushed in; duplicate groups summed
            grp = {g[0]: g for g in it["groups"]}
            norm_t = Wt.norm(dim=1)
            ct = torch.stack([(Nt[:, g] * norm_t[g]).sum(1)
                              for g in (grp.get(int(i), [int(i)]) for i in ti)], 1)
            # recon side: sum the recon's own duplicate group of each paired unit
            gr = {}
            for g in info_r[l]["groups"]:
                for j in g:
                    gr[j] = g
            norm_r = Wr.norm(dim=1)
            cr = torch.stack([(Nr[:, g] * norm_r[g]).sum(1)
                              for g in (gr.get(int(j), [int(j)]) for j in perm)], 1)
            col_rel = (cr - ct).norm(dim=0) / ct.norm(dim=0).clamp_min(1e-300)
            rec.update(corr_min=corr.min().item(), corr_median=corr.median().item(),
                       angle_median=ang.median().item(), angle_max=ang.max().item(),
                       angle=ang.cpu(),
                       bias_err_median=bias_err.median().item(),
                       unit_eps_max=(Ut - Ur).abs().max().item(),
                       unit_eps_mean=(Ut - Ur).abs().mean().item(),
                       col_rel_median=col_rel.median().item(), col_rel_max=col_rel.max().item())
            used = torch.zeros(Wr.shape[0], dtype=torch.bool, device=X.device)
            used[perm] = True
            for j in perm:
                used[gr.get(int(j), [int(j)])] = True
        else:
            used = torch.zeros(Wr.shape[0], dtype=torch.bool, device=X.device)

        # affine part on X, into the output: linear units + dead-in constants act(b)
        # + output bias (the constants are interchangeable with the bias)
        def affine(W, b, N, bn, info, mask):
            s = torch.where(info["side"][mask] > 0, 1.0, slope).to(W.dtype)
            M = N[:, mask] @ (s[:, None] * torch.cat([W[mask], b[mask, None]], 1))
            di = info["dead_in"]
            M[:, -1] += bn + N[:, di] @ torch.nn.functional.leaky_relu(b[di], slope)
            return M
        Mt = affine(Wt, bt, Nt, t.layers[l + 1].bias, it, it["linear"])
        Mr = affine(Wr, br, Nr, r.layers[l + 1].bias, info_r[l], info_r[l]["linear"] & ~used)
        rec["n_linear_recon"] = int((info_r[l]["linear"] & ~used).sum())
        rec["affine_rel_err"] = ((Mr - Mt).norm() / Mt.norm().clamp_min(1e-300)).item()
        rec["recon_spurious_kinking"] = int((info_r[l]["ident"] & ~used).sum())
        out.append(rec)
    fc = ft - ft.mean(0)
    return {"layers": out, "fn_rel_err": ((fr - ft).norm() / fc.norm()).item()}


def probe_set(dims, device, n_data=10000, n_box=10000, buffer=None, n_buffer=20000, seed=0):
    """Probe domain X. With `buffer` (a run's query buffer): ONLY a random subset of
    it -- identifiability is about where the attacker actually queried, and images /
    U[-1,1]^d cross kinks the committee never does. Without: test images + U[-1,1]^d."""
    if buffer is not None:
        g = torch.Generator().manual_seed(seed)
        return buffer[torch.randperm(len(buffer), generator=g)[:n_buffer]].float().to(device)
    from data import load_data
    g = torch.Generator().manual_seed(seed)
    (_, _), (xte, _) = load_data(dims, "cpu")
    parts = [xte[torch.randperm(len(xte), generator=g)[:n_data]],
             torch.rand(n_box, dims[0], generator=g) * 2 - 1]
    return torch.cat(parts).to(device)
