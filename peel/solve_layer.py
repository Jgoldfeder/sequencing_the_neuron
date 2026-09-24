"""
solve_layer.py -- cryptanalytic refinement of ONE hidden layer of a deep sigmoid MLP.

============================  THE SETTING  ============================
We want the weights of weight-matrix `n` (0-indexed) of a black-box MLP.

Given:
  * BB(x):            the black box.  Queryable forward pass, input -> output.
  * a recovered solve of layers 0..n-1:  W_early=[W_0..W_{n-1}], b_early=[b_0..b_{n-1}].
                      (Only used to SEAL the network at layer n-1's activations.
                       May be imperfect -- the method degrades gracefully.)
  * a guess at layer n:   Wn_guess (d_n x d_{n-1}),  bn_guess (d_n).
                      (e.g. a committee/consensus estimate.)
  * the architecture `dims` = [d_in, d_1, d_2, ..., d_out].

REQUIREMENT (the funnel / rank certificate):  layer n must narrow into layer n+1,
                      i.e.  dims[n+2] < dims[n+1].   d_next < d_n.

============================  THE METHOD  ============================
Weight matrix n maps activation a_{n-1} (dim d_{n-1}=dims[n]) to pre-activation
z = W_n a_{n-1} + b_n, then a_n = sigmoid(z) (dim d_n = dims[n+1]).

SEAL the network at the a_{n-1} space.  Using the recovered early layers, invert
layers 0..n-1 to get an input that realises a target activation h = a_{n-1}:
        x_of_h(h) = pinv(W_0)( logit( ... pinv(W_{n-1})(logit(h)-b_{n-1}) ... ) - b_0 )
        BBh(h)    = BB(x_of_h(h))              # R^{d_{n-1}} -> R^{d_out}
Then                BBh(h) = tail( sigmoid( W_n h + b_n ) )
whose Jacobian is   J(h) = tail'(a_n) . diag(sigma'(W_n h + b_n)) . W_n .
Because weight n+1 has only d_next rows,  tail'(a_n) has rank <= d_next.

  1. ROWSPACE.  J(h) = (stuff) . W_n, so the rowspace of stacked J(h) IS W_n's
     rowspace.  Take B = top-d_n right singular vectors of stacked J  (orthonormal,
     d_n x d_{n-1}).  Parametrize  W_n = A @ B  with A an unknown d_n x d_n.

  2. UNDO THE SIGMOID.  With U = B h (=: A^{-1} z-ish) and D = diag(sigma'(A U + b_n)),
        K(h) = ( J(h) @ B^T ) @ inv(A) / D            # (d_out x d_n)
     At the TRUE (A, b_n):   J(h) B^T inv(A) = tail' diag(sigma') W_n B^T inv(A)
                                             = tail' diag(sigma')            (since W_n B^T = A)
     so  K(h) = tail'(a_n)  ==>  rank(K) <= d_next.

  3. CERTIFICATE.  Stack K over many probes -> (N*d_out, d_n).  At truth its
     bottom (d_n - d_next) singular values vanish.  Minimize that bottom singular
     energy over (A, b_n) by an alternating loose solver:
        - freeze N = bottom-(d_n-d_next) right-singular subspace of stacked K,
        - take an LM / Gauss-Newton step on (A,b_n) reducing || K . N ||^2,
        - accept iff the surrogate decreases; adapt damping.
     Return W_n = A @ B, b_n.

The early layers and the guess only need to be good enough to land in the basin;
their residual error sets a floor on how far (A,b_n) can be refined.
=====================================================================
"""
import torch
from torch.func import jvp, vjp

def logit(h): return torch.log(h / (1 - h))
def sig1(z):  s = torch.sigmoid(z); return s * (1 - s)

# ---- matrix-free damped least squares (LSQR) for the LM step ----
def lsqr(Aop, Atop, b, n, damp, iters=70):
    beta = b.norm()
    if float(beta) == 0: return torch.zeros(n, device=b.device, dtype=b.dtype)
    u = b / beta; v = Atop(u); al = v.norm(); v = v / al.clamp_min(1e-30)
    w = v.clone(); x = torch.zeros(n, device=b.device, dtype=b.dtype); pb = beta; rb = al
    for _ in range(iters):
        u = Aop(v) - al * u; beta = u.norm(); u = u / beta.clamp_min(1e-30)
        v = Atop(u) - beta * v; al = v.norm(); v = v / al.clamp_min(1e-30)
        r1 = (rb**2 + damp**2).sqrt(); c1 = rb / r1; pb = c1 * pb
        rho = (r1**2 + beta**2).sqrt(); cc = r1 / rho; sg = beta / rho
        th = sg * al; rb = -cc * al; phi = cc * pb; pb = sg * pb
        x = x + (phi / rho) * w; w = v - (th / rho) * w
    return x

# ---- build the sealed map BBh: a_{n-1} -> output, from recovered early layers ----
def build_seal(BB, W_early, b_early, n, act="sigmoid"):
    # ginv inverts the activation to recover a pre-activation from an activation.
    # sigmoid: a = sigmoid(z) => z = logit(a).   relu (active region a>0): z = a.
    ginv = (lambda a: logit(a)) if act == "sigmoid" else (lambda a: a)
    pinvs = [Wi.t() @ torch.linalg.inv(Wi @ Wi.t()) for Wi in W_early]   # d_{i-1} x d_i
    def x_of_h(h):
        a = h
        for i in range(n - 1, -1, -1):
            a = (ginv(a) - b_early[i]) @ pinvs[i].t()
        return a
    def BBh(h): return BB(x_of_h(h))
    return BBh, x_of_h

# ---- finite-diff Jacobian of the sealed map: (N, d_out, d_prev) ----
def sealed_jac(BBh, H, d_prev, d_out, fd=1e-4, chunk=150):
    N = H.shape[0]; E = torch.eye(d_prev, device=H.device) * fd; o = []
    for s in range(0, N, chunk):
        Hc = H[s:s+chunk]; m = Hc.shape[0]
        Hp = (Hc[:, None, :] + E[None]).reshape(-1, d_prev).clamp(1e-4, 1-1e-4)
        Hm = (Hc[:, None, :] - E[None]).reshape(-1, d_prev).clamp(1e-4, 1-1e-4)
        o.append(((BBh(Hp).reshape(m, d_prev, d_out) - BBh(Hm).reshape(m, d_prev, d_out)) / (2*fd)).permute(0, 2, 1))
    return torch.cat(o, 0)

def solve_layer(BB, W_early, b_early, Wn_guess, bn_guess, dims, n,
                n_probes=400, n_rowspace_probes=1000, iters=2000,
                seed=7, dev="cuda", score=None, log_every=250,
                fp32=False, fp64_finish=300, bias_steps=1,
                retarget_every=0, exc_thresh=0.05, probes_per_neuron=40, max_under=16,
                max_mult=4, refresh_every=8):
    """Refine weight matrix n. Returns (Wn_refined, bn_refined, info).

    retarget_every>0 enables AUTOMATIC targeted probing: every k iters, flag the
    under-excited neurons (teacher-free: peak sigma'(z_j) over the probes < exc_thresh
    -- they're saturated so the certificate is blind to them), mint probes on their
    current knees, and splice them into the probe set so the solver can pin them.
    The generic n_probes stay fixed; the targeted block is refreshed each retarget."""
    d_prev, d_n, d_next = dims[n], dims[n+1], dims[n+2]
    assert d_next < d_n, f"funnel condition violated: d_next({d_next}) !< d_n({d_n})"
    bottom = d_n - d_next
    BBh, _ = build_seal(BB, W_early, b_early, n)
    d_out = dims[-1]

    # 1) recover rowspace B from stacked sealed Jacobians
    gB = torch.Generator(device=dev).manual_seed(seed + 1)
    HB = torch.sigmoid(torch.randn(n_rowspace_probes, d_prev, generator=gB, device=dev) * 1.2).clamp(2e-2, 1-2e-2)
    B = torch.linalg.svd(sealed_jac(BBh, HB, d_prev, d_out).reshape(-1, d_prev), full_matrices=False)[2][:d_n]

    # 2) probe set for the certificate, precompute Q = J B^T and U = H B^T
    gh = torch.Generator(device=dev).manual_seed(seed)
    H = torch.sigmoid(torch.randn(n_probes, d_prev, generator=gh, device=dev) * 1.0).clamp(2e-2, 1-2e-2)
    U = H @ B.t()
    Q = sealed_jac(BBh, H, d_prev, d_out) @ B.t()                     # (N, d_out, d_n)
    # fp32 mixed precision: iterate in fp32 for ~2-3x cheaper einsums, then finish the
    # last fp64_finish iters in f64 to sink below the fp32 error floor (~0.2% werr).
    Q32 = Q.float() if fp32 else None
    U32 = U.float() if fp32 else None
    Qc, Uc = Q, U                                    # current-precision ingredients (Kof/analytic read these)
    NB = n_probes                                    # current probe count
    Qgen, Ugen = Q, U                                # fixed generic blocks
    tQ, tU = {}, {}                                  # per-neuron targeted probe blocks {j: tensor}
    gT = torch.Generator(device=dev).manual_seed(seed + 2); _rc = [0]

    def do_retarget(A, bn):
        """Adaptive targeted probing. Flag under-excited neurons (teacher-free), and give each
        one probes on its CURRENT knee -- MORE probes the more saturated it is (adaptive count,
        up to max_mult x), REPLACING that neuron's block (fresh knee). Every refresh_every
        retargets it also re-mints ALL targeted neurons onto their current knees, so an
        excited-but-slow row whose knee drifted a little isn't left pinned to a stale one.
        Per-neuron blocks (a dict) => no stale accumulation, no oscillation, bounded set."""
        nonlocal Q, U, Q32, U32, NB
        with torch.no_grad():
            _rc[0] += 1
            exc = sig1(U @ A.t() + bn).max(dim=0).values           # FULL-set excitation -> FLAG under-excited rows
            exc_gen = sig1(Ugen @ A.t() + bn).max(dim=0).values    # GENERIC-only -> MULTIPLIER: a row's inherent
                                                                    #   saturation (stays low even after it's targeted),
                                                                    #   so the boost persists through refreshes.
            under = [int(j) for j in torch.argsort(exc)[:max_under] if float(exc[j]) < exc_thresh]
            todo = set(under)
            if refresh_every > 0 and _rc[0] % refresh_every == 0:   # periodic knee refresh of all targeted
                todo |= set(tQ.keys())
            if not todo: return 0, 0
            Wcur = A @ B; added = 0
            for j in todo:
                mult = min(max_mult, max(1, round(exc_thresh / max(float(exc_gen[j]), exc_thresh / max_mult))))
                nj = probes_per_neuron * mult                       # adaptive: more saturated -> more probes
                wj = Wcur[j]; base = torch.sigmoid(torch.randn(nj, d_prev, generator=gT, device=dev) * 1.2)
                proj = (base - ((base @ wj + bn[j]) / (wj @ wj))[:, None] * wj[None, :]).clamp(2e-2, 1 - 2e-2)
                tU[j] = proj @ B.t(); tQ[j] = sealed_jac(BBh, proj, d_prev, d_out) @ B.t()   # REPLACE j's block (fresh knee)
                added += nj
            keys = list(tQ.keys())                                  # rebuild full set = generic + all per-neuron blocks
            Q = torch.cat([Qgen] + [tQ[j] for j in keys], 0)
            U = torch.cat([Ugen] + [tU[j] for j in keys], 0)
            NB = U.shape[0]
            if fp32: Q32 = Q.float(); U32 = U.float()
            return len(todo), added

    # parametrize from the guess:  A0 = Wn_guess B^T,   bn0 = bn_guess
    A = (Wn_guess @ B.t()).clone(); bn = bn_guess.clone()
    P = d_n * d_n
    p = torch.cat([A.reshape(-1), bn])

    def Kof(A, bn):
        # ---- THE HEART OF THE METHOD: "undo the sigmoid" to expose the rank ----
        # Recall the sealed Jacobian at a probe factors as   J = tail'(a_n) . diag(sigma'(z)) . W_n,
        # where z = W_n h + b_n is layer n's pre-activation and tail'(a_n) has rank <= d_next.
        # Q := J B^T = tail'(a_n) . diag(sigma'(z)) . A_true      (since W_n B^T = A_true).
        #
        # We try a candidate (A, bn). Two operations peel the factors off Q:
        #   * @ inv(A)                 -- cancels the trailing A_true IF A == A_true.
        #   * / sigma'(A U + bn)       -- cancels diag(sigma'(z)) IF (A,bn) == truth, because then
        #                                 A U + bn = A_true (B h) + b_true = z, so this reproduces
        #                                 the very diag that appears inside Q, and it divides out.
        # At the TRUE (A,bn) BOTH cancellations are exact and  K = tail'(a_n)  ==>  rank(K) <= d_next.
        # At a WRONG (A,bn) the per-probe, per-coordinate factor sigma'(A U + bn) no longer matches
        # the true diag(sigma'(z)) baked into Q, so the cancellation fails and K picks up full rank.
        # That mismatch is the entire signal we descend on.  (It is why a *curved* activation is
        # required: sigma' varies with z. For ReLU, sigma'=1 on the active region, nothing to cancel,
        # and K stays low-rank for EVERY A -- the certificate goes blind. See relu_vs_sigmoid test.)
        Dp = sig1(Uc @ A.t() + bn)                                  # sigma'(candidate pre-activations)
        return torch.einsum('nok,kj->noj', Qc, torch.linalg.inv(A)) / Dp[:, None, :]
    def resid(pp, Nm):
        # residual we drive to zero: K projected onto its bottom-(d_n-d_next) singular subspace Nm.
        # ||K . Nm|| == 0  <=>  K has rank <= d_next  <=>  (A,bn) hit the true weights (up to the
        # ingredient-quality floor). Nm is frozen from the current K each outer step (see below).
        A = pp[:P].reshape(d_n, d_n); bn = pp[P:]
        return torch.einsum('noj,jm->nom', Kof(A, bn), Nm).reshape(-1)

    # ---- alternating "loose" solver ----------------------------------------------------------
    # The exact objective ||bottom singular energy of K(A,bn)|| has a subtlety: its true gradient
    # is (near-)orthogonal to the weight error, so plain gradient descent on it stalls. Instead we
    # ALTERNATE: (1) freeze Nm = the current bottom singular subspace, turning the objective into a
    # plain nonlinear least-squares ||K.Nm||=0; (2) take one damped Gauss-Newton (Levenberg-Marquardt)
    # step on (A,bn) toward that frozen target; accept only if the residual actually drops, else raise
    # damping. Re-extracting Nm each step lets the target track the solution inward. The GN step is
    # solved matrix-free by LSQR on the Jacobian-vector products (Jv/Jt), so we never form the Jacobian.
    lam = 1e-8; hist = []
    switch_at = (iters - fp64_finish) if fp32 else 0        # iters < switch_at run in fp32
    for it in range(iters + 1):
        use32 = fp32 and it < switch_at
        want = torch.float32 if use32 else torch.float64
        if p.dtype != want:                                 # cast at phase boundary (once)
            p = p.to(want)
            if score is not None: print(f"  [precision] it{it}: -> {'fp32' if use32 else 'fp64'}", flush=True)
        if retarget_every > 0 and it % retarget_every == 0:  # AUTOMATIC targeted probing
            nu, na = do_retarget(p[:P].reshape(d_n, d_n).to(B.dtype), p[P:].to(B.dtype))
            if score is not None and na:
                print(f"  [retarget] it{it}: {nu} rows (re)targeted -> +{na} probes (NB={NB})", flush=True)
        Qc, Uc = (Q32, U32) if use32 else (Q, U)
        A = p[:P].reshape(d_n, d_n); bn = p[P:]
        if score is not None and (it % log_every == 0):
            s = score(A.to(B.dtype) @ B, bn.to(B.dtype)); hist.append((it, s)); print(f"  it{it:5d}: {s}", flush=True)
        with torch.no_grad():
            # freeze the current bottom-(d_n - d_next) right-singular subspace of K as the LSQ target
            _, _, Vh = torch.linalg.svd(Kof(A, bn).reshape(NB * d_out, d_n), full_matrices=True)
            Nm = Vh[d_next:].t().contiguous()
        r0 = resid(p, Nm); bb = float(r0 @ r0)
        # ANALYTIC matrix-free J, J^T (exact; ~2x faster than autograd jvp/vjp).
        # Precompute the pieces once per outer step (Nm is frozen); the residual is
        #   r = (Q A^-1 / D) . Nm ,  D = sigma'(A U^T + bn).
        # dr uses d(A^-1) = -A^-1 dA A^-1 and dD = sigma''(z) dz, dz = dA U^T + db.
        Ai = torch.linalg.inv(A)
        Mm = torch.einsum('nok,kj->noj', Qc, Ai)                # Q_i A^-1
        z_ = Uc @ A.t() + bn; s_ = torch.sigmoid(z_)
        D_ = s_ * (1 - s_); Dpp = D_ * (1 - 2 * s_); D2 = D_ * D_
        mN = Nm.shape[1]
        def Jv(v):
            dA = v[:P].reshape(d_n, d_n); db = v[P:]
            dM = -torch.einsum('noa,ab,bj->noj', Mm, dA, Ai)
            T1 = torch.einsum('noj,jk->nok', dM / D_[:, None, :], Nm)
            dz = Uc @ dA.t() + db
            T2 = -torch.einsum('noj,jk->nok', Mm * (Dpp * dz / D2)[:, None, :], Nm)
            return (T1 + T2).reshape(-1)
        def Jt(uvec):
            Ub = uvec.reshape(NB, d_out, mN)
            P1 = torch.einsum('nok,jk->noj', Ub, Nm)
            R = torch.einsum('noj,bj->nob', P1 / D_[:, None, :], Ai)
            Ab = -torch.einsum('noa,nob->ab', Mm, R)
            C = -torch.einsum('noj,noj->nj', P1, Mm) * (Dpp / D2)
            Ab = Ab + torch.einsum('nj,nc->jc', C, Uc)
            return torch.cat([Ab.reshape(-1), C.sum(0)])
        ok = False
        for _ in range(6):                                     # LM backtracking on the damping lam
            d = lsqr(Jv, Jt, -r0, P + d_n, lam); pn = p + d
            if float(resid(pn, Nm) @ resid(pn, Nm)) < bb: p = pn; lam = max(lam/3, 1e-16); ok = True; break
            lam *= 5
        if not ok:
            if fp32 and use32:                  # fp32 hit its floor early -> drop to fp64 and continue
                switch_at = it + 1; lam = 1e-8
                if score is not None: print(f"  [precision] it{it}: fp32 stalled -> fp64 finish", flush=True)
                continue
            print(f"  converged/no-progress at it{it}", flush=True); break
        # --- BIAS STAGE (alternating) -- THE SAME loose trick as the weights ------------------
        # Freeze N and take a damped Gauss-Newton (LM) step on bn ALONE against ||K.N||^2 (the
        # exact residual, NOT the bottom-energy ratio). The ratio has a blow-up escape -- a large
        # bn saturates sigma' -> 0, one column of K explodes, and the bottom-energy FRACTION goes
        # to ~0 without touching the true bias, so Adam runs the bias off to infinity. ||K.N||^2
        # has no such escape (inflating bn RAISES it), and a step is accepted only if it truly
        # drops. Solving for bn alone (A excluded) gives the bias its own full least-squares step
        # -- the joint weight+bias step drowns it out, which is why bn never moved before.
        for _ in range(bias_steps):
            A = p[:P].reshape(d_n, d_n); bn = p[P:]
            with torch.no_grad():
                _, _, Vb = torch.linalg.svd(Kof(A, bn).reshape(NB * d_out, d_n), full_matrices=True)
                Nb = Vb[d_next:].t().contiguous()
            Aib = torch.linalg.inv(A); Mb = torch.einsum('nok,kj->noj', Qc, Aib)
            sb = torch.sigmoid(Uc @ A.t() + bn); Db = sb * (1 - sb); Dppb = Db * (1 - 2 * sb); D2b = Db * Db
            mb = Nb.shape[1]
            def JvB(db):                                   # Jv restricted to a pure-bias direction (dA=0)
                return (-torch.einsum('noj,jk->nok', Mb * (Dppb * db / D2b)[:, None, :], Nb)).reshape(-1)
            def JtB(uv):                                   # bias part of the adjoint J^T
                Ub = uv.reshape(NB, d_out, mb); P1 = torch.einsum('nok,jk->noj', Ub, Nb)
                return (-torch.einsum('noj,noj->nj', P1, Mb) * (Dppb / D2b)).sum(0)
            rb = resid(p, Nb); bbb = float(rb @ rb); lam_b = 1e-8; okb = False
            for _ in range(6):                             # LM backtracking on the bias step
                db = lsqr(JvB, JtB, -rb, d_n, lam_b); pn = p.clone(); pn[P:] = p[P:] + db
                if float(resid(pn, Nb) @ resid(pn, Nb)) < bbb: p = pn; okb = True; break
                lam_b *= 5
            if not okb: break
    A = p[:P].reshape(d_n, d_n).to(B.dtype); bn = p[P:].to(B.dtype)   # back to f64 for output
    return A @ B, bn, {"B": B, "A": A, "hist": hist}


# ================================  DEMO  ================================
if __name__ == "__main__":
    import sys
    from scipy.optimize import linear_sum_assignment
    torch.set_default_dtype(torch.float64); dev = "cuda"
    sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
    from nets import MLP
    pk = torch.load("peel_committee.pt", map_location=dev, weights_only=False); dims = pk["dims"]
    tt = MLP(dims, act="sigmoid").to(dev).double()
    tt.load_state_dict({k: v.double() for k, v in pk["teacher_state"].items()}); tt.eval()
    def BB(x):
        with torch.no_grad(): return tt(x)

    n = 1  # solve weight matrix 1 (the 128 -> 80 map, "W2"); next layer is 80 -> 40 (d_next=40 < d_n=80)
    # recovered layers 0..n-1 = [layer 0]:  exact W0 from committee, refined bias b0 from prior bias-certificate run
    W0r = pk["pop_states"][0]["layers.0.weight"].to(dev).double()
    b0r = torch.load("rank40_fullsealed_traj.pt", map_location=dev, weights_only=False)["b1r"].to(dev).double()  # recovered layer-0 bias
    # guess at layer n = consensus estimate
    cf = torch.load("consensus_full.pt", map_location=dev, weights_only=False)
    Wn_guess = cf["consensus"]["layers.1.weight"].double(); bn_guess = cf["consensus"]["layers.1.bias"].double()

    # oracle scorer (DEMO ONLY): mean per-neuron relative weight error under best sign-matching + Hungarian
    W2s = tt.layers[1].weight.detach(); nt = W2s.norm(dim=1)
    def score(Wn, bn):
        Cp = torch.cdist(Wn, W2s); Cm = torch.cdist(-Wn, W2s); C = torch.minimum(Cp, Cm).cpu().numpy()
        ri, ci = linear_sum_assignment(C)
        return f"werr {float((torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))])/nt[ci].cpu()).mean())*100:.4f}%"

    print(f"[demo] solving weight-matrix n={n}: {dims[n]}->{dims[n+1]}, next layer {dims[n+1]}->{dims[n+2]} (rank cert = {dims[n+2]})", flush=True)
    Wn, bn, info = solve_layer(BB, [W0r], [b0r], Wn_guess, bn_guess, dims, n,
                               iters=500, dev=dev, score=score)
    print("[demo] done. final:", score(Wn, bn), flush=True)
