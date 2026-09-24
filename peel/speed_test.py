"""A/B test: does reducing lsqr iters / fp32 / momentum speed up solve_layer?
Setup (B, Q, U) is computed ONCE; each config runs the SAME iteration from the SAME
init, tracking werr vs iteration AND wall-clock."""
import sys, time, torch
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from scipy.optimize import linear_sum_assignment
from torch.func import jvp, vjp
from nets import MLP
from data import make_teacher
from solve_layer import build_seal, sealed_jac
torch.set_default_dtype(torch.float64); dev = "cuda"

ck = torch.load("recon/mergedbest512_sigmoid__784x128x80x40x32x10__s0_consensus.pt",
                map_location=dev, weights_only=False)
dims = ck["dims"]; d_prev, d_n, d_next = dims[0], dims[1], dims[2]; d_out = dims[-1]
Wc = ck["consensus_state"]["layers.0.weight"].to(dev).double()
bc = ck["consensus_state"]["layers.0.bias"].to(dev).double()
teacher = make_teacher(dims, epochs=25, seed=0, device=dev, act="sigmoid").to(dev).double(); teacher.eval()
def BB(x):
    with torch.no_grad(): return teacher(x)
Wt = teacher.layers[0].weight.detach(); nt = Wt.norm(dim=1)

# ---- setup ONCE (f64) ----
print("[setup] recovering B, Q, U ...", flush=True)
BBh, _ = build_seal(BB, [], [], 0)
gB = torch.Generator(device=dev).manual_seed(8)
HB = torch.sigmoid(torch.randn(1000, d_prev, generator=gB, device=dev) * 1.2).clamp(2e-2, 1-2e-2)
B = torch.linalg.svd(sealed_jac(BBh, HB, d_prev, d_out).reshape(-1, d_prev), full_matrices=False)[2][:d_n]
gh = torch.Generator(device=dev).manual_seed(7)
H = torch.sigmoid(torch.randn(400, d_prev, generator=gh, device=dev) * 1.0).clamp(2e-2, 1-2e-2)
U0 = H @ B.t(); Q0 = sealed_jac(BBh, H, d_prev, d_out) @ B.t()
A0 = (Wc @ B.t()).clone(); P = d_n * d_n
print("[setup] done.\n", flush=True)

def werr(A):
    W = (A.double() @ B); Cp = torch.cdist(W, Wt); Cm = torch.cdist(-W, Wt)
    C = torch.minimum(Cp, Cm).cpu().numpy(); ri, ci = linear_sum_assignment(C)
    return float((torch.tensor([C[ri[i], ci[i]] for i in range(len(ri))]) / nt[ci].cpu()).mean())

def sig1(z): s = torch.sigmoid(z); return s * (1 - s)

def run(tag, dt, lsqr_iters, beta, iters=200, analytic=False):
    Q = Q0.to(dt); U = U0.to(dt)
    p = torch.cat([A0.reshape(-1), bc]).to(dt); p_prev = p.clone()
    def Kof(A, bn):
        Dp = sig1(U @ A.t() + bn); return torch.einsum('nok,kj->noj', Q, torch.linalg.inv(A)) / Dp[:, None, :]
    def resid(pp, Nm):
        A = pp[:P].reshape(d_n, d_n); bn = pp[P:]
        return torch.einsum('noj,jm->nom', Kof(A, bn), Nm).reshape(-1)
    def lsqr(Aop, Atop, b, nn, damp):
        beta_ = b.norm()
        if float(beta_) == 0: return torch.zeros(nn, device=b.device, dtype=b.dtype)
        u = b/beta_; v = Atop(u); al = v.norm(); v = v/al.clamp_min(1e-30); w = v.clone()
        x = torch.zeros(nn, device=b.device, dtype=b.dtype); pb = beta_; rb = al
        for _ in range(lsqr_iters):
            u = Aop(v)-al*u; beta_ = u.norm(); u = u/beta_.clamp_min(1e-30)
            v = Atop(u)-beta_*v; al = v.norm(); v = v/al.clamp_min(1e-30)
            r1 = (rb**2+damp**2).sqrt(); c1 = rb/r1; pb = c1*pb; rho = (r1**2+beta_**2).sqrt()
            cc = r1/rho; sg = beta_/rho; th = sg*al; rb = -cc*al; phi = cc*pb; pb = sg*pb
            x = x + (phi/rho)*w; w = v - (th/rho)*w
        return x
    lam = 1e-8; traj = []; t0 = time.time()
    for it in range(iters + 1):
        A = p[:P].reshape(d_n, d_n); bn = p[P:]
        if it % 40 == 0:
            traj.append((it, time.time()-t0, werr(A)))
        with torch.no_grad():
            _, _, Vh = torch.linalg.svd(Kof(A, bn).reshape(400*d_out, d_n), full_matrices=True)
            Nm = Vh[d_next:].t().contiguous()
        r0 = resid(p, Nm); bb = float(r0 @ r0)
        if analytic:
            Ai = torch.linalg.inv(A); Mm = torch.einsum('nok,kj->noj', Q, Ai)
            z_ = U @ A.t() + bn; s_ = torch.sigmoid(z_)
            D_ = s_*(1-s_); Dpp = D_*(1-2*s_); D2 = D_*D_; mN = Nm.shape[1]
            def Jv(v):
                dA = v[:P].reshape(d_n, d_n); db = v[P:]
                dM = -torch.einsum('noa,ab,bj->noj', Mm, dA, Ai)
                T1 = torch.einsum('noj,jk->nok', dM/D_[:,None,:], Nm)
                T2 = -torch.einsum('noj,jk->nok', Mm*(Dpp*(U@dA.t()+db)/D2)[:,None,:], Nm)
                return (T1+T2).reshape(-1)
            def Jt(uvec):
                Ub = uvec.reshape(400, d_out, mN); P1 = torch.einsum('nok,jk->noj', Ub, Nm)
                Ab = -torch.einsum('noa,nob->ab', Mm, torch.einsum('noj,bj->nob', P1/D_[:,None,:], Ai))
                C = -torch.einsum('noj,noj->nj', P1, Mm)*(Dpp/D2)
                return torch.cat([(Ab+torch.einsum('nj,nc->jc', C, U)).reshape(-1), C.sum(0)])
        else:
            f = lambda q: resid(q, Nm); Jv = lambda v: jvp(f, (p,), (v,))[1]; Jt = lambda u: vjp(f, p)[1](u)[0]
        ok = False
        for _ in range(6):
            d = lsqr(Jv, Jt, -r0, P + d_n, lam); pn = p + d
            if float(resid(pn, Nm) @ resid(pn, Nm)) < bb:
                if beta > 0: pn = pn + beta * (p - p_prev)      # heavy-ball momentum
                p_prev = p; p = pn; lam = max(lam/3, 1e-16); ok = True; break
            lam *= 5
        if not ok: break
    traj.append((iters, time.time()-t0, werr(p[:P].reshape(d_n, d_n))))
    dtn = "f32" if dt == torch.float32 else "f64"
    print(f"=== {tag}  ({dtn}, lsqr={lsqr_iters}, momentum={beta}) ===", flush=True)
    for it, tt, w in traj:
        print(f"    it{it:4d}  {tt:6.1f}s  werr {w*100:.4f}%", flush=True)
    return traj

ITERS = 200
run("baseline autograd", torch.float64, 70, 0.0, ITERS, analytic=False)
run("ANALYTIC f64",      torch.float64, 70, 0.0, ITERS, analytic=True)
run("ANALYTIC fp32",     torch.float32, 70, 0.0, ITERS, analytic=True)
print("DONE", flush=True)
