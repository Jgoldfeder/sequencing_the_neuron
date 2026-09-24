"""Analytic Jv / J^T u for the certificate residual, vs autograd. Verifies
correctness (adjointness + match to torch.func) then times it."""
import sys, time, torch
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from torch.func import jvp, vjp
torch.set_default_dtype(torch.float64); dev = "cuda"

# synthetic but structurally identical problem (L1 sizes)
N, d_out, d_n, d_next = 400, 10, 128, 80
m = d_n - d_next
g = torch.Generator(device=dev).manual_seed(0)
Q = torch.randn(N, d_out, d_n, generator=g, device=dev)
U = torch.randn(N, d_n, generator=g, device=dev) * 0.3
Nm = torch.linalg.qr(torch.randn(d_n, m, generator=g, device=dev))[0]
A = torch.eye(d_n, device=dev) + 0.1 * torch.randn(d_n, d_n, generator=g, device=dev)
bn = torch.randn(d_n, generator=g, device=dev) * 0.1
P = d_n * d_n
p = torch.cat([A.reshape(-1), bn])

def sig1(z): s = torch.sigmoid(z); return s * (1 - s)
def Kof(A, bn):
    Dp = sig1(U @ A.t() + bn)
    return torch.einsum('nok,kj->noj', Q, torch.linalg.inv(A)) / Dp[:, None, :]
def resid(pp):
    A = pp[:P].reshape(d_n, d_n); bn = pp[P:]
    return torch.einsum('noj,jm->nom', Kof(A, bn), Nm).reshape(-1)

# ---- analytic: precompute once per (A,bn) ----
def precompute(A, bn):
    Ai = torch.linalg.inv(A)
    M = torch.einsum('nok,kj->noj', Q, Ai)            # Q_i A^{-1}
    z = U @ A.t() + bn
    s = torch.sigmoid(z); D = s * (1 - s); Dpp = D * (1 - 2 * s)
    return Ai, M, D, Dpp, D * D
def Jv_analytic(v, pre):
    Ai, M, D, Dpp, D2 = pre
    dA = v[:P].reshape(d_n, d_n); db = v[P:]
    dM = -torch.einsum('noa,ab,bj->noj', M, dA, Ai)
    Term1 = torch.einsum('noj,jk->nok', dM / D[:, None, :], Nm)
    dz = U @ dA.t() + db
    MG = M * (Dpp * dz / D2)[:, None, :]
    Term2 = -torch.einsum('noj,jk->nok', MG, Nm)
    return (Term1 + Term2).reshape(-1)
def Jt_analytic(uvec, pre):
    Ai, M, D, Dpp, D2 = pre
    Ubar = uvec.reshape(N, d_out, m)
    P1 = torch.einsum('nok,jk->noj', Ubar, Nm)
    R = torch.einsum('noj,bj->nob', P1 / D[:, None, :], Ai)
    Abar1 = -torch.einsum('noa,nob->ab', M, R)
    T = torch.einsum('noj,noj->nj', P1, M)
    C = -T * (Dpp / D2)
    Abar = Abar1 + torch.einsum('nj,nc->jc', C, U)
    bbar = C.sum(0)
    return torch.cat([Abar.reshape(-1), bbar])

# ---- correctness ----
f = lambda q: resid(q)
v = torch.randn(P + d_n, generator=g, device=dev)
u = torch.randn(N * d_out * m, generator=g, device=dev)
pre = precompute(A, bn)
jv_auto = jvp(f, (p,), (v,))[1]
jv_ana = Jv_analytic(v, pre)
jt_auto = vjp(f, p)[1](u)[0]
jt_ana = Jt_analytic(u, pre)
print(f"Jv   match (rel err): {(jv_ana-jv_auto).norm()/jv_auto.norm():.2e}")
print(f"J^Tu match (rel err): {(jt_ana-jt_auto).norm()/jt_auto.norm():.2e}")
# adjointness of the analytic pair: <u, Jv> == <J^Tu, v>
lhs = float(u @ jv_ana); rhs = float(jt_ana @ v)
print(f"adjointness <u,Jv> vs <J^Tu,v>: {abs(lhs-rhs)/abs(lhs):.2e}")

# ---- speed: 70 Jv+Jt calls (one lsqr's worth), autograd vs analytic ----
def bench(fn, k=70):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(k): fn()
    torch.cuda.synchronize(); return time.time() - t
Jv_a = lambda: jvp(f, (p,), (v,))[1]
Jt_a = lambda: vjp(f, p)[1](u)[0]
pre = precompute(A, bn)
Jv_n = lambda: Jv_analytic(v, pre)
Jt_n = lambda: Jt_analytic(u, pre)
for nm2, fn in [("autograd Jv", Jv_a), ("analytic Jv", Jv_n),
                ("autograd Jt", Jt_a), ("analytic Jt", Jt_n)]:
    fn()  # warm
    print(f"  {nm2}: {bench(fn)*1000:.1f} ms / 70 calls")
# full lsqr-equivalent: precompute + 70*(Jv+Jt)
def auto_round():
    for _ in range(70): jvp(f, (p,), (v,))[1]; vjp(f, p)[1](u)[0]
def ana_round():
    pr = precompute(A, bn)
    for _ in range(70): Jv_analytic(v, pr); Jt_analytic(u, pr)
auto_round(); ana_round()
ta = bench(auto_round, 1); tn = bench(ana_round, 1)
print(f"\nfull lsqr step (precompute + 70x(Jv+Jt)):  autograd {ta*1000:.0f} ms  ->  analytic {tn*1000:.0f} ms  ({ta/tn:.1f}x)")
