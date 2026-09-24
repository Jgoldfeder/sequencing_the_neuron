"""Derive the single-coordinate JET INVARIANT for a deep downstream and TEST it
numerically. Deep case F(t)=v*sig(w*sig(a t+b)+c)+d (1->1->1 downstream, H nonlinear).
Groebner elimination of the downstream (w,P) from R2=F''/F',R3=F'''/F',R4=F''''/F'
gives a cubic in s=sig(a t+b). Solve -> s -> b, NO downstream fit. Does it work?"""
import sympy as sp, torch, math
torch.set_default_dtype(torch.float64)

# ---- 1. symbolic invariant ----
s,P,w,a,R2,R3,R4 = sp.symbols('s P w a R2 R3 R4')
s1=a*s*(1-s); s2=a**2*s*(1-s)*(1-2*s); s3=a**3*s*(1-s)*(1-6*s+6*s**2)
s4=a**4*s*(1-s)*(1-14*s+36*s**2-24*s**3)
A=(1-2*P)*w; B=(1-6*P+6*P**2)*w**2; C=(1-14*P+36*P**2-24*P**3)*w**3
e2=R2*s1-(s2+A*s1**2)
e3=R3*s1-(s3+3*A*s1*s2+B*s1**3)
e4=R4*s1-(s4+(4*A*s1*s3+3*A*s2**2)+6*B*s1**2*s2+C*s1**4)
G=sp.groebner([sp.expand(e2),sp.expand(e3),sp.expand(e4)], w,P,s, order='lex')
inv=[g.as_expr() for g in G.polys if s in g.free_symbols and w not in g.free_symbols and P not in g.free_symbols][0]
poly_s=sp.Poly(sp.expand(inv),s)
print("invariant degree in s:",poly_s.degree()," coeffs depend on:",inv.free_symbols)
f_inv=sp.lambdify((s,R2,R3,R4,a), inv, 'numpy')          # residual(s)
coeffs=[sp.lambdify((R2,R3,R4,a),c,'numpy') for c in poly_s.all_coeffs()]

# ---- 2. build a random deep 1->1->1 net ----
g=torch.Generator().manual_seed(7)
def rp(): return (2*torch.rand(1,generator=g)-1).item()
a_t=1.3+0.5*rp(); b_t=0.4*rp(); w_t=1.7*rp(); c_t=0.8*rp(); v_t=1.5*rp(); d_t=0.6*rp()
def F(t):  # scalar->scalar deep net
    return v_t*torch.sigmoid(w_t*torch.sigmoid(a_t*t+b_t)+c_t)+d_t

# ---- 3. exact derivatives via autodiff at a probe point ----
def derivs(t0):
    t=torch.tensor(t0,requires_grad=True)
    d1=torch.autograd.grad(F(t),t,create_graph=True)[0]
    d2=torch.autograd.grad(d1,t,create_graph=True)[0]
    d3=torch.autograd.grad(d2,t,create_graph=True)[0]
    d4=torch.autograd.grad(d3,t,create_graph=True)[0]
    return [float(x) for x in (d1,d2,d3,d4)]

import numpy as np
print(f"\ntrue: a={a_t:.4f} b={b_t:.4f}   (recover b from jet at several probe points)")
for t0 in (-0.7,0.0,0.6,1.2):
    d1,d2,d3,d4=derivs(t0)
    r2,r3,r4=d2/d1,d3/d1,d4/d1
    cs=[c(r2,r3,r4,a_t) for c in coeffs]
    roots=np.roots(cs)
    roots=[rt.real for rt in roots if abs(rt.imag)<1e-9 and 1e-6<rt.real<1-1e-6]
    # true s at this t
    s_true=1/(1+math.exp(-(a_t*t0+b_t)))
    best=min(roots,key=lambda rr:abs(rr-s_true)) if roots else float('nan')
    b_rec=math.log(best/(1-best))-a_t*t0 if roots else float('nan')
    print(f"  t={t0:+.2f}: s_true={s_true:.5f} roots={[f'{r:.4f}' for r in roots]} "
          f"b_rec={b_rec:+.6f} err={abs(b_rec-b_t):.2e}")
