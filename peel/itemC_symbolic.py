import sympy as sp
s,w2,w3,P,Q,a=sp.symbols('s w2 w3 P Q a')
R2,R3,R4,R5,R6=sp.symbols('R2 R3 R4 R5 R6'); x=sp.symbols('x')
cc=[sp.Integer(1),1-2*x,1-6*x+6*x**2,1-14*x+36*x**2-24*x**3,
    1-30*x+150*x**2-240*x**3+120*x**4,1-62*x+540*x**2-1560*x**3+1800*x**4-720*x**5]
cn=lambda k,val: cc[k-1].subs(x,val)
uu=[w3*w2**k*P*(1-P)*cn(k,P) for k in range(1,7)]
QQ=[Q*(1-Q)*cn(k,Q) for k in range(1,7)]
Qs=list(sp.symbols('Qd1:7')); us=list(sp.symbols('ud1:7'))
def Dop(e):
    r=0
    for i in range(5):
        r+=sp.diff(e,Qs[i])*Qs[i+1]*us[0]+sp.diff(e,us[i])*us[i+1]
    return r
H=[None,Qs[0]*us[0]]
for n in range(2,7): H.append(sp.expand(Dop(H[-1])))
sub={Qs[i]:QQ[i] for i in range(6)}; sub.update({us[i]:uu[i] for i in range(6)})
Hs=[None]+[sp.expand(H[n].subs(sub)) for n in range(1,7)]
sd=[None]+[a**k*s*(1-s)*cn(k,s) for k in range(1,7)]
Hd=list(sp.symbols('Hd1:7')); sds=list(sp.symbols('sdd1:7'))
def Dt(e):
    r=0
    for i in range(5):
        r+=sp.diff(e,Hd[i])*Hd[i+1]*sds[0]+sp.diff(e,sds[i])*sds[i+1]
    return r
F=[None,Hd[0]*sds[0]]
for n in range(2,7): F.append(sp.expand(Dt(F[-1])))
sub2={Hd[i]:Hs[i+1] for i in range(6)}; sub2.update({sds[i]:sd[i+1] for i in range(6)})
Fe=[None]+[sp.expand(F[n].subs(sub2)) for n in range(1,7)]
Rsym=[None,None,R2,R3,R4,R5,R6]
eqs=[sp.expand(Rsym[n]*Fe[1]-Fe[n]) for n in range(2,7)]
print("built equations; total degrees:", [sp.total_degree(e) for e in eqs])
print("eliminating w2,w3,P,Q ... (lex Groebner)")
G=sp.groebner(eqs, w2,w3,P,Q,s, order='lex')
inv=[g for g in G.polys if s in g.free_symbols and not any(v in g.free_symbols for v in (w2,w3,P,Q))]
if inv:
    poly=inv[0].as_expr()
    print("FOUND relation in s (deep 1->1->1->1 invariant):")
    print("  degree in s:", sp.Poly(poly,s).degree(), " uses R2..R6:", {R2,R3,R4,R5,R6}&poly.free_symbols)
else:
    print("no s-only relation isolated; basis tail vars:")
    for g in list(G.polys)[-3:]: print("  ", g.free_symbols)
