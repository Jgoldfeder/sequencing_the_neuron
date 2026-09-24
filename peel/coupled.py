import sympy as sp, numpy as np
from scipy.optimize import least_squares
s,w2,w3,P,Q,a=sp.symbols('s w2 w3 P Q a'); x=sp.symbols('x')
cc=[sp.Integer(1),1-2*x,1-6*x+6*x**2,1-14*x+36*x**2-24*x**3,1-30*x+150*x**2-240*x**3+120*x**4,1-62*x+540*x**2-1560*x**3+1800*x**4-720*x**5]
cn=lambda k,val: cc[k-1].subs(x,val)
uu=[w3*w2**k*P*(1-P)*cn(k,P) for k in range(1,7)]; QQ=[Q*(1-Q)*cn(k,Q) for k in range(1,7)]
Qs=list(sp.symbols('Qd1:7')); us=list(sp.symbols('ud1:7'))
Dop=lambda e: sum(sp.diff(e,Qs[i])*Qs[i+1]*us[0]+sp.diff(e,us[i])*us[i+1] for i in range(5))
H=[None,Qs[0]*us[0]]
for n in range(2,7): H.append(sp.expand(Dop(H[-1])))
sub={Qs[i]:QQ[i] for i in range(6)}; sub.update({us[i]:uu[i] for i in range(6)})
Hs=[None]+[H[n].subs(sub) for n in range(1,7)]
sd=[None]+[a**k*s*(1-s)*cn(k,s) for k in range(1,7)]
Hd=list(sp.symbols('Hd1:7')); sds=list(sp.symbols('sdd1:7'))
Dt=lambda e: sum(sp.diff(e,Hd[i])*Hd[i+1]*sds[0]+sp.diff(e,sds[i])*sds[i+1] for i in range(5))
Fl=[None,Hd[0]*sds[0]]
for n in range(2,7): Fl.append(sp.expand(Dt(Fl[-1])))
sub2={Hd[i]:Hs[i+1] for i in range(6)}; sub2.update({sds[i]:sd[i+1] for i in range(6)})
Fe=[None]+[Fl[n].subs(sub2) for n in range(1,7)]
Ff=[None]+[sp.lambdify((s,w2,w3,P,Q,a),Fe[n],'numpy') for n in range(1,7)]
sig=lambda z:1/(1+np.exp(-z)); logit=lambda p:np.log(p/(1-p))
print('built')
def run(seed, Pn=3):
    rng=np.random.default_rng(seed); av=1.0
    bt=rng.uniform(-0.6,0.6); w2t=rng.uniform(-2,2); c2=rng.uniform(-1,1); w3t=rng.uniform(-2,2); c3=rng.uniform(-1,1)
    tps=np.linspace(-0.7,0.7,Pn)+rng.uniform(-0.1,0.1,Pn)
    R=[]; 
    for tp in tps:
        st=sig(av*tp+bt); Pt=sig(w2t*st+c2); Qt=sig(w3t*Pt+c3)
        F1=Ff[1](st,w2t,w3t,Pt,Qt,av); R.append([None,None]+[Ff[n](st,w2t,w3t,Pt,Qt,av)/F1 for n in range(2,7)])
    lam=[np.exp(av*(tps[p]-tps[0])) for p in range(Pn)]
    def resid(v):
        w2v,w3v=v[0],v[1]; sv=v[2:2+Pn]; Pv=v[2+Pn:2+2*Pn]; Qv=v[2+2*Pn:2+3*Pn]; r=[]
        for p in range(Pn):
            F1=Ff[1](sv[p],w2v,w3v,Pv[p],Qv[p],av)
            for n in range(2,7): r.append(R[p][n]*F1-Ff[n](sv[p],w2v,w3v,Pv[p],Qv[p],av))
        for p in range(1,Pn): r.append(sv[p]*(1-sv[0])-lam[p]*sv[0]*(1-sv[p]))
        return r
    roots=[]
    for i in range(100):
        r=np.random.default_rng(seed*9999+i+1)
        x0=np.concatenate([r.uniform(-3,3,2), r.uniform(0.05,0.95,Pn), r.uniform(0.05,0.95,Pn), r.uniform(0.05,0.95,Pn)])
        try: sol=least_squares(resid,x0,method='lm',max_nfev=400,xtol=1e-14,ftol=1e-14)
        except Exception: continue
        if sol.cost<1e-16:
            sv=sol.x[2:2+Pn]; Pv=sol.x[2+Pn:2+2*Pn]; Qv=sol.x[2+2*Pn:2+3*Pn]
            if np.all(sv>1e-3)&np.all(sv<1-1e-3)&np.all(Pv>0)&np.all(Pv<1)&np.all(Qv>0)&np.all(Qv<1):
                roots.append(sol.x[2])   # s_1
    u=[]
    for v in sorted(roots):
        if not u or abs(v-u[-1])>1e-6: u.append(v)
    st1=sig(av*tps[0]+bt); sg=sig(logit(st1)+0.05*(2*rng.random()-1))
    if not u: return None
    spick=min(u,key=lambda v:abs(v-sg))
    b_rec=logit(spick)-av*tps[0]
    return len(u), int(any(abs(v-st1)<1e-4 for v in u)), abs(b_rec-bt)
print('COUPLED multi-probe (P=3 (shared w2,w3 + s-coupling), 6 nets:')
errs=[]; nr=[]
for sd_ in range(6):
    res=run(sd_)
    if res: nr.append(res[0]); errs.append(res[2]); 
    print(f'  seed {sd_}: #distinct s1-roots={res[0]:2d}  true-found={res[1]}  bias err={res[2]:.2e}' if res else f'  seed {sd_}: none')
errs=np.array(errs)
print(f'SUMMARY: median #roots={int(np.median(nr))}  bias err median={np.median(errs):.2e} max={errs.max():.2e}  frac<1e-4={(errs<1e-4).mean():.0%}')
