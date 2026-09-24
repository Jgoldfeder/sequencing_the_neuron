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
sig=lambda z:1/(1+np.exp(-z))
def run(seed):
    rng=np.random.default_rng(seed); av=1.0
    b_true=rng.uniform(-0.5,0.5); w2t=rng.uniform(-2,2); c2=rng.uniform(-1,1); w3t=rng.uniform(-2,2); c3=rng.uniform(-1,1)
    tps=rng.uniform(-1.0,1.0,5)                      # 5 probes
    rootsets=[]
    for tp in tps:
        st=sig(av*tp+b_true); Pt=sig(w2t*st+c2); Qt=sig(w3t*Pt+c3)
        F1=Ff[1](st,w2t,w3t,Pt,Qt,av); Rv=[None,None]+[Ff[n](st,w2t,w3t,Pt,Qt,av)/F1 for n in range(2,7)]
        def resid(v):
            F1v=Ff[1](*v,av); return [Rv[n]*F1v-Ff[n](*v,av) for n in range(2,7)]
        roots=[]
        for i in range(220):
            r=np.random.default_rng(abs(seed*7000+int(1e4*tp)+i)+1)
            x0=[r.uniform(0.03,0.97),r.uniform(-3,3),r.uniform(-3,3),r.uniform(0.03,0.97),r.uniform(0.03,0.97)]
            try: sol=least_squares(resid,x0,method='lm',max_nfev=300,xtol=1e-14,ftol=1e-14)
            except Exception: continue
            if sol.cost<1e-18 and 1e-3<sol.x[0]<1-1e-3 and 0<sol.x[3]<1 and 0<sol.x[4]<1: roots.append(sol.x[0])
        u=[]
        for v in sorted(roots):
            if not u or abs(v-u[-1])>1e-6: u.append(v)
        rootsets.append((tp,np.array(u)))
    # cross-probe consistency: find b minimizing sum_p min_root |sig(a t_p+b)-root|
    bg=b_true+0.05*(2*rng.random()-1)
    def dist(b):
        d=0.0
        for tp,rs in rootsets:
            if len(rs)==0: return 9.9
            d+=np.min(np.abs(sig(av*tp+b)-rs))
        return d
    bs=np.linspace(bg-0.12,bg+0.12,4001); b0=bs[np.argmin([dist(b) for b in bs])]
    bs2=np.linspace(b0-1e-3,b0+1e-3,2001); b1=bs2[np.argmin([dist(b) for b in bs2])]
    bs3=np.linspace(b1-3e-6,b1+3e-6,6001); b2=bs3[np.argmin([dist(b) for b in bs3])]
    nroots=[len(rs) for _,rs in rootsets]
    return abs(b2-b_true), abs(b0-b_true), nroots, float(dist(b_true))
print('1->1->1->1 CROSS-PROBE consistency (5 probes, no downstream init):')
for sd_ in range(5):
    err,errgrid,nr,dtrue=run(sd_)
    print(f'  seed {sd_}: bias err={err:.2e} (grid {errgrid:.2e})  #roots/probe={nr}  dist(true)={dtrue:.1e}')
