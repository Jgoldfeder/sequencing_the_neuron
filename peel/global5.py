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
sig=lambda z:1/(1+np.exp(-z)); print('built')
def setup(seed,Pn):
    rng=np.random.default_rng(seed); av=1.0
    bt=rng.uniform(-0.6,0.6); w2t=rng.uniform(-2,2); c2=rng.uniform(-1,1); w3t=rng.uniform(-2,2); c3=rng.uniform(-1,1)
    tps=np.linspace(-0.8,0.8,Pn)
    Rm=[]
    for tp in tps:
        st=sig(av*tp+bt); Pt=sig(w2t*st+c2); Qt=sig(w3t*Pt+c3); F1=Ff[1](st,w2t,w3t,Pt,Qt,av)
        Rm.append([Ff[n](st,w2t,w3t,Pt,Qt,av)/F1 for n in range(2,7)])
    return (bt,w2t,c2,w3t,c3),tps,np.array(Rm),av
def resid(theta,tps,Rm,av):
    b,w2v,c2v,w3v,c3v=theta; r=[]
    for i,tp in enumerate(tps):
        sp_=sig(av*tp+b); Pp=sig(w2v*sp_+c2v); Qp=sig(w3v*Pp+c3v); F1=Ff[1](sp_,w2v,w3v,Pp,Qp,av)
        for j,n in enumerate(range(2,7)): r.append(Ff[n](sp_,w2v,w3v,Pp,Qp,av)/F1 - Rm[i,j])
    return r
# CONTROL: fix b=b*, optimize (w2,c2,w3,c3) from random
print('CONTROL (b fixed at true, optimize 4 downstream from random, 5 nets x 15 restarts):')
for sd_ in range(5):
    (bt,w2t,c2,w3t,c3),tps,Rm,av=setup(sd_,5); best=9
    for rs in range(15):
        r=np.random.default_rng(sd_*100+rs); th0=[bt,r.uniform(-3,3),r.uniform(-2,2),r.uniform(-3,3),r.uniform(-2,2)]
        sol=least_squares(lambda t:resid([bt]+list(t[1:]),tps,Rm,av) if False else resid(np.array([bt,*t[1:]]),tps,Rm,av),th0,method='lm',max_nfev=600)
        best=min(best,sol.cost)
    print(f'  net {sd_}: best residual cost (b fixed true) = {best:.2e}')
# FULL: b init guess, downstream random
print('FULL (b init=guess+/-0.05, downstream random, 12 nets x 25 restarts):')
errs=[]
for sd_ in range(12):
    (bt,w2t,c2,w3t,c3),tps,Rm,av=setup(sd_,5); rng=np.random.default_rng(sd_+777); bg=bt+0.05*(2*rng.random()-1)
    bestx=None;bestc=9
    for rs in range(25):
        r=np.random.default_rng(sd_*211+rs); th0=[bg,r.uniform(-3,3),r.uniform(-2,2),r.uniform(-3,3),r.uniform(-2,2)]
        sol=least_squares(resid,th0,args=(tps,Rm,av),method='lm',max_nfev=600)
        if sol.cost<bestc: bestc=sol.cost; bestx=sol.x
    errs.append(abs(bestx[0]-bt))
    print(f'  net {sd_}: bias err={abs(bestx[0]-bt):.2e}  (final residual {bestc:.1e})')
errs=np.array(errs)
print(f'SUMMARY full: bias err median={np.median(errs):.2e} max={errs.max():.2e} frac<1e-4={(errs<1e-4).mean():.0%} frac<1e-6={(errs<1e-6).mean():.0%}')
