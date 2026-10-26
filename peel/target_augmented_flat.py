"""Add L7 kink points where the sole unexcited known L6 feature changes sign."""
import sys,runpy,torch
sys.argv=['augmented_missing_probe.py','--reuse']
v=runpy.run_path('peel/augmented_missing_probe.py')
from peel.local_kink_refine import locate_target
from verify_layer1 import _Oracle
K=v['K'];aug=v['aug'];net=v['net'];t=v['t'];chosen=v['chosen'];target=v['target'];ids=v['ids']
fit=v['v'].cuda();flat=int(v['flat'][0]);c=int(ids[flat]);bb=_Oracle(t)
gen=torch.Generator(device='cuda').manual_seed(718)
points=[v['archive']['points'][chosen].cuda()]
with torch.no_grad():aug.layers[6].weight[chosen]=fit[:-1];aug.layers[6].bias[chosen]=fit[-1]
for rnd in range(12):
 x=torch.randn(128,3072,device='cuda',dtype=torch.float64,generator=gen)*[1.,4.,16.][rnd%3]
 desired=[.05,.2,.8][rnd%3]
 for it in range(30):
  g,n=K._g_and_normal(aug,x,6,aug.layers[6].weight[chosen],aug.layers[6].bias[chosen])
  z,p=K._g_and_normal(net,x,5,net.layers[5].weight[c],net.layers[5].bias[c])
  a=n.square().sum(1);b=(n*p).sum(1);cc=p.square().sum(1);det=(a*cc-b*b).clamp_min(1e-30)
  dx=((cc*g-b*(z-desired))/det)[:,None]*n+((a*(z-desired)-b*g)/det)[:,None]*p
  x-=dx
 g,n=K._g_and_normal(aug,x,6,aug.layers[6].weight[chosen],aug.layers[6].bias[chosen])
 h=K._phi(aug,x,6);z=net.layers[5](K._phi(net,x,5))[:,c]
 valid=(g.abs()<1e-7)&(z>desired*.5)&torch.isfinite(x).all(1)&(x.norm(dim=1)<1e6)
 x=x[valid];n=n[valid];h=h[valid]
 if not len(x):print('TARGET',rnd,'no reachable points',flush=True);continue
 norm=n.norm(dim=1);u=n/norm[:,None]
 r=3*[.001,.005,.02][rnd%3]*(1+h.norm(dim=1))/399**.5/norm
 xs,ok=locate_target(bb,aug,6,chosen,x,u,r,gen)
 if ok.any():points.append(xs[ok])
 print('TARGET',rnd,'proposed',len(x),'hits',int(ok.sum()),'q',bb.n,flush=True)
 if sum(len(p) for p in points)>800:break
raw=torch.cat(points)
h=K._phi(aug,raw,6)
for rounds in [8,32]:
 w,b,kept,gap=K.solve_from_points(h,aug.layers[6].weight[chosen],rounds=rounds,drop=.02)
 normal=w[199:];normal/=normal.norm()
 print('REFIT','gap',gap,'kept',kept,'cos',float((normal@target).abs()),flush=True)
torch.save(dict(points=raw.cpu(),weight=w,bias=b,queries=bb.n,total_queries=v['q']+bb.n,normal=normal.cpu()),'recon/_augmented_targeted_missing.pt')
