"""Locate the unknown bias given a recovered normal, forward-only."""
import sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import scale_normalize_,align_clone_to
from peel.informative_kinks import candidate_pool
from verify_layer1 import _Oracle
import kink_solve as K
torch.set_num_threads(2)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
r=torch.load('recon/_precision_L6.pt',map_location='cpu',weights_only=False)
a=torch.load('recon/_augmented_targeted_missing.pt',map_location='cpu',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
t=model(d['teacher_state']);m=model(r['state_dict']);scale_normalize_(m)
w=a['normal'].cuda();c=138;l=5;bb=_Oracle(t);gen=torch.Generator(device='cuda').manual_seed(817)
# Only the normal is supplied; bias zero is a search center, not teacher data.
with torch.no_grad():m.layers[l].weight[c]=w;m.layers[l].bias[c]=0.
points=[]
with torch.no_grad():
 probe=torch.randn(4096,3072,device='cuda',dtype=torch.float64,generator=gen)*.5
 values=K._phi(m,probe,l)@w
 centers=(-torch.quantile(values,torch.linspace(.001,.999,33,device='cuda',dtype=torch.float64))).tolist()
 print('SEARCH_RANGE',min(centers),max(centers),flush=True)
for rnd,center in enumerate(centers):
 eps=.003
 with torch.no_grad():m.layers[l].bias[c]=center
 with torch.no_grad():
  x,u,radius,h,cross=candidate_pool(m,l,[c],gen,96,(.5,1.,2.),{c:eps})[c]
  x=x[:16];u=u[:16];radius=radius[:16]
  xs,ok=K.locate(bb,x,u,radius,gen,m,l,w.expand(len(x),-1),K=97,fp_tol=.02)
  if ok.any():points.append(xs[ok])
  print('BIAS_SEARCH',center,'hits',int(ok.sum()),'q',bb.n,flush=True)
  if sum(len(v) for v in points)>=16:break
if not points:raise RuntimeError('No matching normal found; keep neuron unsolved')
raw=torch.cat(points);h=K._phi(m,raw,l)
with torch.no_grad():
 biases=-(h@w);b=biases.median();keep=(biases-b).abs()<1e-6
 print('BIAS',float(b),'consistent',int(keep.sum()),'total',len(biases),'spread',float((biases-b).abs().median()),flush=True)
 if int(keep.sum())<8:raise RuntimeError('Bias cluster not established')
 # A missing rare neuron is oriented toward its uncommon positive branch.
 probe=torch.randn(2048,3072,device='cuda',dtype=torch.float64,generator=gen)
 z=K._phi(m,probe,l)@w+b
 if float((z>0).double().mean())>.5:w=-w;b=-b
 m.layers[l].weight[c]=w;m.layers[l].bias[c]=b
ref,truth=align_clone_to(t,m)
print('DIAGNOSTIC','W',float((ref.layers[l].weight[c]-truth.layers[l].weight[c]).abs().max()),'b',float((ref.layers[l].bias[c]-truth.layers[l].bias[c]).abs()),flush=True)
torch.save(dict(dims=d['dims'],act=d['act'],state_dict=m.state_dict(),points={c:raw[keep].cpu()},bias_queries=bb.n,discovery_queries=a['total_queries']+bb.n,orientation='rare_positive_branch'), 'recon/_missing_L6_found.pt')
