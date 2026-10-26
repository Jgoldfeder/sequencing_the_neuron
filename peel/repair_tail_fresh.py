"""Use tracked fits as proposals, then independent points for precision cleanup."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import align_clone_to
from peel.local_kink_refine import recover_layer
from peel.precision_prefix import cleanup_layer
import kink_solve as K
p=argparse.ArgumentParser();p.add_argument('--layer',type=int,default=7);a=p.parse_args();l=a.layer-1
torch.set_num_threads(2)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
r=torch.load(f'recon/_precision_L{l+1}_tail.pt',map_location='cpu',weights_only=False)
raw=torch.load(f'peel/tail_L{l+1}_tracking_points.pt',map_location='cpu',weights_only=False)['points']
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
t=model(d['teacher_state']);m=model(r['state_dict']);mask=r['mask'].cuda();left=(~mask).nonzero().flatten().tolist()
with torch.no_grad():
 for c in left:
  if c not in raw:continue
  h=K._phi(m,raw[c].cuda(),l)
  w,b,kept,gap=K.solve_from_points(h,m.layers[l].weight[c],rounds=32,drop=.04)
  old=torch.cat([m.layers[l].weight[c],m.layers[l].bias[c:c+1]]);v=torch.cat([w,b.reshape(1)])
  cos=float((v@old)/(v.norm()*old.norm()))
  print('PROPOSAL',c,'gap',gap,'cos',cos,flush=True)
  if gap<1e-6 and cos>.97:
   scale=old.norm()/v.norm();m.layers[l].weight[c]=w*scale;m.layers[l].bias[c]=b*scale
archive={};w,b,ok,q=recover_layer(t,m,l,channels=left,rounds=24,need=400,archive=archive,seed=993)
torch.save(archive,f'peel/tail_L{l+1}_fresh_points.pt')
m,ok,n,_,stats=cleanup_layer(t,m,l,points=archive['points'],channels=left,frozen_mask=mask);mask|=ok
ref,truth=align_clone_to(t,m);e=(ref.layers[l].weight-truth.layers[l].weight).abs().amax(1)
print('RESULT',int(mask.sum()),'NEW_QUERIES',q+n,'MAX_W',float(e[mask].max()),flush=True)
result=dict(dims=d['dims'],act=d['act'],state_dict=m.state_dict(),mask=mask,tail_queries=r['tail_queries']+q+n,stats=stats,assumption='float32_teacher_parameters')
torch.save(result,f'recon/_precision_L{l+1}_tail.pt')
if mask.all():torch.save(result,f'recon/_precision_L{l+1}_complete.pt')
