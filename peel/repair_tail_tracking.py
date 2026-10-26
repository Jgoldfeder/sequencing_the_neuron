"""Reuse known kink anchors instead of repeatedly scanning straggler guesses."""
import sys,argparse,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from verify_layer1 import _Oracle
from peel.precision_prefix import cleanup_layer
from align import align_clone_to
import kink_solve as K
ap=argparse.ArgumentParser();ap.add_argument('--layer',type=int,default=7);a=ap.parse_args();l=a.layer-1
torch.set_num_threads(2)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
r=torch.load(f'recon/_precision_L{l+1}_tail.pt',map_location='cpu',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
t=model(d['teacher_state']);m=model(r['state_dict']);mask=r['mask'].cuda();left=(~mask).nonzero().flatten().tolist();points={}
for file in sorted(Path('peel').glob(f'tail_L{l+1}_attempt*_points.pt')):
 data=torch.load(file,map_location='cpu',weights_only=False)['points']
 for c in left:
  if c in data and (c not in points or len(data[c])>len(points[c])):points[c]=data[c]
anchors={c:(x.cuda(),torch.full((len(x),),c,device='cuda',dtype=torch.long)) for c,x in points.items()}
print('ANCHORS',{c:len(v[0]) for c,v in anchors.items()},flush=True)
tracking=copy.deepcopy(m)
with torch.no_grad():
 for c in anchors:
  norm=torch.cat([tracking.layers[l].weight[c],tracking.layers[l].bias[c:c+1]]).norm()
  tracking.layers[l].weight[c]/=norm;tracking.layers[l].bias[c]/=norm
bb=_Oracle(t);archive={}
K.track_layer(bb,tracking,l,anchors,torch.Generator(device='cuda').manual_seed(919),need=500,step=2.,max_steps=300,archive=archive,verbose=True)
torch.save(dict(points=archive,queries=bb.n),f'peel/tail_L{l+1}_tracking_points.pt')
m,ok,n,_,stats=cleanup_layer(t,m,l,points=archive,channels=list(archive),frozen_mask=mask);mask|=ok
ref,truth=align_clone_to(t,m);e=(ref.layers[l].weight-truth.layers[l].weight).abs().amax(1)
print('RESULT','solved',int(mask.sum()),'NEW_QUERIES',bb.n+n,'MAX_W',float(e[mask].max()),flush=True)
result=dict(dims=d['dims'],act=d['act'],state_dict=m.state_dict(),mask=mask,tail_queries=r['tail_queries']+bb.n+n,stats=stats,assumption='float32_teacher_parameters')
torch.save(result,f'recon/_precision_L{l+1}_tail.pt')
if mask.all():torch.save(result,f'recon/_precision_L{l+1}_complete.pt')
