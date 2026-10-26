import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from peel.local_kink_refine import recover_layer
from peel.precision_prefix import cleanup_layer
from align import align_clone_to
torch.set_num_threads(2)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
r=torch.load('recon/_precision_L6.pt',map_location='cpu',weights_only=False)
f=torch.load('recon/_missing_L6_found.pt',map_location='cpu',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
t=model(d['teacher_state']);m=model(r['state_dict']);found=model(f['state_dict']);c=138
with torch.no_grad():
 scale=torch.ones(d['dims'][0],device='cuda',dtype=torch.float64)
 for l in range(5):scale=(m.layers[l].weight*scale[None]).norm(dim=1)
 m.layers[5].weight[c]=found.layers[5].weight[c]/scale
 m.layers[5].bias[c]=found.layers[5].bias[c]
archive={}
w,b,ok,q=recover_layer(t,m,5,channels=[c],rounds=16,archive=archive,seed=827)
torch.save(archive,'peel/missing_L6_precision_points.pt')
m,accepted,n,raw,stats=cleanup_layer(t,m,5,points=archive['points'],channels=[c]);q+=n
mask=r['mask'].cuda()|accepted
ref,truth=align_clone_to(t,m);err=(ref.layers[5].weight-truth.layers[5].weight).abs().max();eb=(ref.layers[5].bias-truth.layers[5].bias).abs().max()
print('RESULT',int(mask.sum()),'NEW_QUERIES',q,'MAX_W',float(err),'MAX_B',float(eb),stats,flush=True)
torch.save(dict(dims=d['dims'],act=d['act'],state_dict=m.state_dict(),mask=mask,cleanup_queries=r['cleanup_queries']+q,missing_discovery_queries=f['discovery_queries'],assumption='float32_teacher_parameters',stats=stats),'recon/_precision_L6_complete.pt')
