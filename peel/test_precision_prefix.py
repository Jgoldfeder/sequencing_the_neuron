import sys,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import align_clone_to,param_errors
from peel.precision_prefix import cleanup_layer
torch.set_num_threads(2)
ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=8);ap.add_argument('--last',type=int,default=1);a=ap.parse_args()
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
r=torch.load('recon/_local_mlp9_L6_canonical_repair.pt',map_location='cpu',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
t=model(d['teacher_state']);m=model(r['state_dict']);q=0
for l in range(a.last):
 points=None if l==0 else torch.load(f'peel/local_chain_L{l+1}_points.pt',map_location='cpu',weights_only=False)['points']
 m,ok,n,archive,stats=cleanup_layer(t,m,l,points=points,channels=list(range(a.count)));q+=n
 ref,teacher=align_clone_to(t,m);e=(ref.layers[l].weight-teacher.layers[l].weight).abs().amax(1);eb=(ref.layers[l].bias-teacher.layers[l].bias).abs()
 print('RESULT',l+1,int(ok.sum()),'Q',q,'WERR',e[:a.count].tolist(),'BERR',eb[:a.count].tolist(),flush=True)
 torch.save(archive,f'peel/precision_L{l+1}_points.pt')
 torch.save(dict(dims=d['dims'],act=d['act'],state_dict=m.state_dict(),mask=ok,cleanup_queries=q,assumption='float32_teacher_parameters',stats=stats),f'recon/_precision_L{l+1}.pt')
 if not bool(ok[:a.count].all()):break
