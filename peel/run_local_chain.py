"""Bounded local-kink experiment; saves after every layer, never declares rejected rows solved."""
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import param_errors
from peel.local_kink_refine import recover_layer
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--stop',type=int,default=8);ap.add_argument('--resume',default='');args=ap.parse_args()
torch.set_num_threads(2)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cuda',weights_only=False)
r=torch.load('recon/_refined_mlp9_track_L1-5.pt',map_location='cuda',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
teacher=model(d['teacher_state']);guess=model(d['pop_states'][2])
guess.layers[0].load_state_dict({k:r['state_dict'][f'layers.0.{k}'] for k in ['weight','bias']})
qtotal=0;resume=None;start=1
if args.resume:
 resume=torch.load(args.resume,map_location='cuda',weights_only=False)
 guess.load_state_dict(resume['state_dict']);qtotal=resume['new_queries'];start=resume['frontier']
for l in range(start,args.stop):
 archive={};t=time.time()
 if resume is not None and l==start:
  w=guess.layers[l].weight.detach().clone();b=guess.layers[l].bias.detach().clone()
  v=torch.cat([w,b[:,None]],1);norm=v.norm(dim=1);w/=norm[:,None];b/=norm
  ok=resume['mask'].to('cuda');q=0
 else:w,b,ok,q=recover_layer(teacher,guess,l,rounds=24,archive=archive)
 qtotal+=q
 left=(~ok).nonzero().flatten().tolist()
 if left:
  from peel.informative_kinks import recover_layer as design
  w2,b2,m2,q2=design(teacher,guess,l,'cuda',only_channels=left,max_rounds=24,verbose=True)
  for c in left:
   if bool(m2[c]):w[c]=w2[c];b[c]=b2[c];ok[c]=True
  q+=q2;qtotal+=q2
 with torch.no_grad():
  idx=ok.nonzero().flatten();g=torch.cat([guess.layers[l].weight[idx],guess.layers[l].bias[idx,None]],1);v=torch.cat([w[idx],b[idx,None]],1);scale=(g*v).sum(1).clamp_min(1e-12);guess.layers[l].weight[idx]=w[idx]*scale[:,None];guess.layers[l].bias[idx]=b[idx]*scale
 # Teacher parameters used only after recovery for reporting.
 e=param_errors(guess,teacher)['max_eps_per_matrix']
 print('RESULT',l+1,int(ok.sum()),q,qtotal,time.time()-t,'errors',e[:2*(l+1)],flush=True)
 if archive:torch.save(archive,f'peel/local_chain_L{l+1}_points.pt')
 torch.save(dict(dims=d['dims'],act=d['act'],state_dict=guess.state_dict(),new_queries=qtotal,imported_prefix='recon/_refined_mlp9_track_L1-5.pt:L1',mask=ok.cpu(),frontier=l),f'recon/_local_chain_L{l+1}.pt')
 if not bool(ok.all()):break
