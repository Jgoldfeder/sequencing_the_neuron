"""Resume optional float32-prior cleanup, querying only rows that failed."""
import sys,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import align_clone_to
from peel.precision_prefix import cleanup_layer
from peel.local_kink_refine import recover_layer
p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=2);p.add_argument('--last',type=int,default=5);a=p.parse_args()
torch.set_num_threads(2)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
checkpoint=Path(f'recon/_precision_L{a.start}.pt')
resume_current=checkpoint.exists()
r=torch.load(checkpoint if resume_current else f'recon/_precision_L{a.start-1}.pt',map_location='cpu',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
t=model(d['teacher_state']);m=model(r['state_dict']);q=r['cleanup_queries']
for l in range(a.start-1,a.last):
 points=torch.load(f'peel/local_chain_L{l+1}_points.pt' if l<5 else 'recon/_local_mlp9_L6.pt.L6.points.pt',map_location='cpu',weights_only=False)['points']
 eligible=torch.ones(200,dtype=torch.bool,device='cuda')
 if l==5:eligible=torch.load('recon/_local_mlp9_L6_canonical_repair.pt',map_location='cpu',weights_only=False)['refined_mask'].cuda()
 if l==a.start-1 and resume_current:ok=r['mask'].cuda();stats=r['stats']
 else:
  m,ok,n,archive,stats=cleanup_layer(t,m,l,points=points,channels=eligible.nonzero().flatten().tolist());q+=n
 for attempt in range(3):
  left=(eligible&~ok).nonzero().flatten().tolist()
  if not left:break
  print('REPAIR',l+1,attempt,'rows',left,flush=True)
  fresh={}
  w,b,refined,n=recover_layer(t,m,l,channels=left,rounds=16,need=464,seed=923+attempt,archive=fresh);q+=n
  torch.save(fresh,f'peel/precision_L{l+1}_retry{attempt}_points.pt')
  # Fresh independent points replace old, contaminated surface samples.
  m,accepted,n,archive,updates=cleanup_layer(t,m,l,points=fresh['points'],channels=[c for c in left if c in fresh['points']]);q+=n
  ok|=accepted;stats.update(updates)
  torch.save(dict(dims=d['dims'],act=d['act'],state_dict=m.state_dict(),mask=ok,cleanup_queries=q,assumption='float32_teacher_parameters',stats=stats),f'recon/_precision_L{l+1}.pt')
 ref,truth=align_clone_to(t,m);err=(ref.layers[l].weight-truth.layers[l].weight).abs().amax(1);eb=(ref.layers[l].bias-truth.layers[l].bias).abs()
 print('SUMMARY',l+1,'accepted',int(ok.sum()),'queries',q,'maxW',float(err.max()),'maxB',float(eb.max()),'acceptedMaxW',float(err[ok].max()),'acceptedMaxB',float(eb[ok].max()),flush=True)
 torch.save(dict(dims=d['dims'],act=d['act'],state_dict=m.state_dict(),mask=ok,cleanup_queries=q,assumption='float32_teacher_parameters',stats=stats),f'recon/_precision_L{l+1}.pt')
 if not bool(ok[eligible].all()):break
