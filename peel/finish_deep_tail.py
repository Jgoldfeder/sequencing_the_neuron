"""Continue from the precision prefix using aligned committee guesses and kinks."""
import sys,argparse,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import align_clone_to,scale_normalize_
from peel.row_geometry import input_scales
from peel.local_kink_refine import recover_layer
from peel.precision_prefix import cleanup_layer
ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=200);ap.add_argument('--start',type=int,default=7);ap.add_argument('--last',type=int,default=8);ap.add_argument('--member',type=int,default=3);ap.add_argument("--reuse",action="store_true");ap.add_argument("--tracking",action="store_true");args=ap.parse_args()
torch.set_num_threads(2)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
r=torch.load('recon/_precision_L6_complete.pt' if args.start==7 else f'recon/_precision_L{args.start-1}_complete.pt',map_location='cpu',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
t=model(d['teacher_state']);m=model(r['state_dict']);q=0
for l in range(args.start-1,args.last):
 # Align against the recovered prefix before installing an actual member's tail.
 canonical=copy.deepcopy(m);scale_normalize_(canonical)
 _,source=align_clone_to(model(d['pop_states'][args.member]),canonical)
 candidates=[]
 for member,state in enumerate(d['pop_states']):
  _,aligned=align_clone_to(model(state),canonical)
  with torch.no_grad():
   if l==6:
    rows=aligned.layers[5].weight;cos=rows@rows[138];cos[138]=-1
    dup=int(cos.argmax())
    if float(cos[dup])>.98:
     ratio=(rows[138]@rows[dup])/rows[dup].square().sum()
     aligned.layers[6].weight[:,dup]+=ratio*aligned.layers[6].weight[:,138]
     aligned.layers[6].weight[:,138]=0
     print('MERGE_DUPLICATE',member,138,dup,float(cos[dup]),flush=True)
   row=torch.cat([aligned.layers[l].weight,aligned.layers[l].bias[:,None]],1)
   candidates.append(row)
 bank=torch.stack(candidates);unit=bank/bank.norm(dim=2,keepdim=True)
 agreement=torch.einsum('mcd,ncd->mnc',unit,unit).sum(1)
 choice=agreement.argmax(0)
 selected=bank[choice,torch.arange(200,device='cuda')]
 with torch.no_grad():source.layers[l].weight.copy_(selected[:,:-1]);source.layers[l].bias.copy_(selected[:,-1])
 with torch.no_grad():
  scale=input_scales(m,l)
  for j in range(l,len(m.layers)):m.layers[j].load_state_dict(source.layers[j].state_dict())
  m.layers[l].weight.div_(scale[None])
 mask=torch.zeros(200,dtype=torch.bool,device='cuda');stats={};points={}
 if args.reuse and Path(f'recon/_precision_L{l+1}_tail.pt').exists():
  saved=torch.load(f'recon/_precision_L{l+1}_tail.pt',map_location='cpu',weights_only=False);mask=saved['mask'].cuda();previous=model(saved['state_dict'])
  with torch.no_grad():
   m.layers[l].weight[mask]=previous.layers[l].weight[mask];m.layers[l].bias[mask]=previous.layers[l].bias[mask]
 for attempt in range(5):
  left=[c for c in range(args.count) if not bool(mask[c])]
  if not left:break
  archive={}
  if args.tracking and attempt==0:
   import kink_solve as K
   raw={}
   w,b,ok,n=K.recover_layer(t,m,l,'cuda',only_channels=left,sampling='track',track_step=2.,track_need=400,track_archive=raw)
   archive=dict(points=raw,queries=n)
  else:
   w,b,ok,n=recover_layer(t,m,l,channels=left,rounds=24,need=240,archive=archive,seed=1218+attempt,initial_points=points,widths=(.01,.02,.005,.04,.002,.001,.0005,.00025))
  q+=n
  points.update(archive['points']);torch.save(archive,f'peel/tail_L{l+1}_attempt{attempt}_points.pt')
  # Lattice cleanup proposes its own robust fit; no teacher scores select rows.
  m,accepted,n,raw,updates=cleanup_layer(t,m,l,points=archive['points'],channels=[c for c in left if c in archive['points']],frozen_mask=mask);q+=n
  mask|=accepted;stats.update(updates)
  ref,truth=align_clone_to(t,m);e=(ref.layers[l].weight-truth.layers[l].weight).abs().amax(1);eb=(ref.layers[l].bias-truth.layers[l].bias).abs()
  print('RESULT',l+1,attempt,'solved',int(mask.sum()),'q',q,'maxW',float(e[mask].max()) if mask.any() else None,'maxB',float(eb[mask].max()) if mask.any() else None,flush=True)
  torch.save(dict(dims=d['dims'],act=d['act'],state_dict=m.state_dict(),mask=mask,tail_queries=q,stats=stats,assumption='float32_teacher_parameters'),f'recon/_precision_L{l+1}_tail.pt')
 if not bool(mask.all()):break
 torch.save(dict(dims=d['dims'],act=d['act'],state_dict=m.state_dict(),mask=mask,tail_queries=q,assumption='float32_teacher_parameters'),f'recon/_precision_L{l+1}_complete.pt')
