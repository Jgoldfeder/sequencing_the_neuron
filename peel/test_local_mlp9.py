import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import align_clone_to,scale_normalize_
from peel.local_kink_refine import recover_layer
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--layer',type=int,default=3);ap.add_argument('--count',type=int,default=8);ap.add_argument('--rounds',type=int,default=16);ap.add_argument('--prefix',default='recon/_refined_mlp9_track_L1-5.pt');ap.add_argument('--canonical',action='store_true');args=ap.parse_args()
torch.set_num_threads(2)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cuda',weights_only=False)
r=torch.load(args.prefix,map_location='cuda',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
teacher=model(d['teacher_state']);guess=model(d['pop_states'][2]);l=args.layer-1
with torch.no_grad():
 for j in range(l):guess.layers[j].load_state_dict({k:r['state_dict'][f'layers.{j}.{k}'] for k in ['weight','bias']})
if args.canonical:scale_normalize_(guess)
archive={};t=time.time();w,b,ok,q=recover_layer(teacher,guess,l,channels=list(range(args.count)),rounds=args.rounds,archive=archive)
with torch.no_grad():
 idx=ok.nonzero().flatten();g=torch.cat([guess.layers[l].weight[idx],guess.layers[l].bias[idx,None]],1);v=torch.cat([w[idx],b[idx,None]],1);scale=(g*v).sum(1).clamp_min(1e-12);guess.layers[l].weight[idx]=w[idx]*scale[:,None];guess.layers[l].bias[idx]=b[idx]*scale
# Align teacher to guess order for per-channel diagnostic only.
a,tg=align_clone_to(teacher,guess)
errors=(a.layers[l].weight-tg.layers[l].weight).abs().amax(1)
print('RESULT',args.layer,int(ok.sum()),q,time.time()-t,'errors',errors[:args.count].tolist(),flush=True)
archive.update(errors=errors.cpu(),mask=ok.cpu());torch.save(archive,f'peel/local_mlp9_L{args.layer}.pt')
print(archive['stats'],flush=True)
