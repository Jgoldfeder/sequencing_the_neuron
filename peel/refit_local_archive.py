import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from peel.local_kink_refine import recover_layer
from align import align_clone_to
import argparse
ap=argparse.ArgumentParser();ap.add_argument("--layer",type=int,default=3);args=ap.parse_args();l=args.layer-1
torch.set_num_threads(2)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
r=torch.load(f'recon/_local_chain_L{l+1}.pt',map_location='cpu',weights_only=False)
a=torch.load(f'peel/local_chain_L{l+1}_points.pt',map_location='cpu',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
teacher=model(d['teacher_state']);guess=model(d['pop_states'][2])
for j in range(l):guess.layers[j].load_state_dict({k:r['state_dict'][f'layers.{j}.{k}'] for k in ['weight','bias']})
channels=[c for c,s in a['stats'].items() if not s['accepted']];out={}
w,b,ok,q=recover_layer(teacher,guess,l,channels=channels,rounds=0,initial_points=a['points'],archive=out)
with torch.no_grad():
 idx=ok.nonzero().flatten();v=torch.cat([w[idx],b[idx,None]],1);g=torch.cat([guess.layers[l].weight[idx],guess.layers[l].bias[idx,None]],1);scale=(v*g).sum(1);guess.layers[l].weight[idx]=w[idx]*scale[:,None];guess.layers[l].bias[idx]=b[idx]*scale
ref,t=align_clone_to(teacher,guess);err=(ref.layers[l].weight-t.layers[l].weight).abs().amax(1)
print('RECOVERED',int(ok.sum()),'OF',len(channels),'NEW_QUERIES',q,'MAX_ERROR',float(err[ok].max()) if ok.any() else None,flush=True)
print(out['stats'],flush=True)
torch.save(out,f'peel/local_archive_retry_L{l+1}.pt')
