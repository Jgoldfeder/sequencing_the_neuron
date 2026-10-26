import sys,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import align_clone_to
import kink_solve as K
from threadpoolctl import threadpool_limits
torch.set_num_threads(2)
ap=argparse.ArgumentParser();ap.add_argument('--layer',type=int,default=5);ap.add_argument('--count',type=int,default=8);a=ap.parse_args();l=a.layer-1
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
r=torch.load('recon/_refined_mlp9_track_L1-5.pt',map_location='cpu',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
t=model(d['teacher_state']);g=model(d['pop_states'][2])
for j in range(l):g.layers[j].load_state_dict({k:r['state_dict'][f'layers.{j}.{k}'] for k in ['weight','bias']})
for mode in ['branch','chain']:
 ar={};start=time.time()
 with threadpool_limits(limits=1):w,b,ok,q=K.recover_layer(t,g,l,'cuda',only_channels=list(range(a.count)),sampling='track',track_step=2.,track_need=400,track_walk=mode,track_archive=ar,gen=torch.Generator(device='cuda').manual_seed(1234+l),verbose=False)
 m=g.clone()
 with torch.no_grad():
  idx=ok.nonzero().flatten();v=torch.cat([w[idx],b[idx,None]],1);v/=v.norm(dim=1,keepdim=True);old=torch.cat([m.layers[l].weight[idx],m.layers[l].bias[idx,None]],1);scale=(old*v).sum(1);m.layers[l].weight[idx]=v[:,:-1]*scale[:,None];m.layers[l].bias[idx]=v[:,-1]*scale
 ref,teacher=align_clone_to(t,m);err=(ref.layers[l].weight-teacher.layers[l].weight).abs().amax(1)
 print(mode,'accepted',int(ok.sum()),'q',q,'seconds',time.time()-start,'errors',err[:a.count].tolist(),flush=True)
 torch.save(dict(points=ar,mask=ok.cpu(),queries=q,errors=err.cpu()),f'peel/track_{mode}_L{l+1}.pt')
