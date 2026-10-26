import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import align_clone_to,scale_normalize_
torch.set_num_threads(2)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
r=torch.load('recon/_local_mlp9_L6.pt.L6.partial.pt',map_location='cpu',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double();m.load_state_dict(s);return m
original=model(d['pop_states'][2]);current=model(r['state_dict']);teacher=model(d['teacher_state'])
old=original.clone();scale_normalize_(old)
new=current.clone();scale_normalize_(new)
_,truth=align_clone_to(teacher,new)
def rows(m):
 a=torch.cat([m.layers[5].weight,m.layers[5].bias[:,None]],1);return a/a.norm(dim=1,keepdim=True)
t=rows(truth);missing=(~r['refined_mask']).nonzero().flatten().tolist()
for label,ref in [('old',old),('refreshed',new)]:
 print(label,'missing guess slots',missing,flush=True)
 for i,state in enumerate(d['pop_states']):
  _,m=align_clone_to(model(state),ref)
  cos=rows(m)@t[missing].T
  v,idx=cos.max(dim=0)
  print(i,'cos',v.tolist(),'rows',idx.tolist(),flush=True)
