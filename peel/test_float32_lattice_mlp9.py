import sys,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import align_clone_to
from peel.float32_lattice import recover_row
torch.set_num_threads(2)
ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=8);a=ap.parse_args()
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
r=torch.load('recon/_local_chain_L1-5_resume.pt',map_location='cpu',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
m=model(r['state_dict']);t=model(d['teacher_state']);start=time.time();found={}
for c in range(a.count):
 w,info=recover_row(m.layers[0].weight[c],atol=1e-10)
 print('ROW',c,'found',w is not None,info,flush=True)
 if w is not None:
  found[c]=(w,info)
  with torch.no_grad():
   s=info['scale'];m.layers[0].weight[c]=w;m.layers[0].bias[c]*=s;m.layers[1].weight[:,c]/=s
ref,truth=align_clone_to(t,m)
e=(ref.layers[0].weight-truth.layers[0].weight).abs().amax(1)
print('RESULT',len(found),'seconds',time.time()-start,'errors',e[:a.count].tolist(),flush=True)
torch.save(dict(rows=found,errors=e.cpu(),state_dict=m.state_dict()),'peel/float32_lattice_L1.pt')
