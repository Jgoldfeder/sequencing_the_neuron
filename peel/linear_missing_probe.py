"""Discover a rare leaky-ReLU normal from its linear branch, reusing queries.

Fit an affine bypass for the one unknown frontier unit, then inspect its rank.
No teacher parameters enter fitting or sample selection. Truth is diagnostics.
"""
import sys,copy,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import align_clone_to
from gn_polish import gn_polish
ap=argparse.ArgumentParser();ap.add_argument("--preconditioned",action="store_true");args=ap.parse_args()
state={}
torch.set_num_threads(2)
data=torch.load('recon/_missing_residual_probe.pt',map_location='cpu',weights_only=False)
r=torch.load('recon/_local_mlp9_L6_canonical_repair.pt',map_location='cpu',weights_only=False)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
H=data['H'].cuda();Y=data['Y'].cuda();missing=(~r['refined_mask']).nonzero().flatten().cuda()
with torch.no_grad():
 mean=H[:20000].mean(0);center=H[:20000]-mean
 vals,vec=torch.linalg.eigh(center.T@center/len(center));roots=vals.clamp_min(vals.max()*1e-7).sqrt()
 enc=vec/roots[None];dec=roots[:,None]*vec.T;Z=(H-mean)@enc
 tail=MLP(d['dims'][5:],act=d['act']).double().cuda();tail.load_state_dict(data['tail_state'])
class Baseline(torch.nn.Module):
 def __init__(self):
  super().__init__();self.first=copy.deepcopy(tail.layers[0]);self.layers=torch.nn.ModuleList(copy.deepcopy(list(tail.layers[1:])));self.skip=torch.nn.Linear(200,200,bias=False,dtype=torch.float64,device='cuda')
  with torch.no_grad():
   w=self.first.weight.clone();self.first.weight.copy_(w@dec.T);self.first.bias.add_(w@mean);self.skip.weight.zero_()
  self.first.requires_grad_(False);self.register_buffer('mask',torch.ones(200,device='cuda',dtype=torch.float64));self.mask[missing]=0
 def forward(self,z):
  h=torch.nn.functional.leaky_relu(self.first(z),.01)*self.mask
  h=torch.nn.functional.leaky_relu(self.layers[0](h)+self.skip(z),.01)
  for j,layer in enumerate(self.layers[1:]):
   h=layer(h)
   if j+2<len(self.layers):h=torch.nn.functional.leaky_relu(h,.01)
  return h
m=Baseline()
# Only used below for a post-hoc normal diagnostic.
def model(s):
 n=MLP(d['dims'],act=d['act']).double().cuda();n.load_state_dict(s);return n
_,teacher_aligned=align_clone_to(model(d['teacher_state']),model(r['state_dict']))
truth=teacher_aligned.layers[5].weight[int(missing[0])].detach();truth/=truth.norm()
mask=torch.ones(20000,device='cuda',dtype=torch.bool)
for stage in range(4):
 print('STAGE',stage,'training',int(mask.sum()),'new_queries=0',flush=True)
 if args.preconditioned:
  from peel.preconditioned_gn import polish
  m,info=polish(m,Z[:20000][mask],Y[:20000][mask],iters=3,cg_iters=60,chunk=4096,state=state)
 else:m,info=gn_polish(m,Z[:20000][mask],Y[:20000][mask],iters=3,cg_iters=50,chunk=4096,device='cuda',tag='[linear-missing]')
 with torch.no_grad():
  residual=torch.cat([(m(z)-y).square().mean(1) for z,y in zip(Z.split(2048),Y.split(2048))])
  limit=torch.quantile(residual[:20000],.95);mask=residual[:20000]<=limit
  B=m.skip.weight@enc.T;u,s,vh=torch.linalg.svd(B)
  print('RESULT','val',float(residual[20000:].mean()),'train_median',float(residual[:20000].median()),'singular',s[:8].tolist(),'diagnostic_cos',(vh[:8]@truth).abs().tolist(),flush=True)
  torch.save(dict(state_dict=m.state_dict(),mean=mean,encode=enc,decode=dec,skip=B,normal_candidates=vh[:8],residual=residual.cpu(),queries=0,stage=stage), 'recon/_linear_missing_preconditioned.pt' if args.preconditioned else 'recon/_linear_missing_probe.pt')
