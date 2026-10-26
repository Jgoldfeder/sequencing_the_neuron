"""Infer the missing normal from a deeper guessed kink's affine remainder.

All 199 recovered L6 activations are known features. On the unknown unit's
negative branch, its contribution to L7 is linear in H5. Fit that remainder
jointly with the known activation coefficients using only forward kink points.
"""
import sys,copy,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import align_clone_to,scale_normalize_
from peel.local_kink_refine import recover_layer
import kink_solve as K
ap=argparse.ArgumentParser();ap.add_argument("--reuse",action="store_true");ap.add_argument("--extend",action="store_true");args=ap.parse_args()
torch.set_num_threads(2)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
r=torch.load('recon/_precision_L6.pt',map_location='cpu',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
t=model(d['teacher_state']);net=model(r['state_dict']);scale_normalize_(net)
_,source=align_clone_to(model(d['pop_states'][3]),net)
_,second=align_clone_to(model(d['pop_states'][1]),net)
known=r['mask'].cuda();missing=int((~known).nonzero()[0]);ids=known.nonzero().flatten()
class Features(torch.nn.Module):
 def __init__(self):
  super().__init__();self.layer=copy.deepcopy(net.layers[5]);self.register_buffer('ids',ids)
 def forward(self,h):
  # Outer net.act undoes this inverse; no new nonlinear boundary is introduced.
  z=torch.cat([torch.nn.functional.leaky_relu(self.layer(h)[:,self.ids],.01),h],1)
  return torch.where(z>=0,z,z/.01)
aug=copy.deepcopy(net);aug.layers[5]=Features()
aug.layers[6]=torch.nn.Linear(399,200,device='cuda',dtype=torch.float64)
with torch.no_grad():
 aug.layers[6].weight[:,:199]=source.layers[6].weight[:,ids]
 aug.layers[6].weight[:,199:]=.01*source.layers[6].weight[:,missing,None]*source.layers[5].weight[missing,None]
 aug.layers[6].bias.copy_(source.layers[6].bias+.01*source.layers[6].weight[:,missing]*source.layers[5].bias[missing])
 def rows(n):
  w=torch.cat([n.layers[6].weight,n.layers[6].bias[:,None]],1);return w/w.norm(dim=1,keepdim=True)
 cos=(rows(source)*rows(second)).sum(1)
 chosen=cos.argmax().item();print('TARGET_L7',chosen,'committee_cos',float(cos[chosen]),flush=True)
aug.requires_grad_(False)
archive={}
previous=torch.load('recon/_augmented_missing_probe.pt',map_location='cpu',weights_only=False) if args.reuse or args.extend else None
if args.reuse:
 archive=previous['archive'];w,b,ok,q=(previous[k] for k in ['weight','bias','mask','queries'])
else:
 w,b,ok,q=recover_layer(t,aug,6,channels=[chosen],need=1050,rounds=60,pool=768,batch=128,archive=archive,seed=721 if args.extend else 720,initial_points=previous['archive']['points'] if previous else None)
 if previous:q+=previous['queries']
torch.save(dict(weight=w,bias=b,mask=ok,queries=q,archive=archive,chosen=chosen,known=ids), 'recon/_augmented_missing_probe.pt')
print('RESULT',int(ok.sum()),'QUERIES',q,archive['stats'],flush=True)
# Diagnostics only, after the inference run.
ref,truth=align_clone_to(t,net);target=truth.layers[5].weight[missing].detach();target=target/target.norm()
if chosen in archive['points']:
 h=K._phi(aug,archive['points'][chosen].cuda(),6)
 z=net.layers[5](K._phi(net,archive['points'][chosen].cuda(),5))[:,ids]
 print('FEATURE_COVERAGE',int(((z.min(0).values<0)&(z.max(0).values>0)).sum()),'of',len(ids),flush=True)
 for rounds in [8,32]:
  ww,bb,kept,gap=K.solve_from_points(h,aug.layers[6].weight[chosen],rounds=rounds,drop=.04)
  v=ww[199:];print('DIAGNOSTIC',rounds,'gap',gap,'linear_norm',float(v.norm()),'cos',float((v@target).abs()/v.norm()),flush=True)

with torch.no_grad():
 variable=(z.min(0).values<0)&(z.max(0).values>0)
 flat=(~variable).nonzero().flatten()
 print('FLAT_FEATURES',flat.tolist(),flush=True)
 A=torch.cat([h,torch.ones(len(h),1,device=h.device,dtype=h.dtype)],1).cpu()
 guess=torch.cat([aug.layers[6].weight[chosen],aug.layers[6].bias[chosen:chosen+1]]).detach().cpu()
 fixed=torch.cat([flat.cpu(),guess.abs().argmax().view(1)]).unique()
 free=torch.ones(400,dtype=torch.bool);free[fixed]=False
 keep=torch.ones(len(A),dtype=torch.bool)
 for iteration in range(8):
  v=guess.clone();v[free]=torch.linalg.lstsq(A[keep][:,free],-A[keep][:,fixed]@v[fixed],driver='gelsd').solution
  res=(A@v).abs();limit=torch.quantile(res,.95);keep=res<=limit
 normal=v[199:399].cuda();normal/=normal.norm()
 print('FIXED_FLAT_RESULT','residual',float(res.median()),'cos',float((normal@target).abs()),flush=True)
 torch.save(dict(normal=normal.cpu(),known_flat=ids[flat].cpu(),weight=v,points=archive['points'][chosen]),'recon/_augmented_missing_candidate.pt')
