"""Production non-cheat path: complete L1, incomplete L2, preserve fp64 L1."""
import copy,sys,tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,method
from nets import MLP

torch.set_num_threads(1);torch.manual_seed(19)
dims=[12,8,6,3];teacher=MLP(dims)
with torch.no_grad():
 for l in teacher.layers:l.bias.normal_(0,.1)
def factory(*a,**kw):
 m=copy.deepcopy(teacher)
 with torch.no_grad():
  for l in m.layers:l.weight.add_(1e-5*torch.randn_like(l.weight))
 return m
original=method._mlp_refine_layer
calls=[];banks=[]
def refine(*a,**kw):
 r=original(*a,**kw)
 if a[2]==1:r[2][-1]=False
 calls.append((a[2],r[2].clone()))
 return r
warm=method._mlp_warm_reinit_population
def rebuild(*a,**kw):
 banks.append({l:(w.clone(),b.clone(),m.clone()) for l,(w,b,m) in a[4].items()})
 return warm(*a,**kw)
cfg=method.Cfg(p=2,q=64,outer=3,epochs=0,qg_steps=0,log_every=1,
 fast_peel_partial=True,freeze_reinit=True,freeze_thresh=.5,freeze_precision=0,
 design_refine=True,normalized_peel=True,cluster_eps=.01)
with patch.object(method,'MLP',side_effect=factory),patch.object(method,'_mlp_refine_layer',side_effect=refine),patch.object(method,'_mlp_warm_reinit_population',side_effect=rebuild):
 method.reconstruct(teacher,dims,cfg,'cpu',torch.randn(64,12))
assert len(banks)==2, len(banks)  # no new solution must not rebuild/reset Adam
assert all(torch.equal(banks[0][0][0],bank[0][0]) for bank in banks)
assert banks[-1][0][0].dtype==torch.float64
assert 1 in banks[-1] and int(banks[-1][1][2].sum())==5
assert not banks[-1][1][2][-1]
print('PASS production partial freeze: original fp64 prefix retained, rejected neuron remains trainable')
