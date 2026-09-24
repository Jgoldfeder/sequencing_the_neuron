"""CPU regression checks for wide-input shallow design dispatch."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import kink_solve as K
from nets import MLP
from align import scale_normalize_

torch.set_num_threads(1);torch.manual_seed(4)
t=MLP([1024,4,3]).double()
with torch.no_grad():
 for L in t.layers:L.bias.normal_(0,.1)
scale_normalize_(t);g=t.clone()
with torch.no_grad():g.layers[0].weight.add_(torch.randn_like(g.layers[0].weight)*1e-4)
class ForwardOnly:
 def __init__(self):self.queries=0
 def __call__(self,x):self.queries+=len(x);return t(x)
f=ForwardOnly();w,b,m,nq=K.recover_layer(f,g,0,'cpu',sampling='design',verbose=False)
assert m.all() and f.queries==nq
n=w.norm(dim=1);error=max(float((w/n[:,None]-t.layers[0].weight).abs().max()),float((b/n-t.layers[0].bias).abs().max()))
assert error<1e-11,error
assert torch.allclose(w.square().sum(1)+b.square(),torch.ones_like(b),atol=1e-14,rtol=0)
for opts in [dict(max_rounds=0),dict(only_channels=[])]:
 before=f.queries;w,b,m,nq=K.recover_layer(f,g,0,'cpu',sampling='design',verbose=False,**opts)
 assert nq==0 and f.queries==before and not m.any()
 assert torch.equal(w,g.layers[0].weight) and torch.equal(b,g.layers[0].bias)
from peel.shallow_sweep import recover_layer
_,_,m,_=recover_layer(lambda x:torch.zeros(len(x),3,dtype=x.dtype),g,'cpu',only_channels=[0],attempts=2,verbose=False)
assert not m.any(), 'A constant oracle cannot identify a neuron'
with torch.no_grad():
 g.layers[0].weight[1]=g.layers[0].weight[0];g.layers[0].bias[1]=g.layers[0].bias[0]
before=f.queries;w,b,m,nq=K.recover_layer(f,g,0,'cpu',sampling='design',verbose=False)
assert nq==0 and not m.any() and f.queries==before
assert torch.equal(w,g.layers[0].weight)
print('PASS: callable-only recovery, query counts, gauge, empty/zero budgets, rank-deficient abstention; error',error)
