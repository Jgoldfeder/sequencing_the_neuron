"""Forward-only hybrid recovery and abstention regression tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import scale_normalize_
from kink_solve import recover_layer
def recover_first(teacher, cons, device, **kwargs):
 if "max_regions" in kwargs: kwargs["max_rounds"] = kwargs.pop("max_regions")
 return recover_layer(teacher, cons, 0, device, sampling="design", **kwargs)

torch.set_num_threads(1);torch.manual_seed(4)
T=MLP([1024,8,8,4]).double()
with torch.no_grad():
 for L in T.layers:L.bias.normal_(0,.1)
scale_normalize_(T);G=T.clone()
with torch.no_grad():
 for L in G.layers[:-1]:L.weight.add_(torch.randn_like(L.weight)*.0001)
 G.layers[-1].weight.zero_();G.layers[-1].bias.zero_()
class Oracle:
 def __init__(self):self.n=0
 def __call__(self,x):self.n+=len(x);return T(x)
f=Oracle();w,b,mask,nq=recover_first(f,G,'cpu',verbose=False)
assert mask.all(),mask
n=w.norm(dim=1);err=max(float((w/n[:,None]-T.layers[0].weight).abs().max()),float((b/n-T.layers[0].bias).abs().max()))
assert err<1e-10,err
assert nq==f.n
assert torch.allclose(w.square().sum(1)+b.square(),torch.ones_like(b),atol=1e-14,rtol=0)
for opts in [dict(max_regions=0),dict(only_channels=[])]:
 before=f.n;w,b,m,q=recover_first(f,G,'cpu',verbose=False,**opts)
 assert q==0 and f.n==before and not m.any()
 assert torch.equal(w,G.layers[0].weight) and torch.equal(b,G.layers[0].bias)
# No output jump: leave the guesses alone.
w,b,m,q=recover_first(lambda x:torch.zeros(len(x),4,dtype=x.dtype),G,'cpu',only_channels=[0],verbose=False)
assert not m.any() and torch.equal(w,G.layers[0].weight)
print('PASS: hybrid forward-only recovery, gauge, query count, zero budget, empty request and constant-oracle abstention; error',err)
