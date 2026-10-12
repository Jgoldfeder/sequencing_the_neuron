"""Nonunit source/destination gauges and partial full-freeze fp64 preservation."""
import copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import scale_normalize_
from method import _cnn_transfer_rows_,_mlp_refinement_record,_mlp_warm_reinit_population

torch.set_num_threads(1);torch.manual_seed(31)
t=MLP([12,8,6,3]).double()
def rescale(m,l,s):
 with torch.no_grad():
  m.layers[l].weight.mul_(s[:,None]);m.layers[l].bias.mul_(s)
  m.layers[l+1].weight.div_(s[None,:])
def canonical(m):
 c=copy.deepcopy(m).double();scale_normalize_(c);return c
src=copy.deepcopy(t);dst=copy.deepcopy(t)
rescale(src,0,torch.logspace(-1,1,8));rescale(dst,0,torch.logspace(1,-1,8))
rescale(src,1,torch.logspace(-1,1,6));rescale(dst,1,torch.logspace(1,-1,6))
idx=torch.arange(6)
with torch.no_grad():dst.layers[1].weight.add_(.01*torch.randn_like(dst.layers[1].weight))
_cnn_transfer_rows_(dst,src,1,idx)
a,b=canonical(dst),canonical(t)
assert (a.layers[1].weight-b.layers[1].weight).abs().max()<1e-12
assert (a.layers[1].bias-b.layers[1].bias).abs().max()<1e-12
# The consensus has only fp32 approximations; previously solved layers must not use them.
cons=canonical(src).float();pm={0:torch.ones(8,dtype=torch.bool),1:torch.tensor([True,True,False,False,False,False])}
frozen={0:(src.layers[0].weight.detach().clone(),src.layers[0].bias.detach().clone(),pm[0].clone())}
record,mask=_mlp_refinement_record(cons,frozen,src,pm,1)
assert torch.equal(record.layers[0].weight,src.layers[0].weight)
assert torch.equal(record.layers[1].weight[:2],src.layers[1].weight[:2])
assert torch.equal(mask,pm[1])
frozen[1]=(record.layers[1].weight.detach().clone(),record.layers[1].bias.detach().clone(),mask)
rebuilt=_mlp_warm_reinit_population([dst.float()],[12,8,6,3],'leaky_relu','cpu',frozen)[0]
a=canonical(rebuilt)
assert (a.layers[1].weight[:2]-b.layers[1].weight[:2]).abs().max()<2e-7
assert torch.equal(frozen[0][0],src.layers[0].weight)
print('PASS source/destination scales, original fp64 prefix, partial masks and warm rebuild')
