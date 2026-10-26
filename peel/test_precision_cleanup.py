"""Synthetic tests for the explicit float32 prior and independent constraints."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from peel.float32_lattice import recover_row
from peel.precision_prefix import sparse_fit

def test_lattice_and_sparse_constraints():
 torch.set_num_threads(1);torch.manual_seed(29)
 original=torch.randn(64).double();original[0]=.75
 original=original.float().double();unit=original/original.norm()
 noisy=unit+torch.randn_like(unit)*1e-13
 recovered,info=recover_row(noisy,atol=3e-12)
 assert recovered is not None,info
 factor=(recovered@original)/(original@original)
 assert torch.max(torch.abs(recovered/factor-original))<1e-9
 bias=torch.tensor(.125,dtype=torch.float64)*factor
 h=torch.randn(256,64,dtype=torch.float64)
 h-=((h@recovered+bias)/recovered.square().sum())[:,None]*recovered
 # Corrupt sparse coefficients plus bias, then recover from independent planes.
 ids=[2,8,15];guess=recovered.clone();guess[ids]+=1e-5
 result=sparse_fit(h,guess,bias+1e-5,ids)
 assert result is not None
 w,b,quality=result
 assert quality['validation']>=.9
 assert torch.equal(w,recovered)
 assert b==bias
 # A wrong fixed coefficient cannot be repaired by moving unrelated columns.
 guess=recovered.clone();guess[35]+=.1
 result=sparse_fit(h,guess,bias,ids)
 assert result is None or result[2]['validation']<.9
 # A generic float64 row does not satisfy this metadata-dependent prior.
 unsupported=torch.randn(64,dtype=torch.float64)
 w,info=recover_row(unsupported/unsupported.norm(),atol=3e-12)
 assert w is None
 print('PASS: float32 lattice, sparse recovery, held-out rejection, unsupported float64')
if __name__=='__main__':test_lattice_and_sparse_constraints()

def test_duplicate_prior_row():
 from nets import MLP
 from peel.precision_prefix import cleanup_layer
 torch.manual_seed(42)
 model=MLP([64,3,2],act='leaky_relu').double()
 with torch.no_grad():
  for p in model.parameters():p.copy_(p.float().double())
  model.layers[0].weight[1]=model.layers[0].weight[0]
  model.layers[0].bias[1]=model.layers[0].bias[0]
  w=model.layers[0].weight[0];b=model.layers[0].bias[0]
  h=torch.randn(256,64,dtype=torch.float64);h-=((h@w+b)/(w@w))[:,None]*w
 def oracle(x):raise AssertionError('Archive-only cleanup must not query')
 _,ok,q,_,stats=cleanup_layer(oracle,model,0,points={1:h},channels=[1],frozen_mask=torch.tensor([True,False,False]))
 assert not bool(ok.any()) and q==0
 assert stats[1]['reason']=='duplicate_solved_row',stats
 print('PASS: duplicate frozen row rejected without oracle queries')
if __name__=='__main__':test_duplicate_prior_row()
