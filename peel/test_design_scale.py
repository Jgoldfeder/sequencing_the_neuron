"""Candidate locations/directions/radii must ignore a positive neuron rescaling."""
import sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from peel.informative_kinks import candidate_pool

torch.set_num_threads(1);torch.manual_seed(8)
t=MLP([12,8,6,3]).double()
for layer in [0,1]:
 base=candidate_pool(t,layer,[0],torch.Generator().manual_seed(2),96,(1.,4.,16.),{0:.02})[0]
 for scale in [.001,100.]:
  g=copy.deepcopy(t)
  with torch.no_grad():g.layers[layer].weight[0]*=scale;g.layers[layer].bias[0]*=scale
  actual=candidate_pool(g,layer,[0],torch.Generator().manual_seed(2),96,(1.,4.,16.),{0:.02})[0]
  for x,y in zip(base,actual):assert torch.allclose(x,y,atol=1e-8,rtol=1e-8),(layer,scale,(x-y).abs().max())
print('PASS candidate geometry invariant across 100000x neuron scale range')
