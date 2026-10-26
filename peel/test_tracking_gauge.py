import sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from verify_layer1 import _Oracle
import kink_solve as K
torch.set_num_threads(1);torch.manual_seed(83)
teacher=MLP([6,4,2],act='leaky_relu').double();teacher.requires_grad_(False)
guess=copy.deepcopy(teacher)
with torch.no_grad():guess.layers[0].weight.add_(torch.randn_like(guess.layers[0].weight)*1e-3)
x=torch.randn(5,6,dtype=torch.float64);w=teacher.layers[0].weight[0];b=teacher.layers[0].bias[0];x-=((x@w+b)/(w@w))[:,None]*w
anchors={0:(x,torch.zeros(len(x),dtype=torch.long))};records=[]
for scale in [1.,1000.]:
 model=copy.deepcopy(guess)
 with torch.no_grad():model.layers[0].weight[0]*=scale;model.layers[0].bias[0]*=scale;model.layers[1].weight[:,0]/=scale
 queries=[]
 def oracle(z):queries.append(z.detach().clone());return teacher(z)
 bb=_Oracle(oracle);raw={}
 K.track_layer(bb,model,0,anchors,torch.Generator().manual_seed(9),need=30,max_steps=50,archive=raw)
 records.append((torch.cat(queries),raw,bb.n))
assert records[0][2]==records[1][2]
assert torch.allclose(records[0][0],records[1][0],rtol=1e-9,atol=1e-9)
assert 0 in records[0][1] and 0 in records[1][1]
print('PASS: equivalent row gauges generate matching tracking queries and counts')
