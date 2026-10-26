import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from peel.input_subspace import project_first_rowspace
torch.manual_seed(31)
m=MLP([24,8,6,3]).double();calls=[]
def oracle(x):calls.append(x.shape);return m(x)
t,g,q=project_first_rowspace(oracle,m)
x=torch.randn(20,8,dtype=torch.double,requires_grad=True)
y=t(x);yy=g(x)
torch.testing.assert_close(y,yy,rtol=1e-12,atol=1e-12)
torch.testing.assert_close(torch.autograd.grad(y.sum(),x)[0],torch.autograd.grad(yy.sum(),x)[0],rtol=1e-12,atol=1e-12)
assert calls==[torch.Size([20,24])]
assert torch.linalg.matrix_rank(g.layers[0].weight)==8
print('PASS forward-only projection, output/gradient equivalence, preserved first-layer reachability')
