import sys,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from peel.local_kink_refine import recover_layer
torch.set_num_threads(1);torch.manual_seed(5)
t=MLP([16,8,6,3]).double();g=t.clone()
with torch.no_grad():g.layers[1].weight.add_(torch.randn_like(g.layers[1].weight)*1e-4)
queries=[0]
def oracle(x):
 queries[0]+=len(x)
 with torch.no_grad():return t(x)
a={};w,b,ok,q=recover_layer(oracle,g,1,need=80,rounds=12,pool=96,batch=32,archive=a,verbose=False)
assert q==queries[0] and q<=(12+8)*6*32*8
assert ok.all(),a['stats']
truth=torch.cat([t.layers[1].weight,t.layers[1].bias[:,None]],1);truth/=truth.norm(dim=1,keepdim=True)
actual=torch.cat([w,b[:,None]],1);assert float((actual-truth).abs().max())<1e-9
assert all(len(x)>0 for x in a['points'].values())
# Load just the pure helper without executing the command-line driver.
source=ast.parse(Path('refine_snapshot_mlp.py').read_text());node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='distinct_candidate')
ns={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'helper','exec'),ns)
fn=ns['distinct_candidate'];wr=torch.eye(3,dtype=torch.double);br=torch.zeros(3,dtype=torch.double);mask=torch.tensor([True,False,False])
assert not fn(1,wr[0],br[0],wr,br,mask)
assert not fn(1,-wr[0],br[0],wr,br,mask)
assert fn(1,wr[1],br[1],wr,br,mask)
print('PASS forward-only callable, exact-prefix precision, bounded query accounting, retained points, duplicate rejection')

# A guessed duplicate must not produce two accepted copies of one surface.
g=t.clone()
with torch.no_grad():
 g.layers[1].weight[1].copy_(g.layers[1].weight[0]);g.layers[1].bias[1].copy_(g.layers[1].bias[0])
w,b,ok,q=recover_layer(oracle,g,1,channels=[0,1],need=80,rounds=12,pool=96,batch=32,verbose=False)
assert ok[0] and not ok[1],ok
print('PASS duplicate guessed neuron is not counted as a second recovered neuron')
