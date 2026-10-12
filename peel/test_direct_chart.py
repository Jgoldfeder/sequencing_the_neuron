"""CPU tests of chart values/gradients, deep exact-tail equivalence and warm gauges."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from method import _FrontierNet, _mlp_warm_reinit_population
from peel.direct_chart import PrefixChart, ChartStudent

torch.set_num_threads(1);torch.manual_seed(19)
path=Path('teachers/teacher_3072x200x200x200x200x200x200x200x100_e25_s1.pt')
if path.exists():
 full=MLP([3072]+[200]*7+[100]).double();full.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
 t=MLP([200]*7+[100]).double()
 with torch.no_grad():
  for dst,src in zip(t.layers,full.layers[1:]):dst.weight.copy_(src.weight);dst.bias.copy_(src.bias)
else:
 t=MLP([32]*7+[8]).double()
report=[]
for n in range(1,6):
 frozen={i:(L.weight,L.bias,torch.ones(L.weight.shape[0],dtype=torch.bool)) for i,L in enumerate(t.layers[:n])}
 chart=PrefixChart(frozen,n,'leaky_relu','cpu')
 z=torch.randn(128,t.layers[n].weight.shape[1],dtype=torch.float64,requires_grad=True)
 x,h=chart.map(z);direct=_FrontierNet(t,n)(h);actual=t(x)
 rel=float((actual-direct).norm()/direct.norm().clamp_min(1e-30))
 ga=torch.autograd.grad(actual.square().sum(),z,retain_graph=True)[0]
 gd=torch.autograd.grad(direct.square().sum(),z)[0]
 grel=float((ga-gd).norm()/gd.norm().clamp_min(1e-30))
 chart.checked(z)
 assert rel<1e-9,(n,rel)
 assert grel<1e-7,(n,grel)
 assert float((x-chart.x0).norm(dim=1).max())<=chart.radius*1.000001
 assert float((h-h.mean(0)).norm())>0,'collapsed chart'
 row=dict(prefix=n,rank=chart.rank,output_relative=rel,gradient_relative=grel,max_raw=float(x.abs().max()),hidden_spread=float(h.std(0).mean()))
 report.append(row);print(row,flush=True)
# Arbitrary positive row scalings are compensated in the next layer at warm freeze.
torch.manual_seed(7);m=MLP([8,7,6,3]);ref=m.clone();x=torch.randn(128,8)
with torch.no_grad():
 for l in range(2):
  scale=torch.logspace(-2,2,ref.layers[l].weight.shape[0])
  ref.layers[l].weight.mul_(scale[:,None]);ref.layers[l].bias.mul_(scale)
  ref.layers[l+1].weight.div_(scale[None,:])
frozen={l:(ref.layers[l].weight.double(),ref.layers[l].bias.double(),torch.ones(ref.layers[l].weight.shape[0],dtype=torch.bool)) for l in range(2)}
out=_mlp_warm_reinit_population([m],[8,7,6,3],'leaky_relu','cpu',frozen)[0]
assert torch.allclose(out(x),m(x),atol=1e-6,rtol=1e-5),(out(x)-m(x)).abs().max()
Path('peel/direct_chart_results.json').write_text(json.dumps(report,indent=2))
print('PASS: local direct values, gradients, bounded inputs, and warm scale preservation')
