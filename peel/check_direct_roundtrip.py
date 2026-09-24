"""Check whether prefix inversion really gives a direct tail oracle in fp64."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from method import _mlp_prefix_inverse, _mlp_prefix_input, _FrontierNet
from align import scale_normalize_
torch.set_num_threads(1);torch.manual_seed(19)
dims=[3072]+[200]*7+[100]
f=MLP(dims).double();f.load_state_dict(torch.load('teachers/teacher_'+'x'.join(map(str,dims))+'_e25_s1.pt',map_location='cpu',weights_only=True))
t=MLP(dims[1:]).double()
with torch.no_grad():
 for dst,src in zip(t.layers,f.layers[1:]):dst.weight.copy_(src.weight);dst.bias.copy_(src.bias)
h=torch.randn(128,200,dtype=torch.float64);out=[]
with torch.no_grad():
 for gauge in ['original','normalized']:
  if gauge=='normalized':scale_normalize_(t)
  for n in range(1,6):
   frozen={i:(L.weight,L.bias,torch.ones(200,dtype=torch.bool)) for i,L in enumerate(t.layers[:n])}
   direct=_FrontierNet(t,n)(h)
   for mode in ['current','solve']:
    if mode=='current':x=_mlp_prefix_inverse(frozen,n,'leaky_relu','cpu')(h)
    else:
     x=h.clone()
     for L in reversed(t.layers[:n]):x=torch.linalg.solve(L.weight,(torch.where(x>=0,x,x/.01)-L.bias).T).T
    hx=_mlp_prefix_input(t,x,n);y=t(x)
    r=dict(gauge=gauge,prefix=n,mode=mode,max_x=float(x.abs().max()),h_max_error=float((hx-h).abs().max()),h_relative=float((hx-h).norm()/h.norm()),output_relative=float((y-direct).norm()/direct.norm()))
    out.append(r);print(json.dumps(r),flush=True)
Path('peel/direct_roundtrip.json').write_text(json.dumps(out,indent=2))
