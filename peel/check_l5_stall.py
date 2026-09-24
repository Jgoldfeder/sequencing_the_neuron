"""Isolate the unusually weak L4 direction in the cached teacher's peel inverse.
The modified teacher is a diagnostic control only, never a production fix.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from method import _mlp_prefix_inverse,_mlp_prefix_input,_FrontierNet

torch.set_num_threads(1);torch.manual_seed(19)
dims=[3072]+[200]*7+[100]
f=MLP(dims).double();f.load_state_dict(torch.load('teachers/teacher_'+'x'.join(map(str,dims))+'_e25_s1.pt',map_location='cpu',weights_only=True))
t=MLP(dims[1:]).double()
with torch.no_grad():
 for dst,src in zip(t.layers,f.layers[1:]):dst.weight.copy_(src.weight);dst.bias.copy_(src.bias)
 h=.5*torch.randn(128,200,dtype=torch.float64)
 spectra=[]
 for i,L in enumerate(t.layers[:-1]):
  s=torch.linalg.svdvals(L.weight);spectra.append(dict(layer=i+1,smallest=float(s[-1]),condition=float(s[0]/s[-1])))
 results=[]
 for control in [False,True]:
  model=t.clone()
  if control:
   U,s,V=torch.linalg.svd(model.layers[3].weight,full_matrices=False)
   s=s.clamp_min(.005);model.layers[3].weight.copy_((U*s)@V)
  frozen={i:(L.weight,L.bias,torch.ones(200,dtype=torch.bool)) for i,L in enumerate(model.layers[:4])}
  for mode in ['current','linear_solve']:
   if mode=='current':x=_mlp_prefix_inverse(frozen,4,'leaky_relu','cpu')(h)
   else:
    x=h.clone()
    for L in reversed(model.layers[:4]):x=torch.linalg.solve(L.weight,(torch.where(x>=0,x,x/.01)-L.bias).T).T
   actual_h=_mlp_prefix_input(model,x,4);y=model(x);direct=_FrontierNet(model,4)(h)
   row=dict(control_l4_floor=control,inverse=mode,max_input=float(x.abs().max()),hidden_relative=float((actual_h-h).norm()/h.norm()),output_relative=float((y-direct).norm()/direct.norm()))
   results.append(row);print(json.dumps(row),flush=True)
Path('peel/l5_stall_diagnostic.json').write_text(json.dumps(dict(spectra=spectra,results=results),indent=2))
