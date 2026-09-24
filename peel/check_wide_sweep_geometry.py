"""Teacher-internal geometry diagnosis only; never used to recover weights."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import scale_normalize_
torch.set_num_threads(1);torch.manual_seed(0);d=3072
T=MLP([d]*4).double()
with torch.no_grad():
 for L in T.layers:L.bias.normal_(0,.1)
scale_normalize_(T)
ng=torch.Generator().manual_seed(1)
# CPU RNG differs from CUDA RNG: load CUDA-generated guesses in a fresh
# process on the same unused device as the main test, but solve small RHS on CPU.
ng=torch.Generator(device='cuda').manual_seed(1);results=[]
with torch.no_grad():
 for layer in range(2):
  W=T.layers[layer].weight+.01/d**.5*torch.randn(d,d,device='cuda',dtype=torch.float64,generator=ng).cpu()
  b=T.layers[layer].bias+.005*torch.randn(d,device='cuda',dtype=torch.float64,generator=ng).cpu()
  norm=W.norm(dim=1);W=W/norm[:,None];b=b/norm
  lu,piv=torch.linalg.lu_factor(W)
  gen=torch.Generator(device='cuda').manual_seed(1234)
  for c in range(2):
   unit=torch.zeros(d,1,dtype=torch.float64);unit[c]=1
   u=torch.linalg.lu_solve(lu,piv,unit)[:,0]
   for attempt in range(2):
    z=(torch.randint(0,2,(d,),device='cuda',generator=gen)*2-1).double().cpu()*2;z[c]=0
    x=torch.linalg.lu_solve(lu,piv,(z-b)[:,None])[:,0]
    left=T.layers[layer](x-u);right=T.layers[layer](x+u)
    crosses=left*right<0
    r=dict(layer=layer,channel=c,attempt=attempt,target_crossed=bool(crosses[c]),other_neurons_crossed=int(crosses.sum())-int(crosses[c]),max_input=float(x.abs().max()),guess_target_residual=float((W@x+b-z).abs().max()),true_target_center=float(T.layers[layer](x)[c]))
    results.append(r);print(json.dumps(r),flush=True)
Path('peel/wide_sweep_geometry.json').write_text(json.dumps(results,indent=2))
