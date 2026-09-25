"""Synthetic good-guess experiment using production design-refine dispatch."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import scale_normalize_
from kink_solve import recover_layer
p=argparse.ArgumentParser();p.add_argument('--channels',type=int,default=4);p.add_argument('--only',type=int,nargs='+');p.add_argument('--output',default='peel/hybrid_probe.json');p.add_argument('--seed',type=int,default=0);p.add_argument('--second',action='store_true');a=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(a.seed)
T=MLP([3072,256,256,256]).double()
with torch.no_grad():
 for L in T.layers:L.bias.normal_(0,.1)
scale_normalize_(T);T=T.cuda();G=T.clone();ng=torch.Generator(device='cuda').manual_seed(1)
with torch.no_grad():
 for L in G.layers[:-1]:
  L.weight.add_(.01/L.weight.shape[1]**.5*torch.randn(L.weight.shape,device='cuda',dtype=torch.float64,generator=ng));L.bias.add_(.005*torch.randn(L.bias.shape,device='cuda',dtype=torch.float64,generator=ng))
 G.layers[-1].weight.zero_();G.layers[-1].bias.zero_()
channels=a.only if a.only is not None else list(range(a.channels))
if a.second and channels != list(range(256)):
 raise ValueError('--second requires all 256 first-layer neurons (--channels 256, no --only)')
results=[]
for layer in range(2 if a.second else 1):
 if layer and a.channels!=256:raise ValueError('Second-layer test needs the complete recovered prefix (--channels 256)')
 start=time.perf_counter();diag={}
 w,b,mask,nq=recover_layer(lambda x:T(x),G,layer,'cuda',sampling='design',only_channels=channels,need=512,diagnostics=diag)
 with torch.no_grad():
  n=w.norm(dim=1);w=w/n[:,None];b=b/n
  ew=(w-T.layers[layer].weight).abs();eb=(b-T.layers[layer].bias).abs();err=torch.maximum(ew.max(1).values,eb)[channels]
  idx=mask.nonzero()[:,0];G.layers[layer].weight[idx]=w[idx];G.layers[layer].bias[idx]=b[idx]
 result=dict(layer=layer,seconds=time.perf_counter()-start,queries=nq,accepted=int(mask[channels].sum()),channels=len(channels),max_accepted=float(err[mask[channels]].max()) if mask[channels].any() else None,mean_accepted=float(torch.cat([ew[channels][mask[channels]].flatten(),eb[channels][mask[channels]]]).mean()) if mask[channels].any() else None,diagnostics=diag)
 results.append(result);Path(a.output).write_text(json.dumps(results,indent=2));print('RESULT',json.dumps({k:v for k,v in result.items() if k!='diagnostics'}),flush=True)
 if not mask[channels].all():break
if a.second and len(results)==2 and bool(mask.all()):
 with torch.no_grad():
  gen=torch.Generator(device='cuda').manual_seed(88)
  x=torch.randn(4096,3072,device='cuda',dtype=torch.float64,generator=gen)
  h=G.act(G.layers[0](x));h=G.act(G.layers[1](h));A=torch.cat([h,torch.ones_like(h[:,:1])],1)
  sol=torch.linalg.lstsq(A.cpu(),T(x).cpu()).solution.cuda()
  G.layers[2].weight.copy_(sol[:-1].T);G.layers[2].bias.copy_(sol[-1])
  xv=torch.randn(1024,3072,device='cuda',dtype=torch.float64,generator=gen);err=(G(xv)-T(xv)).abs()
 out=dict(output_max=float(err.max()),output_mean=float(err.mean()),fit_queries=4096,validation_queries=1024)
 Path(a.output).with_suffix('.output.json').write_text(json.dumps(out,indent=2));print('OUTPUT',out,flush=True)
if len(channels)==256:torch.save(dict(student=G.state_dict(),teacher=T.state_dict(),results=results),str(Path(a.output).with_suffix('.pt')))
