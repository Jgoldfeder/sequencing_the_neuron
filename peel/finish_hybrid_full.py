"""Continue the saved hybrid experiment after fixing the optional bias-foot gate."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from peel.hybrid_sweep import recover_first
from peel.informative_kinks import recover_layer

torch.set_num_threads(1);saved=torch.load('peel/hybrid_full.pt',map_location='cpu',weights_only=False)
T=MLP([3072,256,256,256]).double().cuda();G=T.clone();T.load_state_dict(saved['teacher']);G.load_state_dict(saved['student'])
failed=[int(c) for c,d in saved['results'][0]['diagnostics'].items() if not d['accepted']]
results=[]
for layer in [0,1]:
 diag={};start=time.perf_counter()
 if layer==0:w,b,mask,nq=recover_first(lambda x:T(x),G,'cuda',only_channels=failed,diagnostics=diag)
 else:w,b,mask,nq=recover_layer(lambda x:T(x),G,1,'cuda',need=512,diagnostics=diag)
 channels=failed if layer==0 else list(range(256))
 with torch.no_grad():
  n=w.norm(dim=1);idx=mask.nonzero()[:,0];G.layers[layer].weight[idx]=w[idx]/n[idx,None];G.layers[layer].bias[idx]=b[idx]/n[idx]
  ew=(G.layers[layer].weight-T.layers[layer].weight).abs();eb=(G.layers[layer].bias-T.layers[layer].bias).abs()
 result=dict(layer=layer,seconds=time.perf_counter()-start,queries=nq,requested=len(channels),accepted=int(mask[channels].sum()),whole_layer_max=max(float(ew.max()),float(eb.max())),whole_layer_mean=float(torch.cat([ew.flatten(),eb]).mean()),diagnostics=diag)
 results.append(result);Path('peel/hybrid_completed.json').write_text(json.dumps(results,indent=2));print('RESULT',json.dumps({k:v for k,v in result.items() if k!='diagnostics'}),flush=True)
 if not mask[channels].all():break
else:
 with torch.no_grad():
  gen=torch.Generator(device='cuda').manual_seed(88)
  x=torch.randn(4096,3072,device='cuda',dtype=torch.float64,generator=gen);h=G.act(G.layers[0](x));h=G.act(G.layers[1](h));A=torch.cat([h,torch.ones_like(h[:,:1])],1)
  sol=torch.linalg.lstsq(A.cpu(),T(x).cpu()).solution.cuda();G.layers[2].weight.copy_(sol[:-1].T);G.layers[2].bias.copy_(sol[-1])
  xv=torch.randn(1024,3072,device='cuda',dtype=torch.float64,generator=gen);err=(G(xv)-T(xv)).abs()
 out=dict(output_max=float(err.max()),output_mean=float(err.mean()),fit_queries=4096,validation_queries=1024)
 Path('peel/hybrid_output.json').write_text(json.dumps(out,indent=2));print('OUTPUT',out,flush=True)
torch.save(dict(student=G.state_dict(),teacher=T.state_dict(),results=results),'peel/hybrid_completed.pt')
