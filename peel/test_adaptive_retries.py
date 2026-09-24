"""Compare retries on the user's cached teacher with controlled layer-2 guesses."""
import sys,json,time,math,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import scale_normalize_
from peel.informative_kinks import recover_layer
p=argparse.ArgumentParser(); p.add_argument('--channels',type=int,nargs='+',default=[74,166]);p.add_argument('--all',action='store_true');p.add_argument('--strategies',nargs='+',default=['legacy','adaptive']);p.add_argument('--noise',type=float,default=.003);p.add_argument('--output',default='peel/adaptive_teacher_retry.json'); a=p.parse_args()
torch.set_num_threads(1)
dims=[3072]+[200]*7+[100]
path=Path('teachers/teacher_'+'x'.join(map(str,dims))+'_e25_s1.pt')
assert path.exists(),path
full=MLP(dims);full.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
T=MLP(dims[1:]).double()
with torch.no_grad():
 for dst,src in zip(T.layers,full.layers[1:]): dst.weight.copy_(src.weight);dst.bias.copy_(src.bias)
scale_normalize_(T);T=T.cuda();G=T.clone();gen=torch.Generator(device='cuda').manual_seed(1)
with torch.no_grad():
 L=G.layers[1]; L.weight.add_(a.noise*torch.randn(L.weight.shape,device='cuda',dtype=torch.float64,generator=gen));L.bias.add_(a.noise*torch.randn(L.bias.shape,device='cuda',dtype=torch.float64,generator=gen))
 for L in G.layers[2:]:L.weight.zero_();L.bias.zero_()
results=[];cs=list(range(200)) if a.all else a.channels
for strategy in a.strategies:
 diag={}; start=time.perf_counter()
 w,b,mask,nq=recover_layer(lambda x:T(x),G,1,'cuda',only_channels=cs,gen=torch.Generator(device='cuda').manual_seed(1),retry_strategy=strategy,diagnostics=diag,verbose=False)
 norm=w.norm(dim=1);err=torch.maximum((w/norm[:,None]-T.layers[1].weight).abs().amax(1),(b/norm-T.layers[1].bias).abs())
 result=dict(strategy=strategy,noise=a.noise,seconds=time.perf_counter()-start,queries=nq,accepted=int(mask[cs].sum()),channels=len(cs),maximum=float(err[cs].max()),neurons={c:dict(error=float(err[c]),**diag[c]) for c in cs})
 results.append(result);Path(a.output).write_text(json.dumps(results,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='neurons'}),flush=True)
