"""Compare forward-only Jacobian-jump experiment with informative kink points."""
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from nets import MLP
from align import scale_normalize_
from peel.jump_kinks import recover_neuron
from peel.informative_kinks import recover_layer

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--widths',type=int,nargs='+',default=[200,400])
p.add_argument('--hidden-layers',type=int,default=7)
p.add_argument('--layer',type=int,default=3)
p.add_argument('--channels',type=int,default=4)
p.add_argument('--regions',type=int,default=8)
p.add_argument('--prefix-noise',type=float,nargs='+',default=[0.,1e-10])
p.add_argument('--device',default='cuda')
p.add_argument('--output',default='peel/jump_comparison.json')
p.add_argument('--skip-baseline',action='store_true')
a=p.parse_args()
if not 0<=a.layer<a.hidden_layers or min(a.widths)<a.channels: p.error('invalid layer/channels')
torch.set_num_threads(1); results=[]
for width in a.widths:
    torch.manual_seed(0)
    T=MLP([width]+[width]*a.hidden_layers+[100]).double()
    with torch.no_grad():
        for L in T.layers: L.bias.normal_(0,.1)
    scale_normalize_(T); T=T.to(a.device).eval()
    for noise in a.prefix_noise:
        G=T.clone(); ng=torch.Generator(device=a.device).manual_seed(1)
        with torch.no_grad():
            L=G.layers[a.layer]
            L.weight.add_(.01/math.sqrt(width)*torch.randn(L.weight.shape,device=a.device,dtype=torch.float64,generator=ng))
            L.bias.add_(.005*torch.randn(L.bias.shape,device=a.device,dtype=torch.float64,generator=ng))
            for L in G.layers[:a.layer]:
                L.weight.add_(noise/math.sqrt(width)*torch.randn(L.weight.shape,device=a.device,dtype=torch.float64,generator=ng))
                L.bias.add_(noise*torch.randn(L.bias.shape,device=a.device,dtype=torch.float64,generator=ng))
            for L in G.layers[a.layer+1:]: L.weight.zero_(); L.bias.zero_()
        def errors(v,c):
            truth=torch.cat([T.layers[a.layer].weight[c],T.layers[a.layer].bias[c:c+1]]).detach().cpu().numpy()
            truth/=np.linalg.norm(truth[:-1]); e=abs(v-truth)
            return dict(mean=float(e.mean()),maximum=float(e.max()))
        start=time.perf_counter(); entries=[]
        for c in range(a.channels):
            gen=torch.Generator(device=a.device).manual_seed(100+c)
            dense,it,diag=recover_neuron(lambda x:T(x),G,a.layer,c,gen,regions=a.regions)
            entry=dict(channel=c,diagnostics=diag,dense_error=errors(dense,c) if dense is not None else None,iterative_error=errors(it,c) if it is not None else None)
            entries.append(entry); print('NEURON',width,noise,json.dumps(entry),flush=True)
        result=dict(method='jump',width=width,layer=a.layer,prefix_noise=noise,seconds=time.perf_counter()-start,queries=sum(e['diagnostics']['queries'] for e in entries),entries=entries)
        results.append(result)
        if not a.skip_baseline:
            start=time.perf_counter(); diag={}
            W,b,mask,nq=recover_layer(lambda x:T(x),G,a.layer,a.device,only_channels=list(range(a.channels)),gen=torch.Generator(device=a.device).manual_seed(1),need=max(400,2*width),pool=max(768,2*width),diagnostics=diag,verbose=False)
            entries=[]
            for c in range(a.channels):
                v=torch.cat([W[c],b[c:c+1]]).cpu().numpy(); v/=np.linalg.norm(v[:-1])
                entries.append(dict(channel=c,accepted=bool(mask[c]),error=errors(v,c)))
            results.append(dict(method='points',width=width,layer=a.layer,prefix_noise=noise,seconds=time.perf_counter()-start,queries=nq,entries=entries))
        Path(a.output).parent.mkdir(parents=True,exist_ok=True)
        Path(a.output).write_text(json.dumps(results,indent=2))
        print('RESULT',json.dumps([{k:v for k,v in r.items() if k!='entries'} for r in results[-2:]]),flush=True)
