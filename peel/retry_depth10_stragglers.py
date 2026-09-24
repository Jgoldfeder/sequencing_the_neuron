"""Diagnose the two unfinished neurons from run_depth10.pt without replaying its prefix."""
import json
import math
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nets import MLP
from align import scale_normalize_
from peel.informative_kinks import recover_layer

torch.set_num_threads(1)
saved=torch.load('peel/run_depth10.pt',map_location='cpu',weights_only=False)
dims=saved['dims']; l=saved['next_layer']
assert l==6, f'Checkpoint advanced to layer {l}; this diagnostic expects the original layer-6 failure'
torch.manual_seed(saved['seed'])
T=MLP(dims).double()
with torch.no_grad():
    for layer in T.layers: layer.bias.normal_(0,.1)
scale_normalize_(T); T=T.cuda().eval()
G=T.clone(); G.load_state_dict(saved['student'])
ng=torch.Generator(device='cuda'); ng.set_state(saved['noise_state'])
with torch.no_grad():
    wt=T.layers[l].weight; bt=T.layers[l].bias
    G.layers[l].weight.copy_(wt+.01*torch.randn(wt.shape,device='cuda',dtype=torch.float64,generator=ng)/math.sqrt(wt.shape[1]))
    G.layers[l].bias.copy_(bt+.005*torch.randn(bt.shape,device='cuda',dtype=torch.float64,generator=ng))
    for layer in G.layers[l+1:]: layer.weight.zero_(); layer.bias.zero_()
diag={}
w,b,mask,nq=recover_layer(T,G,l,'cuda',only_channels=[168,172],
                          gen=torch.Generator(device='cuda').manual_seed(1),
                          max_rounds=160,diagnostics=diag)
with torch.no_grad():
    n=w.norm(dim=1); w=w/n[:,None]; b=b/n
    err=torch.maximum((w-wt).abs().amax(1),(b-bt).abs())
result=dict(layer=l,queries=nq,neurons={c:dict(error=err[c].item(),**diag[c]) for c in [168,172]})
print('RETRY_RESULT',json.dumps(result),flush=True)
Path('peel/depth10_straggler_retry.json').write_text(json.dumps(result,indent=2)+'\n')
