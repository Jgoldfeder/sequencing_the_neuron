"""Exact-prefix ablation and recursive validation for reconstruction-designed queries."""
import argparse
import json
import math
import sys
import time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nets import MLP
from align import scale_normalize_
from peel import informative_kinks as I
import kink_solve as K

p=argparse.ArgumentParser(description='Recover a synthetic leaky-ReLU network from good layer guesses using forward teacher queries.')
p.add_argument('--hidden-layers', type=int, default=7, help='Number of hidden layers; input and linear output are excluded (default: 7)')
p.add_argument('--width', type=int, default=200, help='Neurons in each hidden layer (default: 200)')
p.add_argument('--input-dim', type=int, help='Input dimension (default: same as --width)')
p.add_argument('--output-dim', type=int, default=100)
p.add_argument('--device', default='cuda', help='Torch device, e.g. cuda, cuda:1, or cpu')
p.add_argument('--layers', type=int, nargs='+', default=None, help='Zero-based hidden layers for exact-prefix tests; ignored with --recursive')
p.add_argument('--channels', type=int, help='Neurons per exact-prefix test (default: min(12, width)); recursive mode always tests all')
p.add_argument('--recursive', action='store_true')
p.add_argument('--selection', choices=['design','random','isolated','track'], default='design')
p.add_argument('--scales', type=float, nargs='+', default=[1,4,16])
p.add_argument('--seed', type=int, default=0)
p.add_argument('--need', type=int, help='Target kink points per neuron (default: max(400, 2*largest layer input dimension))')
p.add_argument('--rounds', type=int, help='Maximum sampling rounds; default scales with point target, at least 160. Completed neurons stop early.')
p.add_argument('--output')
p.add_argument('--checkpoint')
p.add_argument('--resume')
a=p.parse_args()
a.input_dim = a.width if a.input_dim is None else a.input_dim
for name in ('hidden_layers', 'width', 'input_dim', 'output_dim'):
    if getattr(a, name) < 1:
        p.error(f'--{name.replace("_", "-")} must be positive')
if a.rounds is not None and a.rounds < 1:
    p.error('--rounds must be positive')
a.channels = min(12, a.width) if a.channels is None else a.channels
if not 1 <= a.channels <= a.width:
    p.error('--channels must be between 1 and --width')
a.need = max(400, 2*max(a.input_dim, a.width)) if a.need is None else a.need
if a.need < 1 or not a.scales or min(a.scales) <= 0:
    p.error('--need and all --scales must be positive')
if a.layers is None:
    a.layers = [l for l in [3,5,6] if l < a.hidden_layers] or [a.hidden_layers-1]
if not a.recursive and any(l < 0 or l >= a.hidden_layers for l in a.layers):
    p.error('--layers indices must be between 0 and --hidden-layers minus 1')
dims = [a.input_dim]+[a.width]*a.hidden_layers+[a.output_dim]
dev = a.device
for filename in (a.output, a.checkpoint):
    if filename:
        Path(filename).expanduser().parent.mkdir(parents=True, exist_ok=True)
print('CONFIG', json.dumps(dict(dims=dims, device=dev, seed=a.seed, recursive=a.recursive,
                               selection=a.selection, points=a.need,
                               sampling_rounds=a.rounds if a.rounds is not None else "auto",
                               guess_weight_noise=.01, guess_bias_noise=.005)), flush=True)
torch.set_num_threads(1)
torch.manual_seed(a.seed)
T=MLP(dims).double()
with torch.no_grad():
    for L in T.layers: L.bias.normal_(0,.1)
scale_normalize_(T); T=T.to(dev).eval()
G=T.clone(); results=[]
noise_gen=torch.Generator(device=dev).manual_seed(1)
gen=torch.Generator(device=dev).manual_seed(1)
first_layer=0
if a.resume:
    if not a.recursive:
        raise ValueError('--resume requires --recursive')
    saved=torch.load(a.resume, map_location=dev, weights_only=False)
    if saved['seed'] != a.seed:
        raise ValueError('Checkpoint teacher seed does not match')
    saved_dims = saved.get('dims')
    if saved_dims is None:  # Original fixed-depth checkpoints.
        weights = [saved['student'][f'layers.{i}.weight'] for i in range(len(saved['student'])//2)]
        saved_dims = [weights[0].shape[1]]+[w.shape[0] for w in weights]
    if saved_dims != dims:
        raise ValueError(f'Checkpoint dimensions {saved_dims} do not match requested {dims}')
    G.load_state_dict(saved['student'])
    noise_gen.set_state(saved['noise_state'].cpu())
    if 'query_state' in saved:
        gen.set_state(saved['query_state'].cpu())
    results=saved['results']; first_layer=saved['next_layer']
layers=range(first_layer,a.hidden_layers) if a.recursive else a.layers
for l in layers:
    if not a.recursive: G=T.clone()
    gen=torch.Generator(device=dev).manual_seed(1)
    # Keep layer guesses equal across separate exact-prefix ablations.
    ng=noise_gen if a.recursive else torch.Generator(device=dev).manual_seed(1)
    with torch.no_grad():
        wt=T.layers[l].weight; bt=T.layers[l].bias
        G.layers[l].weight.copy_(wt+.01*torch.randn(wt.shape, device=dev,dtype=torch.float64,generator=ng)/math.sqrt(wt.shape[1]))
        G.layers[l].bias.copy_(bt+.005*torch.randn(bt.shape,device=dev,dtype=torch.float64,generator=ng))
        # The solver must not depend on a correct downstream reconstruction.
        for L in G.layers[l+1:]:
            L.weight.zero_(); L.bias.zero_()
    channels=list(range(a.width if a.recursive else a.channels))
    diag={}; start=time.time()
    solver=K.recover_layer if a.selection=='track' else I.recover_layer
    W,b,mask,nq=solver(T,G,l,dev,only_channels=channels,gen=gen,
                       scales=tuple(a.scales),selection=a.selection,
                       need=a.need,max_rounds=a.rounds,diagnostics=diag,
                       pool=max(768, 2*max(a.input_dim,a.width)))
    with torch.no_grad():
        norm=W.norm(dim=1); W=W/norm[:,None]; b=b/norm
        ew=(W-wt).abs()[channels]; eb=(b-bt).abs()[channels]
        e=torch.maximum(ew.amax(1),eb)
        mean_absolute=torch.cat([ew.flatten(),eb]).mean().item()
    result=dict(dims=dims,seed=a.seed,recursive=a.recursive,layer=l,selection=a.selection,scales=a.scales,refined=int(mask[channels].sum()),
                channels=len(channels),mean_absolute_parameter_error=mean_absolute,
                mean_neuron_max_error=e.mean().item(),maximum=e.max().item(),median=e.median().item(),
                accepted_max=e[mask[channels]].max().item() if mask[channels].any() else None,
                below_1e12=int((e<1e-12).sum()),below_1e10=int((e<1e-10).sum()),
                queries=nq,seconds=time.time()-start,diagnostics=diag)
    results.append(result)
    print('RESULT',json.dumps({k:v for k,v in result.items() if k!='diagnostics'}),flush=True)
    if a.output:
        with open(a.output,'w') as f: json.dump(results,f,indent=2)
    if a.recursive:
        if not bool(mask.all()):
            print('STOP: incomplete layer, refusing to propagate failed guesses',flush=True)
            break
        with torch.no_grad():
            G.layers[l].weight.copy_(W); G.layers[l].bias.copy_(b)
        if a.checkpoint:
            torch.save(dict(student=G.state_dict(), noise_state=noise_gen.get_state(),
                            query_state=gen.get_state(),
                            results=results, next_layer=l+1, seed=a.seed, dims=dims), a.checkpoint)
else:
    if a.recursive:
        with torch.no_grad():
            if str(dev).startswith("cuda"): torch.cuda.synchronize()
            output_start=time.perf_counter()
            nfit=max(4000, 2*(a.width+1))
            X=torch.randn(nfit,a.input_dim,device=dev,dtype=torch.float64,generator=gen)
            H=K._phi(G,X,a.hidden_layers); A=torch.cat([H,torch.ones_like(H[:,:1])],1)
            sol=torch.linalg.lstsq(A.cpu(),T(X).cpu()).solution.to(dev)
            G.layers[-1].weight.copy_(sol[:-1].T); G.layers[-1].bias.copy_(sol[-1])
            if str(dev).startswith("cuda"): torch.cuda.synchronize()
            output_seconds=time.perf_counter()-output_start
            error=torch.cat([(G.layers[-1].weight-T.layers[-1].weight).abs().flatten(),
                             (G.layers[-1].bias-T.layers[-1].bias).abs()])
            X=torch.randn(2000,a.input_dim,device=dev,dtype=torch.float64,generator=gen)
            output=dict(seconds=output_seconds,mean_absolute_parameter_error=error.mean().item(),parameter_error=max((G.layers[-1].weight-T.layers[-1].weight).abs().max().item(),
                                            (G.layers[-1].bias-T.layers[-1].bias).abs().max().item()),
                        function_gap=(G(X)-T(X)).abs().max().item(), recovery_queries=nfit,
                        evaluation_queries=2000)
            print('OUTPUT',json.dumps(output),flush=True)
            results[-1]['output_layer']=output
            if a.output:
                with open(a.output,'w') as f: json.dump(results,f,indent=2)
