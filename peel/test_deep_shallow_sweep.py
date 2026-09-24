"""Bounded depth/width test of the unmodified shallow sweep algorithm.

A two-layer carrier supplies only the target guess to the solver. Its output
layer is zero and unused. This bypasses the architecture restriction only in
this diagnostic, leaving production dispatch unchanged. Direct tail access is
an explicitly privileged control; simulated-tail queries call the full teacher.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import scale_normalize_
from peel.shallow_sweep import recover_layer
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--width',type=int,default=3072);p.add_argument('--channels',type=int,default=2)
p.add_argument('--attempts',type=int,default=2);p.add_argument('--device',default='cuda')
p.add_argument('--noise',type=float,default=.01);p.add_argument('--seed',type=int,default=0)
p.add_argument('--cases',nargs='+',default=['first_full','second_direct','second_simulated'])
p.add_argument('--output',default='peel/deep_shallow_sweep.json');a=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(a.seed);dev=a.device;d=a.width
T=MLP([d]*4).double()
with torch.no_grad():
 for L in T.layers:L.bias.normal_(0,.1)
scale_normalize_(T);T=T.to(dev).eval()
ng=torch.Generator(device=dev).manual_seed(1)
guesses=[]
for layer in range(2):
 G=MLP([d,d,1]).double().to(dev)
 with torch.no_grad():
  G.layers[0].weight.copy_(T.layers[layer].weight+a.noise/d**.5*torch.randn(d,d,device=dev,dtype=torch.float64,generator=ng))
  G.layers[0].bias.copy_(T.layers[layer].bias+a.noise/2*torch.randn(d,device=dev,dtype=torch.float64,generator=ng))
  G.layers[1].weight.zero_();G.layers[1].bias.zero_()
 guesses.append(G)
with torch.no_grad():
 lu,piv=torch.linalg.lu_factor(T.layers[0].weight)
def inverse(h):
 return torch.linalg.lu_solve(lu,piv,(torch.where(h>=0,h,h/.01)-T.layers[0].bias).T).T
report=dict(config=vars(a),dims=[d]*4,results=[])
for case in a.cases:
 layer=0 if case=='first_full' else 1
 if case=='first_full':oracle=lambda x:T(x)
 elif case=='second_direct':oracle=lambda h:T.layers[2](T.act(T.layers[1](h)))
 elif case=='second_simulated':oracle=lambda h:T(inverse(h))
 else:raise ValueError(case)
 print('START',case,flush=True);torch.cuda.reset_peak_memory_stats() if dev.startswith('cuda') else None
 diag={};start=time.perf_counter()
 w,b,mask,nq=recover_layer(oracle,guesses[layer],dev,only_channels=list(range(a.channels)),gen=torch.Generator(device=dev).manual_seed(1234),diagnostics=diag,attempts=a.attempts)
 with torch.no_grad():
  norm=w.norm(dim=1);ew=(w/norm[:,None]-T.layers[layer].weight).abs();eb=(b/norm-T.layers[layer].bias).abs()
  errors=torch.maximum(ew.max(1).values,eb)[:a.channels]
  # Assess forward round trip on independent, ordinary hidden inputs.
  h=torch.randn(128,d,device=dev,dtype=torch.float64,generator=torch.Generator(device=dev).manual_seed(20))
  hx=T.act(T.layers[0](inverse(h)))
 result=dict(case=case,layer=layer,seconds=time.perf_counter()-start,queries=nq,accepted=int(mask[:a.channels].sum()),channels=a.channels,maximum_returned_error=float(errors.max()),accepted_max=float(errors[mask[:a.channels]].max()) if mask[:a.channels].any() else None,mean_returned_error=float(torch.cat([ew[:a.channels].flatten(),eb[:a.channels]]).mean()),diagnostics=diag,prefix_roundtrip_relative=float((hx-h).norm()/h.norm()),peak_gpu_bytes=torch.cuda.max_memory_allocated() if dev.startswith('cuda') else None)
 report['results'].append(result);Path(a.output).write_text(json.dumps(report,indent=2))
 print('RESULT',json.dumps({k:v for k,v in result.items() if k!='diagnostics'}),flush=True)
