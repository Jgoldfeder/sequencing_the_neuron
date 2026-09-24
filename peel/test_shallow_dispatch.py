"""Exercise actual design dispatch on the unaligned saved consensus and teacher."""
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import param_errors
from method import _mlp_refine_layer

torch.set_num_threads(1);dev='cuda'
s=torch.load('recon/mergedbest512__3072x256x100__s0_consensus.pt',map_location='cpu',weights_only=False)
t=MLP(s['dims']).double().to(dev);t.load_state_dict(torch.load('teachers/teacher_3072x256x100_e25_s0.pt',map_location='cpu',weights_only=True))
g=MLP(s['dims']).double().to(dev);g.load_state_dict(s['consensus_state'])
start=time.perf_counter();w,b,mask,nq=_mlp_refine_layer(t,g,0,dev,'leaky_relu',design=True)
with torch.no_grad():
 idx=mask.nonzero()[:,0];gw=g.layers[0].weight[idx];gb=g.layers[0].bias[idx]
 scale=(gw*w[idx]).sum(1)+gb*b[idx]
 g.layers[0].weight[idx]=w[idx]*scale[:,None];g.layers[0].bias[idx]=b[idx]*scale
refine_seconds=time.perf_counter()-start
assert mask.all(),mask
errors=param_errors(g,t)
assert max(errors['max_eps_per_matrix'][:2])<1e-10,errors
with torch.no_grad():
 gen=torch.Generator(device=dev).manual_seed(8);x=torch.randn(2048,3072,device=dev,dtype=torch.float64,generator=gen)
 H=g.act(g.layers[0](x));A=torch.cat([H,torch.ones_like(H[:,:1])],1)
 solution=torch.linalg.lstsq(A.cpu(),t(x).cpu()).solution.to(dev)
 g.layers[1].weight.copy_(solution[:-1].T);g.layers[1].bias.copy_(solution[-1])
 val=torch.randn(512,3072,device=dev,dtype=torch.float64,generator=gen)
 err=float((g(val)-t(val)).abs().max())
assert err<1e-9,err
result=dict(accepted=int(mask.sum()),queries=nq,refine_seconds=refine_seconds,hidden_max=max(errors['max_eps_per_matrix'][:2]),output_max_error=err,total_seconds=time.perf_counter()-start)
Path('peel/shallow_dispatch.json').write_text(json.dumps(result,indent=2));print('PASS',json.dumps(result),flush=True)
