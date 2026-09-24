import sys,time,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import align_clone_to,param_errors
from peel.shallow_sweep import recover_layer
p=argparse.ArgumentParser();p.add_argument('--channels',type=int,default=4);p.add_argument('--output',default='peel/shallow_sweep_probe.json');a=p.parse_args()
torch.set_num_threads(1)
saved=torch.load('recon/mergedbest512__3072x256x100__s0_consensus.pt',map_location='cpu',weights_only=False)
t=MLP(saved['dims']).double();t.load_state_dict(torch.load('teachers/teacher_3072x256x100_e25_s0.pt',map_location='cpu',weights_only=True));t=t.cuda()
g=MLP(saved['dims']).double().cuda();g.load_state_dict(saved['consensus_state'])
# Align only to score corresponding true neurons; solver never sees teacher parameters.
tref,g=align_clone_to(g,t);tref=tref.double();g=g.double()
diag={};start=time.perf_counter();w,b,mask,nq=recover_layer(lambda x:tref(x),g,'cuda',only_channels=list(range(a.channels)),diagnostics=diag)
n=w.norm(dim=1);ew=(w/n[:,None]-tref.layers[0].weight).abs();eb=(b/n-tref.layers[0].bias).abs();err=torch.maximum(ew.max(1).values,eb)
r=dict(seconds=time.perf_counter()-start,queries=nq,accepted=int(mask[:a.channels].sum()),channels=a.channels,maximum=float(err[:a.channels].max()),mean=float(torch.cat([ew[:a.channels].flatten(),eb[:a.channels]]).mean()),per_neuron=err[:a.channels].tolist(),diagnostics=diag)
Path(a.output).write_text(json.dumps(r,indent=2));print('RESULT',json.dumps({k:v for k,v in r.items() if k!='diagnostics'}),flush=True)
