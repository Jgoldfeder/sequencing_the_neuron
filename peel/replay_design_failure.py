"""Replay a production design-refinement snapshot without retraining guesses."""
import sys,argparse,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from peel.informative_kinks import recover_layer
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('snapshot');p.add_argument('--device',default='cuda');p.add_argument('--failed-only',action='store_true');p.add_argument('--strategy',choices=['adaptive','legacy']);p.add_argument('--rounds',type=int)
a=p.parse_args();saved=torch.load(a.snapshot,map_location='cpu',weights_only=False)
if saved['teacher'] is None: p.error('Snapshot has no serializable teacher; supply its forward callable in a custom replay.')
t=MLP(saved['dims'],act=saved['act']).double().to(a.device);g=t.clone()
t.load_state_dict(saved['teacher']);g.load_state_dict(saved['student'])
opts=saved['options'].copy()
if a.strategy:opts['retry_strategy']=a.strategy
if a.rounds is not None:opts['max_rounds']=a.rounds
channels=saved['failed'] if a.failed_only else saved['channels']
gen=torch.Generator(device=a.device);gen.set_state(saved['rng'])
diag={};start=time.perf_counter()
w,b,mask,nq=recover_layer(t,g,saved['frontier'],a.device,only_channels=channels,gen=gen,diagnostics=diag,**opts)
result=dict(accepted=int(mask[channels].sum()),channels=len(channels),queries=nq,seconds=time.perf_counter()-start,diagnostics=diag)
path=Path(a.snapshot).with_suffix('.replay.json');path.write_text(json.dumps(result,indent=2))
print('Saved',path)
