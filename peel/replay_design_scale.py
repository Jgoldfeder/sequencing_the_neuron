"""Replay captured failed rows after making candidate radii scale invariant."""
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from peel.informative_kinks import recover_layer
from align import scale_normalize_, align_clone_to
torch.set_num_threads(1)
d=torch.load('peel/design_failures/layer1_1790812936183464928.pt',map_location='cpu',weights_only=False)
t=MLP(d['dims']).double().cuda();g=t.clone();t.load_state_dict(d['teacher']);g.load_state_dict(d['student'])
cs=d['failed'];diag={};start=time.perf_counter()
w,b,m,q=recover_layer(lambda x:t(x),g,1,'cuda',only_channels=cs,max_rounds=40,diagnostics=diag)
ref=g.clone()
with torch.no_grad():
 for c in cs:
  if m[c]:ref.layers[1].weight[c]=w[c];ref.layers[1].bias[c]=b[c]
ref, t = align_clone_to(t, ref)
idx=[c for c in cs if m[c]]
e=torch.cat([(ref.layers[1].weight[idx]-t.layers[1].weight[idx]).flatten(),ref.layers[1].bias[idx]-t.layers[1].bias[idx]])
r=dict(requested=len(cs),accepted=len(idx),queries=q,seconds=time.perf_counter()-start,max_error=float(e.abs().max()) if len(e) else None,failed=[c for c in cs if not m[c]],diagnostics=diag)
torch.save(dict(student=ref.state_dict(),mask=m.cpu()),'peel/design_scale_recovered.pt')
Path('peel/design_scale_replay.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='diagnostics'},flush=True)
