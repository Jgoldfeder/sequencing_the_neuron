"""Smoke the real CLI, checkpoint metadata and complete/incomplete output paths."""
import sys,tempfile,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
torch.set_num_threads(1);torch.manual_seed(19)
t=MLP([12,6,5,3]).double();states=[]
for i in range(2):
 m=t.clone()
 with torch.no_grad():
  for p in m.parameters():p.add_(torch.randn_like(p)*1e-4)
 states.append(m.state_dict())
with tempfile.TemporaryDirectory() as td:
 p=Path(td);torch.save(dict(dims=t.dims,act='leaky_relu',teacher_state=t.state_dict(),pop_states=states),p/'snapshot.pt')
 cmd=[sys.executable,'refine_snapshot_mlp.py',str(p/'snapshot.pt'),'--member','0','--device','cpu','--local-refine','--head-points','128','--out',str(p/'result.pt')]
 import os
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=120)
 assert result.returncode==0,result.stderr+result.stdout
 r=torch.load(p/'result.pt',weights_only=False);assert r['complete'] and r['completed_layers']==2,result.stdout
 assert (p/'result.pt.L2.points.pt').exists()
 result=subprocess.run(cmd+['--stop-after','1'],env=env,capture_output=True,text=True,timeout=120)
 assert result.returncode==0,result.stderr+result.stdout
 r=torch.load(p/'result.pt',weights_only=False);assert not r['complete'] and r['completed_layers']==1
 assert 'output fit deferred' in result.stdout
 print('PASS actual local-refine CLI, complete recovery on small fixture, point checkpoints, partial completion metadata, deferred output fit')
