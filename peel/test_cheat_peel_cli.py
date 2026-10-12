"""CLI shortcut uses actual full-teacher suffix weights, not a new teacher."""
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,run
from nets import MLP
class Checked(Exception):pass
for drop in [1,2]:
 saved=[]
 def make(dims,**kw):
  assert dims==[12,8,6,3]
  net=MLP(dims);saved.append(net);return net
 def inspect(teacher,dims,cfg,device,eval_pts,**kw):
  assert dims==[12,8,6,3][drop:]
  for got,want in zip(teacher.layers,saved[0].layers[drop:]):
   assert torch.equal(got.weight,want.weight) and torch.equal(got.bias,want.bias)
  assert eval_pts.shape==(2000,dims[0])
  raise Checked
 argv=['run.py','--variant','mergedbest512','--arch','12,8,6,3','--device','cpu','--no-save-recon','--cheat-peel']
 if drop==2:argv.append('2')
 with patch.object(sys,'argv',argv),patch.object(run,'make_teacher',side_effect=make),patch.object(run,'reconstruct',side_effect=inspect):
  try:run.main()
  except Checked:pass
  else:raise AssertionError('Did not reach reconstruction')
print('PASS bare and explicit cheat-peel CLI use exact teacher suffix')
