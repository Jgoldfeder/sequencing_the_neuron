"""Parse production flag and inspect MLP configuration without loading datasets."""
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,run
from nets import MLP
class Checked(Exception):pass
def inspect(teacher,dims,cfg,device,eval_pts,**kw):
 assert cfg.normalized_peel and not cfg.peel_seal
 assert cfg.fast_peel_partial and cfg.freeze_reinit and cfg.peel_restart and cfg.design_refine
 raise Checked
with patch.object(sys,'argv',['run.py','--variant','mergedbest512','--arch','8,6,4,2','--device','cpu','--peel',
 '--fast-peel-partial','--peelrestart','--design-refine','--normalized-peel']), \
 patch.object(run,'make_teacher',side_effect=lambda dims,**kw:MLP(dims)), \
 patch.object(run,'load_data',return_value=((torch.randn(8,8),torch.zeros(8,dtype=torch.long)),(torch.randn(8,8),torch.zeros(8,dtype=torch.long)))), \
 patch.object(run,'reconstruct',side_effect=inspect):
 try:run.main()
 except Checked:print('PASS normalized-peel CLI wiring')
 else:raise AssertionError('Did not reach reconstruction')
