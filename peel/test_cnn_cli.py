"""Check the user's CLI flags without running training or saving checkpoints."""
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run
from nets import ConvNet

class Checked(Exception): pass

def teacher(shape,conv,fc,out,**kwargs):
    return ConvNet(shape,conv,fc,out,kwargs['act'])

def inspect(teacher,shape,conv,out,cfg,device,**kwargs):
    assert cfg.fast_peel_partial and cfg.freeze_reinit and cfg.peel_refresh and cfg.loc_refine
    assert cfg.freeze_thresh==1. and cfg.freeze_precision==0.
    assert cfg.retry==5 and cfg.peel_angle_gate==12.
    raise Checked

with patch.object(sys,'argv',['run.py','--variant','mergedbest','--conv','lenet',
        '--cnn-act','leaky_relu','--device','cpu','--outer','30','--window','30',
        '--q','2000','--fast-peel-partial','--peelrefresh','--loc-refine','--retry']), \
     patch.object(run,'make_teacher_cnn',side_effect=teacher), \
     patch.object(run,'reconstruct_cnn',side_effect=inspect):
    try:
        run.main()
    except Checked:
        print('PASS: user command enables full-layer refresh with partial peeling')
    else:
        raise AssertionError('Reconstruction was not intercepted')
