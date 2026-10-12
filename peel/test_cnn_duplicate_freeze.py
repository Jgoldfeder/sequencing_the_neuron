"""Regression: a duplicate cannot bypass partial rejection through full-layer peel."""
import copy,io,sys,tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import method,kink_solve
from nets import ConvNet

torch.set_num_threads(1);torch.manual_seed(17)
shape=(1,6,6);conv=[(1,2,3,1,1,2),(2,3,2,1,0,0)];fc=(4,)
teacher=ConvNet(shape,conv,fc,2,'leaky_relu')
for partial in [True,False]:
 calls=[]
 def factory(*args,**kwargs):
  m=copy.deepcopy(teacher);m.requires_grad_(True);return m
 def fake_recovery(t,guess,frontier,device,**kw):
  calls.append(frontier)
  w=guess.layers[frontier].weight.detach().clone();b=guess.layers[frontier].bias.detach().clone()
  w[1]=w[0];b[1]=b[0]
  return w,b,torch.ones(len(b),dtype=torch.bool),0
 cfg=method.Cfg(p=3,q=32,outer=1,epochs=0,qg_steps=0,log_every=1,
  act='leaky_relu',fast_peel_partial=partial,fast_peel=True,freeze_reinit=True,
  loc_refine=True,cluster_eps=.005,cluster_quorum=.67,freeze_thresh=1.,freeze_precision=0.)
 stream=io.StringIO()
 with patch.object(method,'ConvNet',side_effect=factory),patch.object(kink_solve,'recover_layer',side_effect=fake_recovery),redirect_stdout(stream):
  method.reconstruct_cnn(teacher,shape,conv,2,cfg,'cpu',fc_dims=fc)
 log=stream.getvalue()
 assert calls==[0],(partial,calls,log)
 if partial:
  assert 'COPY' in log and '[kink-refine]' not in log,log
 else:
  assert 'full-layer refinement rejected duplicate' in log,log
  assert '[kink-refine] solved L1 1/2' in log,log
 print('PASS', 'partial rejection invalidates full-layer trigger' if partial else 'full-layer path independently rejects duplicate')
# A separately recovered candidate must also be checked against existing exact rows.
w=torch.tensor([[1.,0.],[1.,0.]]);b=torch.zeros(2)
assert method._dedupe_new_rows([1],w,b,w[:1],b[:1])==([],[1])
print('PASS existing-row duplicate check')
