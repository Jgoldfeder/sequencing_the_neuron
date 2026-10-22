"""Round-trip the real LeNet production checkpoint's exact-row banks."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
from method import _cnn_checkpoint_record,_cnn_consensus_eps
from align import cnn_canonicalize_
torch.set_num_threads(4)
p=Path(Path('audit_lenet/resume_smoke_checkpoint.txt').read_text().strip())
r=torch.load(p,weights_only=True,map_location='cpu')
def net():return ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).double()
m=net();m.load_state_dict(r['state_dict']);state,masks=_cnn_checkpoint_record(m,r);m.load_state_dict(state)
t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',weights_only=True));cnn_canonicalize_(t)
sm=[masks.get(l,torch.zeros(layer.weight.shape[0],dtype=torch.bool)) for l,layer in enumerate(m.layers)];errors=_cnn_consensus_eps(m,sm,t)
report={'snapshot':str(p),'old_loader_partial_counts':{l:int(mm.sum()) for l,mm in r.get('partial_mask',{}).items()},'restored_counts':{l:int(mm.sum()) for l,mm in masks.items()},'canonical_errors':[e['max'] for e in errors[:3]]}
assert report['old_loader_partial_counts'] != report['restored_counts'];assert report['restored_counts']=={0:6,1:16,2:103};assert max(report['canonical_errors'])<3.2e-10
# A snapshot can contain only frozen rows, or both frozen-prefix and partial-frontier banks.
only_frozen=dict(r,partial_state=None,partial_mask={});assert {l:int(mm.sum()) for l,mm in _cnn_checkpoint_record(net(),only_frozen)[1].items()}=={0:6,1:16,2:103}
original=torch.load('recon/_record_lenet_c1c2_c5_105.pt',weights_only=False,map_location='cpu');base=net();base.load_state_dict(original['state_dict']);assert {l:int(mm.sum()) for l,mm in _cnn_checkpoint_record(base,original)[1].items()}=={0:6,1:16,2:103}
assert _cnn_checkpoint_record(net(),{}) is None
report['frozen_only_and_combined_cases']='PASS'
Path('audit_lenet/cnn_checkpoint_resume_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));print('PASS')
