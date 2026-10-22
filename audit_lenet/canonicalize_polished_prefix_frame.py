"""Transfer the F6 bank/candidate into the newly unit-normalized C5 input gauge."""
import sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nets import ConvNet
from align import cnn_canonicalize_
torch.set_num_threads(4)
old=torch.load('audit_lenet/unsigned_f6_round4.pt',weights_only=False,map_location='cpu');r=torch.load('audit_lenet/unsigned_f6_precise_prefix_fixed.pt',weights_only=False,map_location='cpu')
W=old['partial_state']['layers.2.weight'].flatten(1);b=old['partial_state']['layers.2.bias'];scale=(W.square().sum(1)+b.square()).sqrt()
n=ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).double();n.load_state_dict(r['partial_state'])
with torch.no_grad():n.layers[3].weight.mul_(scale[None])
cnn_canonicalize_(n);r['partial_state']={k:v.clone() for k,v in n.state_dict().items()};r['state_dict']=r['partial_state'];r['frozen']={l:(n.layers[l].weight.clone(),n.layers[l].bias.clone(),torch.ones(n.layers[l].weight.shape[0],dtype=torch.bool)) for l in [0,1,2]};r['prefix_polish_gauge_fixed']=True;torch.save(r,'audit_lenet/unsigned_f6_precise_canonical.pt')
z=torch.load('audit_lenet/direct_f6_last_affine.pt',weights_only=False,map_location='cpu')
for f in z['hyperplanes']:
 row=f['row'].clone();row[:-1]*=scale;row/=row.norm();f['row']=row;f['previous_frame_score_only']=f.pop('score_only',None)
z['input_gauge_conversion']=scale;torch.save(z,'audit_lenet/direct_f6_last_affine_canonical.pt')
print('F6 input columns and candidate transformed into the polished C5 gauge')
