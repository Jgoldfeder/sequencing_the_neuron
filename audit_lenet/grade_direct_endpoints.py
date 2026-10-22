"""Teacher-only coverage diagnostic; never imported by recovery/search."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
from align import cnn_canonicalize_,cnn_align_to_
from residual_seed import prefix
torch.set_num_threads(4);dev='cuda:2';r=torch.load('recon/_record_lenet_c1c2_c5_105.pt',weights_only=False,map_location=dev)
def net():return ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).to(dev).double()
n=net();n.load_state_dict(r['partial_state']);cnn_canonicalize_(n);t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',weights_only=True,map_location=dev));cnn_canonicalize_(t);cnn_align_to_(t,n)
z=torch.load('audit_lenet/direct_c5_20261021_b_setup.pt',weights_only=False,map_location=dev)
with torch.no_grad():
 H=prefix(t,z['ends'],2)[0];Z=H@t.layers[2].weight.flatten(1).T+t.layers[2].bias;free=(~r['partial_mask'][2]).nonzero().flatten();data={'score_only':True,'missing_indices_in_record_order':free.tolist(),'endpoint_positive_counts':(Z[:,free]>0).sum(0).tolist(),'endpoint_min_z':Z[:,free].min(0).values.tolist(),'endpoint_max_z':Z[:,free].max(0).values.tolist()}
Path('audit_lenet/direct_endpoint_coverage_score_only.json').write_text(json.dumps(data,indent=2));print(json.dumps(data),flush=True)
