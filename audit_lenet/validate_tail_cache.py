"""Check cached-tail training against the actual ConvNet and original image labels."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from nets import ConvNet
from align import cnn_canonicalize_,cnn_align_to_
from residual_seed import prefix
torch.set_num_threads(4)
conv=[(1,6,5,1,2,2),(6,16,5,1,0,2),(16,120,5,1,0,0)]
def net():return ConvNet((1,28,28),conv,(84,),10,'leaky_relu').double()
r=torch.load('recon/_record_lenet_c1c2_c5_105.pt',weights_only=False,map_location='cpu');rec=net();rec.load_state_dict(r['partial_state']);cnn_canonicalize_(rec)
t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',weights_only=True));cnn_canonicalize_(t);cnn_align_to_(t,rec)
d=torch.load('audit_lenet/boundary_test_20261021_e/dataset.pt',weights_only=True,map_location='cpu');x=d['target_images'][:32].double();h,_=prefix(rec,x,2);savedh=d['boundary_H'][len(d['gaussian_images'])//2:][:32]
ck=torch.load('audit_lenet/boundary_test_20261021_a/boundary_free_latest.pt',weights_only=True,map_location='cpu');sd=ck['model']
m=rec.clone().float()
with torch.no_grad():
 for i in range(3):
  m.layers[i+2].weight.copy_(sd[f'ws.{i}'][0].reshape_as(m.layers[i+2].weight));m.layers[i+2].bias.copy_(sd[f'bs.{i}'][0])
 v=h.float()
 for i in range(3):
  v=F.linear(v,sd[f'ws.{i}'][0],sd[f'bs.{i}'][0]);v=F.leaky_relu(v,.01) if i<2 else v
 diff=float((m(x.float())-v).abs().max());labeldiff=float((t(x).float()-d['boundary_Y'][len(d['gaussian_images'])//2:][:32]).abs().max())
 # At zero replacement delta the oracle isolation model must equal labels too.
 H=d['boundary_H'][:128];Z=F.leaky_relu(H@t.layers[2].weight.flatten(1).float().T+t.layers[2].bias.float(),.01)
 pred=F.linear(F.leaky_relu(F.linear(Z,t.layers[3].weight.float(),t.layers[3].bias.float()),.01),t.layers[4].weight.float(),t.layers[4].bias.float())
 floor=float((pred-d['boundary_Y'][:128]).abs().max())
report={'cached_features_max_error':float((h.float()-savedh).abs().max()),'full_convnet_vs_cached_tail_max_error':diff,'saved_output_labels_max_error':labeldiff,'oracle_control_true_parameters_max_error':floor}
assert diff<2e-5 and labeldiff<2e-5 and floor<2e-5
Path('audit_lenet/tail_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
