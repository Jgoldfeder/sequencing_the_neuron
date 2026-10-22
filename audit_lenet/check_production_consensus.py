"""Apply the actual CNN quorum routine to the successful diagnostic population."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
from align import cnn_canonicalize_
from method import _cnn_consensus
torch.set_num_threads(4)
r=torch.load('recon/_record_lenet_c1c2_c5_105.pt',weights_only=False,map_location='cpu');rec=ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).double();rec.load_state_dict(r['partial_state']);cnn_canonicalize_(rec)
ck=torch.load('audit_lenet/boundary_test_20261021_e/boundary_fixedtail_latest.pt',weights_only=True,map_location='cpu');pop=[]
with torch.no_grad():
 for j in range(8):
  m=rec.clone().float()
  for i in range(3):
   m.layers[i+2].weight.copy_(ck['model'][f'ws.{i}'][j].reshape_as(m.layers[i+2].weight));m.layers[i+2].bias.copy_(ck['model'][f'bs.{i}'][j])
  pop.append(m)
 ref=min(range(8),key=lambda j:ck['metrics']['loss_by_member'][j]);cons,counts,masks=_cnn_consensus(pop,ref,.02,5)
 free=~r['partial_mask'][2];result={'epoch':ck['epoch'],'reference_member':ref,'eps':.02,'quorum':5,'C5_consensus_total':int(masks[2].sum()),'missing_C5_in_consensus':int(masks[2][free].sum()),'hidden_layer_counts':counts}
Path('audit_lenet/production_consensus_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
