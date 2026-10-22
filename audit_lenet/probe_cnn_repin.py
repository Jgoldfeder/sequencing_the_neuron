"""Check that rebuilding a CNN population restores recorded partial rows before freezing."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
from align import cnn_canonicalize_
from method import _cnn_reinit_frozen_population,_repin_partial

torch.set_num_threads(4);torch.manual_seed(931)
r=torch.load('recon/_record_lenet_c1c2_c5_105.pt',weights_only=False,map_location='cpu');rec=ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).double();rec.load_state_dict(r['partial_state']);cnn_canonicalize_(rec)
frozen={l:(rec.layers[l].weight.detach().clone(),rec.layers[l].bias.detach().clone(),r['partial_mask'][l]) for l in (0,1)}
pop=_cnn_reinit_frozen_population(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act'],'cpu',2,frozen);opts=[torch.optim.Adam(m.parameters(),lr=.001) for m in pop];mask={2:r['partial_mask'][2]};hooked=set();live={}
_repin_partial(pop,opts,mask,rec,frozen,hooked,live,'cpu')
errors=[]
with torch.no_grad():
 for m in pop:
  check=m.clone().double();cnn_canonicalize_(check);mm=mask[2];errors.append(float((check.layers[2].weight[mm]-rec.layers[2].weight[mm]).abs().max()))
report={'canonical_pinned_errors':errors,'live_mask_counts':[int(live[(id(m),2)].sum()) for m in pop]}
Path('audit_lenet/cnn_repin_probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

assert max(errors) < 1e-6
assert all(int(live[(id(m),2)].sum()) == 103 for m in pop)
print("PASS: CNN recorded rows restored before freezing")
