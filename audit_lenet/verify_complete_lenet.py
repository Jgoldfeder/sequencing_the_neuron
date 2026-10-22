"""Independent reload, parameter grading, and fresh-output tests of final recovery."""
import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
from align import cnn_canonicalize_,cnn_align_to_
torch.set_num_threads(4);dev='cuda:2';path=Path('audit_lenet/record_lenet_complete_20261021.pt');r=torch.load(path,map_location=dev,weights_only=False)
assert r['complete_lenet_recovery'] and not r.get('provisional_f6',False)
def net():return ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).to(dev).double()
n=net();n.load_state_dict(r['state_dict']);t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',map_location=dev,weights_only=True))
for m in [n,t]:
 for p in m.parameters():p.requires_grad_(False)
g=torch.Generator(device=dev).manual_seed(23001);results={}
with torch.no_grad():
 for scale in [.25,.5,1.,3.,6.]:
  X=torch.randn(1024,784,device=dev,dtype=torch.double,generator=g)*scale;Y=torch.cat([t(x) for x in X.split(256)]);P=torch.cat([n(x) for x in X.split(256)]);d=(P-Y).abs();results[str(scale)]={'max_output_error':float(d.max()),'mean_output_error':float(d.mean())}
ns=n.clone();ts=t.clone();cnn_canonicalize_(ns);cnn_canonicalize_(ts);cnn_align_to_(ts,ns)
errors={str(l):max(float((ns.layers[l].weight-ts.layers[l].weight).abs().max()),float((ns.layers[l].bias-ts.layers[l].bias).abs().max())) for l in range(5)}
counts={str(l):int(r['partial_mask'][l].sum()) for l in range(5)}
assert list(counts.values())==[6,16,120,84,10]
assert errors['2']<3e-11 and errors['3']<1e-8 and errors['4']<1e-4
assert max(z['max_output_error'] for z in results.values())<1e-6
report={'checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'recovered_rows_by_layer':counts,'canonical_max_coefficient_error_by_layer':errors,'independent_output_tests':results,'independent_output_queries':5120,'oracle_contract':'fp64 teacher outputs','provisional':False}
Path('audit_lenet/complete_lenet_independent_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
