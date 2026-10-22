"""Recover the final linear head from labels after signed C1/C2/C5/F6 recovery."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
from residual_seed import prefix
from align import cnn_canonicalize_,cnn_align_to_
torch.set_num_threads(4);dev='cuda:2';r=torch.load(sys.argv[1],map_location=dev,weights_only=False)
def net():return ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).to(dev).double()
n=net();n.load_state_dict(r['partial_state']);t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',map_location=dev,weights_only=True))
for m in [n,t]:
 for p in m.parameters():p.requires_grad_(False)
g=torch.Generator(device=dev).manual_seed(22801)
@torch.no_grad()
def features(X):return torch.cat([prefix(n,xx,4)[0] for xx in X.split(256)])
@torch.no_grad()
def labels(X):return torch.cat([t(xx) for xx in X.split(256)])
X=torch.cat([torch.randn(2048,784,device=dev,dtype=torch.double,generator=g)*1.5,(torch.rand(2048,784,device=dev,dtype=torch.double,generator=g)*2-1)*3])
H=features(X);Y=labels(X);A=torch.cat([H,torch.ones(len(H),1,device=dev,dtype=torch.double)],1)
sv=torch.linalg.svdvals(A);assert float(sv[-1])>1e-9
coef=torch.linalg.lstsq(A.cpu(),Y.cpu(),driver='gelsd').solution.to(dev)
with torch.no_grad():n.layers[4].weight.copy_(coef[:-1].T);n.layers[4].bias.copy_(coef[-1])
# Independent samples with several scales; no validation response is used for fitting.
checks={}
for scale in [.5,1.5,3.0]:
 xx=torch.randn(2048,784,device=dev,dtype=torch.double,generator=g)*scale
 with torch.no_grad():pred=torch.cat([n(x) for x in xx.split(256)]);truth=labels(xx);err=(pred-truth).abs()
 checks[str(scale)]={'max_output_error':float(err.max()),'mean_output_error':float(err.mean())}
# Only now grade parameters, after the head solve is fixed.
ns=n.clone();ts=t.clone();cnn_canonicalize_(ns);cnn_canonicalize_(ts);cnn_align_to_(ts,ns)
errs={str(l):max(float((ns.layers[l].weight-ts.layers[l].weight).abs().max()),float((ns.layers[l].bias-ts.layers[l].bias).abs().max())) for l in range(5)}
report={'provisional_f6':r.get('provisional_f6',False),'head_fit_label_queries':len(X),'independent_validation_label_queries':6144,'design_rank':int((sv>sv[0]*1e-12).sum()),'design_condition':float(sv[0]/sv[-1]),'canonical_max_coefficient_error_by_layer':errs,'independent_outputs':checks,'teacher_parameters_used_for_fit':False,'oracle_contract':'fp64 forward evaluations'}
r['source_peel_refinement_queries']=r.pop('peel_refinement_queries',None);r['query_accounting_note']='See individual recovery logs; legacy source-record query counts are separate.'
r['partial_state']={k:v.detach().cpu() for k,v in n.state_dict().items()};r['state_dict']=r['partial_state'];r['refined_state']=r['partial_state'];r['complete_lenet_recovery']=not r.get('provisional_f6',False);r['head_recovery_report']=report;r['note']='All layers recovered from the original 103-C5 bank with output-only direct kink recovery and a linear head fit.';r['partial_mask']={l:torch.ones(n.layers[l].weight.shape[0],dtype=torch.bool) for l in range(5)};r['frozen']={l:(n.layers[l].weight.detach().cpu(),n.layers[l].bias.detach().cpu(),r['partial_mask'][l]) for l in range(5)}
torch.save(r,sys.argv[2]);Path(sys.argv[2]+'.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
