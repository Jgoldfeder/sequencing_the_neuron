"""Independent checkpoint validation; teacher coefficients used for grading only."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
from align import cnn_canonicalize_,cnn_align_to_
from residual_seed import prefix
torch.set_num_threads(4);dev='cuda:2';torch.manual_seed(22701)
r=torch.load('audit_lenet/record_c5_signed_120_20261021.pt',weights_only=False,map_location=dev)
def net():return ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).to(dev).double()
n=net();n.load_state_dict(r['partial_state']);t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',map_location=dev,weights_only=True));cnn_canonicalize_(n);cnn_canonicalize_(t);cnn_align_to_(t,n)
for m in [n,t]:
 for p in m.parameters():p.requires_grad_(False)
errs={str(l):max(float((n.layers[l].weight-t.layers[l].weight).abs().max()),float((n.layers[l].bias-t.layers[l].bias).abs().max())) for l in range(3)}
slots=r['unsigned_slots'];new=torch.tensor(slots,device=dev);newerr=max(float((n.layers[2].weight[new]-t.layers[2].weight[new]).abs().max()),float((n.layers[2].bias[new]-t.layers[2].bias[new]).abs().max()))
# Fresh random starting points, not the saved discovery anchors.
ids=new.repeat_interleave(32);x=(torch.randn(len(ids),784,device=dev,dtype=torch.double)*1.5).requires_grad_(True);W=n.layers[2].weight.flatten(1)[ids];b=n.layers[2].bias[ids]
for it in range(35):
 val=(prefix(n,x,2)[0]*W).sum(1)+b;grad=torch.autograd.grad(val.sum(),x)[0]
 with torch.no_grad():x.sub_(val[:,None]*grad/grad.square().sum(1,keepdim=True).clamp_min(1e-30))
v=grad.detach();v/=v.norm(dim=1,keepdim=True);xx=x.detach();ss=torch.tensor([-2,-1,0,1,2],device=dev,dtype=torch.double)*1e-5
with torch.no_grad():
 Z=(xx[:,None]+ss[None,:,None]*v[:,None]).reshape(-1,784);Y=torch.cat([t(zz) for zz in Z.split(256)]).reshape(len(ids),5,10);dd=(Y[:,2:]-2*Y[:,1:-1]+Y[:,:-2]).norm(dim=2);hits=(dd[:,1]>100*torch.maximum(dd[:,0],dd[:,2]).clamp_min(1e-13)).reshape(-1,32).sum(1)
 X=torch.randn(1024,784,device=dev,dtype=torch.double)*1.5
 features=lambda m: m.act(prefix(m,X,2)[0]@m.layers[2].weight.flatten(1).T+m.layers[2].bias)
 ferr=float((features(n)-features(t)).abs().max())
report={'signed_C5_rows':int(r['partial_mask'][2].sum()),'prefix_canonical_max_errors':errs,'new_17_rows_max_coefficient_error':newerr,'new_slots':slots,'independent_kink_hits_out_of_32':hits.tolist(),'max_C5_feature_error_on_1024_fresh_images':ferr,'validation_output_queries':len(Z),'new_signs_from_teacher_weights':False,'assumption':'fp64 black-box outputs, same as existing kink solver'}
assert report['signed_C5_rows']==120 and errs['2']<3.2e-10 and newerr<1e-10
assert int(hits.min())>=30
Path('audit_lenet/c5_complete_independent_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
