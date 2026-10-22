"""Focused checks of production LeNet repairs. Uses teacher rows only as a test oracle."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
from align import cnn_canonicalize_,cnn_align_to_
from cnn_balance import scale_channels_
from residual_seed import prefix,residual_subspace
from method import (_cnn_residual_record_view,_cnn_input_scale_to_member,
                    _cnn_frontier_input,_cnn_tail_forward,_cnn_boundary_weight_in_ref,_spatial_boundary_normals)
torch.set_num_threads(4);torch.manual_seed(43);dev='cpu'
r=torch.load('recon/_record_lenet_c1c2_c5_105.pt',weights_only=False,map_location=dev)
def net():return ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).double()
rec=net();rec.load_state_dict(r['partial_state']);cnn_canonicalize_(rec)
t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',weights_only=True));cnn_canonicalize_(t);cnn_align_to_(t,rec)
for m in (rec,t):
 for p in m.parameters():p.requires_grad_(False)
# Reuse the previously identified strongest missing channel to expose the numerical floor.
previous=json.loads(Path('audit_lenet/probe_results.json').read_text());added=previous['controlled_promoted_channel'];mask=r['partial_mask'][2].clone();mask[added]=True
promoted=rec.clone()
with torch.no_grad():promoted.layers[2].weight[added]=t.layers[2].weight[added];promoted.layers[2].bias[added]=t.layers[2].bias[added]
frozen={l:(promoted.layers[l].weight.detach().clone(),promoted.layers[l].bias.detach().clone(),mask if l==2 else r['partial_mask'][l]) for l in (0,1,2)}
member=promoted.clone().float();fixed=_cnn_residual_record_view(member,frozen,None,{},2)
results={}
for name,m in [('promoted_fp32',member.clone().double()),('production_record_view',fixed)]:
 V,S,nq=residual_subspace(t,m,2,mask,n_points=8,gen=torch.Generator().manual_seed(43),verbose=False)
 results[name]={'used_rank':len(V),'threshold_rank':int((S>1e-4*S[0]).sum()),'queries':nq,'first_noise_singular_value':float(S[16])}
assert results['promoted_fp32']['threshold_rank']>16
assert results['promoted_fp32']['used_rank']==0
assert results['production_record_view']['used_rank']==16
# Member-specific redraw view must retain precise prefix coefficients in that member's scales.
member=promoted.clone()
for l in (0,1,2):scale_channels_(member,l,torch.exp(torch.randn(member.layers[l].weight.shape[0],dtype=torch.double)*.7))
gview=_cnn_residual_record_view(member,frozen,None,{},2,member_input_gauge=True)
X=torch.randn(32,784,dtype=torch.double)*.5
with torch.no_grad():results['member_gauge_exact_prefix_error']=float((prefix(member,X,2)[0]-prefix(gview,X,2)[0]).abs().max())
assert results['member_gauge_exact_prefix_error']<1e-10
# The same input-coordinate conversion must work at conv2, C5, and F6.
errs={};rowerrs={}
with torch.no_grad():
 for fr in (1,2,3):
  h=_cnn_frontier_input(promoted,X,fr);scale=_cnn_input_scale_to_member(member,promoted,fr,h.shape[1]);y=_cnn_tail_forward(member,h*scale,fr)
  errs[str(fr)]=float((y-member(X)).abs().max())
  w=_cnn_boundary_weight_in_ref(member,promoted,fr)
  if fr==2:
   z=h@w.flatten(1).T+member.layers[fr].bias;mh=_cnn_frontier_input(member,X,fr);real=mh@member.layers[fr].weight.flatten(1).T+member.layers[fr].bias;rowerrs[str(fr)]=float((z-real).abs().max())
with torch.no_grad():
 h=_cnn_frontier_input(promoted,X,1)
 normals,biases=_spatial_boundary_normals(promoted,[promoted,member],1,promoted.input_shape,h.shape[1],16,32,2,'cpu',torch.Generator().manual_seed(3))
 predicted=(h*normals).sum(1)+biases
 actual=[]
 for i,m in enumerate([promoted,member]):
  hm=_cnn_frontier_input(m,X[i*16:(i+1)*16],1).reshape(16,6,14,14)
  z=m.layers[1](hm);actual.append(z[torch.arange(16),torch.arange(16),0,0])
 rowerrs['spatial_conv2']=float((predicted-torch.cat(actual)).abs().max())
results['tail_coordinate_errors']=errs;results['boundary_covector_errors']=rowerrs
assert max(errs.values())<1e-10 and max(rowerrs.values())<1e-10
Path('audit_lenet/c5_fix_validation.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2));print('PASS')
