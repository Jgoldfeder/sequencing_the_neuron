"""Audit original sampler using alternative parameterizations of the actual stalled student.
All model coefficients come from the attacker snapshot; teacher is only a scoring oracle.
The eight-member population is constructed, not claimed to be the unsaved historical committee.
"""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
from align import cnn_canonicalize_,cnn_align_to_
from method import Cfg,gen_queries,disagreement,_cnn_frontier_input
from run import VARIANTS

torch.set_num_threads(4);torch.manual_seed(22103);dev=torch.device('cuda:2');start=time.time()
r=torch.load('recon/_record_lenet_c1c2_c5_105.pt',weights_only=False,map_location=dev)
def net():return ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).to(dev).double()
student=net();student.load_state_dict(r['state_dict']);cnn_canonicalize_(student)
# Keep the record's own channel ordering: no teacher-based matching enters construction.
free=(~r['partial_mask'][2]).nonzero().flatten();t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',weights_only=True,map_location=dev))
for m in (student,t):
 for p in m.parameters():p.requires_grad_(False)
g=torch.Generator(device=dev).manual_seed(2261);X=torch.cat([torch.randn(20000,784,device=dev,generator=g,dtype=torch.double)*.5,torch.rand(20000,784,device=dev,generator=g,dtype=torch.double)*2-1])
with torch.no_grad():
 Z=torch.cat([_cnn_frontier_input(student,xx,2)@student.layers[2].weight.flatten(1).T+student.layers[2].bias for xx in X.split(512)])
 frac=(Z>0).double().mean(0);dead=free[(Z[:,free]<=0).all(0)]
assert len(dead)>1
pop=[]
for i in range(8):
 m=student.clone();n=len(dead);strength=.25+.02*i;T=(1-strength)*torch.eye(n,device=dev,dtype=torch.double)+strength*torch.roll(torch.eye(n,device=dev,dtype=torch.double),i%(n-1)+1,0)
 with torch.no_grad():
  W=m.layers[2].weight.flatten(1);W[dead]=T@W[dead];m.layers[2].bias[dead]=T@m.layers[2].bias[dead];m.layers[3].weight[:,dead]=m.layers[3].weight[:,dead]@torch.linalg.inv(T)
 for p in m.parameters():p.requires_grad_(False)
 pop.append(m)
class Cast(torch.nn.Module):
 def __init__(self,m):super().__init__();self.m=m
 def forward(self,x):return self.m(x.to(next(self.m.parameters()).dtype))
report={'construction':'actual stalled attacker snapshot; positive mixing of its own negative unpinned rows; learned F6/head retained','probe_images':len(X),'unpinned':free.tolist(),'student_positive_fractions':frac[free].tolist(),'student_dead_unpinned':dead.tolist(),'arms':[]}
with torch.no_grad():
 report['alternative_vs_student_max_output_error']=max(float((m(xx)-student(xx)).abs().max()) for xx in X.split(512) for m in pop)
 report['student_teacher_L1_on_probe']=float(torch.cat([(student(xx)-t(xx)).abs().flatten() for xx in X.split(512)]).mean())
for precision in ['fp64','fp32']:
 models=[Cast(m.clone().to(torch.double if precision=='fp64' else torch.float)) for m in pop]
 cfg=Cfg(p=8,q=512,act='leaky_relu',**VARIANTS['mergedbest']);initial=torch.rand(cfg.q,784,device=dev,generator=torch.Generator(device=dev).manual_seed(3301))*2-1
 leaf=initial.clone().requires_grad_(True);loss=disagreement(torch.stack([m(leaf) for m in models]),cfg.disagree,cfg.qg_dist);grad=torch.autograd.grad(loss,leaf)[0]
 mined=gen_queries(models,cfg,784,dev,torch.Generator(device=dev).manual_seed(3301))
 with torch.no_grad():
  end=disagreement(torch.stack([m(mined) for m in models]),cfg.disagree,cfg.qg_dist);z=_cnn_frontier_input(student,mined.double(),2)@student.layers[2].weight.flatten(1).T+student.layers[2].bias
  report['arms'].append({'precision':precision,'initial_disagreement':-float(loss),'final_disagreement':-float(end),'max_abs_initial_sampling_gradient':float(grad.abs().max()),'max_pixel_movement':float((mined-initial).abs().max()),'previously_dead_student_units_reached':int((z[:,dead]>0).any(0).sum()),'student_teacher_L1_on_mined':float((student(mined.double())-t(mined.double())).abs().mean())})
# Student-only feature ablation: retain the same query optimizer; expose uncertain
# C5 preactivations, normalized to the students' canonical row gauges.
feature_models=[m.clone() for m in pop]
for m in feature_models:cnn_canonicalize_(m)
features=[lambda x,m=m: (_cnn_frontier_input(m,x.double(),2)@m.layers[2].weight.flatten(1).T+m.layers[2].bias)[:,free] for m in feature_models]
cfg=Cfg(p=8,q=512,act='leaky_relu',**VARIANTS['mergedbest'])
mined=gen_queries([Cast(m) for m in feature_models],cfg,784,dev,torch.Generator(device=dev).manual_seed(3301),feats=features)
with torch.no_grad():
 z=_cnn_frontier_input(student,mined.double(),2)@student.layers[2].weight.flatten(1).T+student.layers[2].bias
 report['student_only_frontier_disagreement']={'previously_dead_student_units_reached':int((z[:,dead]>0).any(0).sum()),'mean_abs_pixel':float(mined.abs().mean()),'output_disagreement_on_feature_mined':-float(disagreement(torch.stack([m(mined.double()) for m in pop]),cfg.disagree,cfg.qg_dist))}
report['seconds']=time.time()-start;Path('audit_lenet/original_student_disagreement_stall.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
