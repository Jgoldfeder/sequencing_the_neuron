"""Test original disagreement on exact alternative representations of actual LeNet C5.
Teacher internals construct a mathematical counterexample, not learner supervision.
No oracle suffix is supplied to a learner and no layer-recovery claim is made.
"""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
from align import cnn_canonicalize_,cnn_align_to_
from method import Cfg,gen_queries,disagreement
from run import VARIANTS
from residual_seed import prefix

torch.set_num_threads(4);torch.manual_seed(22101);dev=torch.device('cuda:2');start=time.time()
r=torch.load('recon/_record_lenet_c1c2_c5_105.pt',weights_only=False,map_location=dev)
def net():return ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).to(dev).double()
rec=net();rec.load_state_dict(r['partial_state']);cnn_canonicalize_(rec)
t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',weights_only=True,map_location=dev));cnn_canonicalize_(t);cnn_align_to_(t,rec)
for p in t.parameters():p.requires_grad_(False)
g=torch.Generator(device=dev).manual_seed(2261)
X=torch.cat([torch.randn(20000,784,device=dev,generator=g,dtype=torch.double)*.5,torch.rand(20000,784,device=dev,generator=g,dtype=torch.double)*2-1])
with torch.no_grad():
 zs=[]
 for xx in X.split(512):h,_=prefix(t,xx,2);zs.append(h@t.layers[2].weight.flatten(1).T+t.layers[2].bias)
 Z=torch.cat(zs);free=(~r['partial_mask'][2]).nonzero().flatten();silent=free[(Z[:,free]<=0).all(0)];frac=(Z>0).double().mean(0)
assert len(silent)>1
pop=[]
for i in range(8):
 m=t.clone();n=len(silent);strength=.25+.02*i;T=(1-strength)*torch.eye(n,device=dev,dtype=torch.double)+strength*torch.roll(torch.eye(n,device=dev,dtype=torch.double),i+1,0)
 with torch.no_grad():
  W=m.layers[2].weight.flatten(1);W[silent]=T@W[silent];m.layers[2].bias[silent]=T@m.layers[2].bias[silent]
  m.layers[3].weight[:,silent]=m.layers[3].weight[:,silent]@torch.linalg.inv(T)
 for p in m.parameters():p.requires_grad_(False)
 pop.append(m)
class Cast(torch.nn.Module):
 def __init__(self,m):super().__init__();self.m=m
 def forward(self,x):return self.m(x.to(next(self.m.parameters()).dtype))
report={'silent_missing_channels':silent.tolist(),'missing_channels':free.tolist(),'teacher_missing_positive_fractions':frac[free].tolist(),'probe_images':len(X),'arms':[]}
with torch.no_grad():
 report['max_output_error_fp64_on_40000_images']=max(float((m(xx)-t(xx)).abs().max()) for xx in X.split(512) for m in pop)
 raw=torch.cat([t.layers[2].weight.flatten(1),t.layers[2].bias[:,None]],1);orig=torch.nn.functional.normalize(raw,dim=1)
 report['minimum_row_cosines_after_mixing']=[float((torch.nn.functional.normalize(torch.cat([m.layers[2].weight.flatten(1),m.layers[2].bias[:,None]],1),dim=1)[silent]*orig[silent]).sum(1).min()) for m in pop]
 # Are the corresponding unpinned units of the actual original best snapshot also silent?
 actual=net();actual.load_state_dict(r['state_dict']);cnn_canonicalize_(actual);perms=cnn_align_to_(actual,t)
 hf=[]
 for xx in X.split(512):h,_=prefix(actual,xx,2);hf.append(h@actual.layers[2].weight.flatten(1).T+actual.layers[2].bias)
 actualfrac=(torch.cat(hf)>0).double().mean(0)
 report['original_best_missing_positive_fractions']=actualfrac[free].tolist()
for precision in ['fp64','fp32']:
 models=[Cast(m.clone().to(torch.double if precision=='fp64' else torch.float)) for m in pop]
 cfg=Cfg(p=8,q=512,act='leaky_relu',**VARIANTS['mergedbest'])
 G=torch.Generator(device=dev).manual_seed(3301);initial=torch.rand(cfg.q,784,device=dev,generator=G)*2-1
 leaf=initial.clone().requires_grad_(True);outs=torch.stack([m(leaf) for m in models]);loss=disagreement(outs,cfg.disagree,cfg.qg_dist);grad=torch.autograd.grad(loss,leaf)[0]
 mined=gen_queries(models,cfg,784,dev,torch.Generator(device=dev).manual_seed(3301))
 with torch.no_grad():
  h,_=prefix(t,mined.double(),2);z=h@t.layers[2].weight.flatten(1).T+t.layers[2].bias
  end=disagreement(torch.stack([m(mined) for m in models]),cfg.disagree,cfg.qg_dist)
  d={'precision':precision,'initial_disagreement':-float(loss),'final_disagreement':-float(end),'initial_max_abs_query_gradient':float(grad.abs().max()),'max_pixel_movement':float((mined-initial).abs().max()),'silent_channels_reached':int((z[:,silent]>0).any(0).sum()),'silent_channels_total':len(silent)}
 report['arms'].append(d);print(json.dumps(d),flush=True)
report['seconds']=time.time()-start
Path('audit_lenet/original_disagreement_blindness.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
