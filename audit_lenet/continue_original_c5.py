"""Controlled adaptive warm restart; NOT a replay of the unsaved historical committee.
Original image optimizer, batch128/L1/Adam; learned suffix; teacher only labels/scores.
Feature arm changes only sampling to canonical C5 preactivation disagreement.
Fixed prefix cached; full vs cached Adam-step equivalence checked before training.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scipy.optimize import linear_sum_assignment
from nets import ConvNet
from align import cnn_canonicalize_,cnn_align_to_
from method import Cfg,gen_queries,_cnn_frontier_input,_cnn_tail_forward,_cnn_install_freeze
from run import VARIANTS
import cnn_balance as cb
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['output','feature'],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--iterations',type=int,default=6);a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(22121);dev=torch.device(f'cuda:{a.gpu}');torch.cuda.manual_seed_all(22121);start=time.time()
r=torch.load('recon/_record_lenet_c1c2_c5_105.pt',weights_only=False,map_location=dev)
def net():return ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).to(dev)
base=net().double();base.load_state_dict(r['state_dict']);cnn_canonicalize_(base);base=base.float()
t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',weights_only=True,map_location=dev));t.eval()
for pp in t.parameters():pp.requires_grad_(False)
mask=r['partial_mask'][2].to(dev);free=(~mask).nonzero().flatten();pm={0:torch.ones(6,device=dev,dtype=torch.bool),1:torch.ones(16,device=dev,dtype=torch.bool),2:mask}
frozen={l:(base.layers[l].weight.detach().clone(),base.layers[l].bias.detach().clone(),mk) for l,mk in pm.items()}
def pin(m):
 _cnn_install_freeze(m,frozen)
 for l in [0,1]:
  m.layers[l].weight.requires_grad_(False);m.layers[l].bias.requires_grad_(False)
# Verify the cached native Conv2d tail against a full-network Adam step.
u=net();u.load_state_dict(base.state_dict());v=net();v.load_state_dict(base.state_dict());pin(u);pin(v)
x=torch.randn(128,784,device=dev)*.5;y=torch.randn(128,10,device=dev)
ou=torch.optim.Adam(filter(lambda pp:pp.requires_grad,u.parameters()),lr=.001);ov=torch.optim.Adam(filter(lambda pp:pp.requires_grad,v.parameters()),lr=.001)
pu=u(x);pv=_cnn_tail_forward(v,_cnn_frontier_input(v,x,2).detach(),2);(pu-y).abs().mean().backward();(pv-y).abs().mean().backward()
gerr=max(float((pp.grad-qq.grad).abs().max()) for pp,qq in zip(u.parameters(),v.parameters()) if pp.grad is not None);ou.step();ov.step();perr=max(float((pp-qq).abs().max()) for pp,qq in zip(u.parameters(),v.parameters()))
assert gerr<1e-6 and perr<1e-6,(gerr,perr)
# Identical warm states in both arms, small perturbations only in unknown parameters.
torch.manual_seed(22122);pop=[];opts=[]
for i in range(8):
 m=net();m.load_state_dict(base.state_dict())
 with torch.no_grad():
  for l in range(2,len(m.layers)):
   lay=m.layers[l];scale=lay.weight.std()*.02
   if l==2:
    lay.weight[free]+=torch.randn_like(lay.weight[free])*scale;lay.bias[free]+=torch.randn_like(lay.bias[free])*scale
   else:
    lay.weight.add_(torch.randn_like(lay.weight)*scale);lay.bias.add_(torch.randn_like(lay.bias)*scale)
 pin(m);pop.append(m);opts.append(torch.optim.Adam(filter(lambda pp:pp.requires_grad,m.parameters()),lr=.001))
# Initial row targets from the actual warm state; preserve all pinned coefficients.
targets=[[None,None,cb.row_norms(m.layers[2]).detach().clone(),cb.row_norms(m.layers[3]).detach().clone()] for m in pop]
g=torch.Generator(device=dev).manual_seed(22123);cfg=Cfg(p=8,q=2048,act='leaky_relu',**VARIANTS['mergedbest']);X=[];Y=[];report={'mode':a.mode,'iterations':a.iterations,'q_per_iteration':cfg.q,'epochs':10,'batch':128,'seed':22121,'pinned_c5':int(mask.sum()),'perturbation_fraction':.02,'adam_lr':.001,'cache_equivalence':{'max_gradient_error':gerr,'max_adam_parameter_error':perr},'limitations':'warm restart of one saved student, new Adam, no original buffer/committee, no new-row pinning; initial row targets preserved; not historical replay','rows':[]}
# Score with teacher only here. Prefix canonical/permutation alignment, then signed C5 matching.
@torch.no_grad()
def score(it):
 scores=[]
 for m in pop:
  mm=m.clone().double();tt=t.clone().double();cnn_canonicalize_(mm);cnn_canonicalize_(tt);cnn_align_to_(tt,mm)
  W=torch.cat([mm.layers[2].weight.flatten(1),mm.layers[2].bias[:,None]],1);V=torch.cat([tt.layers[2].weight.flatten(1),tt.layers[2].bias[:,None]],1);cos=W@V.T;ii,jj=linear_sum_assignment((-cos).cpu().numpy());cs=cos[ii,jj];err=(W[ii]-V[jj]).abs().amax(1)
  scores.append({'cos99':int((cs>.99).sum()),'cos999':int((cs>.999).sum()),'max_coefficient_error_below_002':int((err<.02).sum()),'free_best_cosines':sorted(cs[~mask].tolist())})
 row={'iteration':it,'seconds':time.time()-start,'members':scores};report['rows'].append(row);print(json.dumps(row),flush=True);Path(f'audit_lenet/adaptive_c5_{a.mode}_20261021.json').write_text(json.dumps(report,indent=2)+'\n')
score(0)
for it in range(1,a.iterations+1):
 features=None
 if a.mode=='feature':
  # Prefix and channel order stay shared. Normalize each augmented row; do not permute.
  scales=[cb.row_norms(m.layers[2]).detach().clamp_min(1e-8) for m in pop]
  features=[lambda xx,m=m,s=s: ((_cnn_frontier_input(m,xx,2)@m.layers[2].weight.flatten(1).T+m.layers[2].bias)/s)[:,free] for m,s in zip(pop,scales)]
 I=gen_queries(pop,cfg,784,dev,g,feats=features)
 with torch.no_grad():
  X.append(_cnn_frontier_input(pop[0],I,2).detach());Y.append(t(I).detach())
 H=torch.cat(X);T=torch.cat(Y)
 for ep in range(10):
  order=torch.randperm(len(H),generator=g,device=dev)
  for idx in order.split(128):
   for m,opt,tg in zip(pop,opts,targets):
    opt.zero_grad();loss=(_cnn_tail_forward(m,H[idx],2)-T[idx]).abs().mean();loss.backward();opt.step();cb.project_rows_(m,tg,pm)
 with torch.no_grad():
  report['last_train_L1']=[float((_cnn_tail_forward(m,H,2)-T).abs().mean()) for m in pop]
  report['last_query_mean_abs_pixel']=float(I.abs().mean())
 score(it)
torch.save({'states':[m.state_dict() for m in pop],'report':report},f'audit_lenet/adaptive_c5_{a.mode}_20261021.pt')
