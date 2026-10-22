"""Oracle-sampling diagnostic of C5 committee learning. No extraction claims.
Uses real reachable image queries and output-only optimization, with exact recovered prefix.
Vectorized eight-member tail training caches that prefix; known 103 C5 rows stay fixed.
"""
import argparse, json, time, sys, os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from nets import ConvNet
from align import cnn_canonicalize_,cnn_align_to_
from residual_seed import prefix
p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');p.add_argument('--epochs',type=int,default=600);p.add_argument('--per-channel',type=int,default=256);p.add_argument('--arms',default='gaussian_rownorm,boundary_rownorm,boundary_free');p.add_argument('--dataset-from');p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(21021);dev=torch.device(a.device)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
conv=[(1,6,5,1,2,2),(6,16,5,1,0,2),(16,120,5,1,0,0)]
def net():return ConvNet((1,28,28),conv,(84,),10,'leaky_relu').to(dev).double()
r=torch.load('recon/_record_lenet_c1c2_c5_105.pt',map_location=dev,weights_only=False)
rec=net();rec.load_state_dict(r['partial_state']);cnn_canonicalize_(rec)
t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',map_location=dev,weights_only=True));cnn_canonicalize_(t);cnn_align_to_(t,rec)
for m in (t,rec):
 for q in m.parameters():q.requires_grad_(False)
mask=r['partial_mask'][2].to(dev);free=(~mask).nonzero().flatten();WT=t.layers[2].weight.flatten(1);bt=t.layers[2].bias
start=time.time()
def emit(d):
 d={'seconds':round(time.time()-start,2),**d};print(json.dumps(d),flush=True)
 with (out/'events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
(out/'config.json').write_text(json.dumps({**vars(a),'pid':os.getpid(),'missing_channels':free.tolist(),'oracle_sampling':True,'training_labels':'10 teacher logits only','members':8,'lr':.001,'batch':512,'loss':'L1','prefix':'fp64 record cached','teacher':'canonical aligned tracked LeNet'},indent=2))
emit({'event':'start','pid':os.getpid(),'missing':free.tolist()})
# Construct balanced target preactivations on each missing teacher row, inside actual images.
if a.dataset_from:
 d=torch.load(Path(a.dataset_from)/'dataset.pt',weights_only=True,map_location=dev)
 HG,YG,HB,YB=[d[k].to(dev) for k in ('gaussian_H','gaussian_Y','boundary_H','boundary_Y')]
 emit({'event':'reuse_dataset','source':a.dataset_from,'n':len(HB)})
else:
 xs=[];target_ids=[];target_values=[]
 for j in free.tolist():
  x=torch.randn(a.per_channel,784,device=dev,dtype=torch.double)*.5
  target=torch.cat([-(.1+.8*torch.rand(a.per_channel//2,device=dev,dtype=torch.double)),.1+.8*torch.rand(a.per_channel-a.per_channel//2,device=dev,dtype=torch.double)])
  x.requires_grad_(True);opt=torch.optim.Adam([x],lr=.05)
  for step in range(250):
   h,_=prefix(rec,x,2);z=h@WT[j]+bt[j];loss=(z-target).square().mean();opt.zero_grad();loss.backward();opt.step()
   with torch.no_grad():x.clamp_(-3,3)
  with torch.no_grad():
   h,_=prefix(rec,x,2);z=h@WT[j]+bt[j]
   emit({'event':'target_channel','channel':j,'target_mae':float((z-target).abs().mean()),'positive':int((z>0).sum()),'negative':int((z<0).sum()),'z_min':float(z.min()),'z_max':float(z.max())})
  xs.append(x.detach());target_ids.extend([j]*len(x));target_values.append(target)
 xt=torch.cat(xs); ng=len(xt)
 # Half Gaussian, half targeted; equal dataset sizes across arms.
 xg=torch.randn(ng*2,784,device=dev,dtype=torch.double)*.5
 xb=torch.cat([xg[:ng],xt])
 @torch.no_grad()
 def dataset(x):
  hh=[];yy=[]
  for chunk in x.split(512):
   h,_=prefix(rec,chunk,2);hh.append(h.float());yy.append(t(chunk).float())
  return torch.cat(hh),torch.cat(yy)
 HG,YG=dataset(xg);HB,YB=dataset(xb)
 with torch.no_grad():
  Z=HB.double()@WT.T+bt; pos=(Z>0).double().mean(0)
  emit({'event':'dataset','n':len(HB),'missing_positive_fractions':pos[free].tolist(),'all_positive_min':float(pos.min()),'all_positive_max':float(pos.max()),'feature_rank':int(torch.linalg.matrix_rank(torch.cat([HB.double(),torch.ones(len(HB),1,device=dev,dtype=torch.double)],1))),'positive_design_ranks_missing':[int(torch.linalg.matrix_rank(torch.cat([HB[Z[:,j]>0].double(),torch.ones(int((Z[:,j]>0).sum()),1,device=dev,dtype=torch.double)],1),rtol=1e-5)) for j in free.tolist()]})
  torch.save({'target_images':xt.cpu(),'target_ids':target_ids,'targets':torch.cat(target_values).cpu(),'gaussian_images':xg.cpu(),'boundary_H':HB.cpu(),'boundary_Y':YB.cpu(),'gaussian_H':HG.cpu(),'gaussian_Y':YG.cpu()},out/'dataset.pt')
# Each student is 400->120->84->10. Identical initial population across arms.
P=8
class Committee(torch.nn.Module):
 def __init__(self):
  super().__init__();self.ws=torch.nn.ParameterList();self.bs=torch.nn.ParameterList()
  for din,dout in [(400,120),(120,84),(84,10)]:
   self.ws.append(torch.nn.Parameter(torch.randn(P,dout,din,device=dev)*(2/(din+dout))**.5));self.bs.append(torch.nn.Parameter(torch.zeros(P,dout,device=dev)))
  with torch.no_grad():
   self.ws[0][:,mask]=rec.layers[2].weight.flatten(1)[mask].float();self.bs[0][:,mask]=rec.layers[2].bias[mask].float()
  self.ws[0].register_hook(lambda g:g.masked_fill(mask[None,:,None],0));self.bs[0].register_hook(lambda g:g.masked_fill(mask[None,:],0))
 def forward(self,h):
  if hasattr(self,'whiten_mean'): h=(h-self.whiten_mean)@self.whiten_invroot.T
  v=h[None].expand(P,-1,-1)
  for i,(w,b) in enumerate(zip(self.ws,self.bs)):
   v=torch.bmm(v,w.transpose(1,2))+b[:,None]
   if i<2:v=F.leaky_relu(v,.01)
  return v
 @torch.no_grad()
 def canonicalize(self,h):
  # Match production one-time RMS canonicalization + equal row norms, function preserving.
  v=h[None].expand(P,-1,-1);targets=[]
  for i in range(2):
   z=torch.bmm(v,self.ws[i].transpose(1,2))+self.bs[i][:,None]
   if i==1: # C5 has pins: keep its fixed record gauge; standardize F6 only.
    f=z.square().mean((1,2)).sqrt().clamp_min(1e-10).reciprocal()[:,None].expand(P,len(self.bs[i][0]))
    self.ws[i].mul_(f[:,:,None]);self.bs[i].mul_(f);self.ws[i+1].div_(f[:,None,:])
    n=(self.ws[i].square().sum(2)+self.bs[i].square()).sqrt();f=n.median(1).values[:,None]/n
    self.ws[i].mul_(f[:,:,None]);self.bs[i].mul_(f);self.ws[i+1].div_(f[:,None,:])
   n=(self.ws[i].square().sum(2)+self.bs[i].square()).sqrt();targets.append(n.median(1).values)
   v=F.leaky_relu(torch.bmm(v,self.ws[i].transpose(1,2))+self.bs[i][:,None],.01)
  return targets
 @torch.no_grad()
 def project(self,targets):
  for i in range(2):
   n=(self.ws[i].square().sum(2)+self.bs[i].square()).sqrt().clamp_min(1e-12);f=targets[i][:,None]/n
   if i==0:f[:,mask]=1
   self.ws[i].mul_(f[:,:,None]);self.bs[i].mul_(f)
base=Committee();initial={k:v.clone() for k,v in base.state_dict().items()}
TR=torch.cat([WT[free],bt[free,None]],1);TR=F.normalize(TR,dim=1)
@torch.no_grad()
def metrics(c,h,y):
 # One-to-one signed teacher matching of open rows; teacher used for measurement only.
 Wraw=c.ws[0].double();braw=c.bs[0].double()
 if hasattr(c,'whiten_mean'):
  Wraw=Wraw@c.whiten_invroot.double();braw=braw-Wraw@c.whiten_mean.double()
 R=F.normalize(torch.cat([Wraw[:,free],braw[:,free,None]],2),dim=2)
 aligned=[];cosines=[];errors=[]
 for rr in R:
  C=TR@rr.T;ii,jj=linear_sum_assignment((-C).cpu().numpy());ordered=rr[torch.as_tensor(jj,device=dev)];aligned.append(ordered);cosines.append((TR*ordered).sum(1));errors.append((TR-ordered).abs().amax(1))
 A=torch.stack(aligned);C=torch.stack(cosines);E=torch.stack(errors)
 D=(A[:,None]-A[None,:]).abs().amax(3) # P,P,missing
 votes=(D<.02).sum(1).amax(0)
 # Agreement alone can be wrong. Count jointly accurate members for each teacher row.
 accurate=(E<.02).sum(0)
 losses=[]
 for hh,yy in zip(h.split(1024),y.split(1024)):losses.append((c(hh)-yy[None]).abs().sum((1,2)))
 loss=torch.stack(losses).sum(0)/(len(h)*10)
 return {'loss_by_member':loss.tolist(),'best_cosine_by_missing_row':C.max(0).values.tolist(),'median_cosine_by_missing_row':C.median(0).values.tolist(),'best_inf_error_by_missing_row':E.min(0).values.tolist(),'missing_with_any_cos99':int((C.max(0).values>.99).sum()),'missing_with_any_inf02':int((E.min(0).values<.02).sum()),'missing_with_4_accurate_inf02':int((accurate>=4).sum()),'missing_with_4_agree_inf02':int((votes>=4).sum()),'accurate_member_counts_inf02':accurate.tolist(),'pinned_max_error':max(float((Wraw[:,mask]-rec.layers[2].weight.flatten(1)[mask]).abs().max()),float((braw[:,mask]-rec.layers[2].bias[mask]).abs().max()))}
for arm in a.arms.split(','):
 c=Committee();c.load_state_dict(initial);h,y=(HG,YG) if arm.startswith('gaussian') else (HB,YB);opt=torch.optim.Adam(c.parameters(),lr=.001);targets=None
 if 'fixedtail' in arm:
  with torch.no_grad():
   for i in (1,2):
    c.ws[i].copy_(t.layers[i+2].weight.float()[None]);c.bs[i].copy_(t.layers[i+2].bias.float()[None])
  for i in (1,2):c.ws[i].requires_grad_(False);c.bs[i].requires_grad_(False)
 if 'whiten' in arm:
  with torch.no_grad():
   mean=h.double().mean(0);hc=h.double()-mean;ev,U=torch.linalg.eigh(hc.T@hc/len(h));root=(U*ev.clamp_min(1e-10).sqrt()[None])@U.T;invroot=(U*ev.clamp_min(1e-10).rsqrt()[None])@U.T
   oldw=c.ws[0].double().clone();c.ws[0].copy_((oldw@root).float());c.bs[0].add_((oldw@mean).float())
   c.register_buffer('whiten_mean',mean.float());c.register_buffer('whiten_invroot',invroot.float())
 emit({'event':'train_start','arm':arm,**metrics(c,h,y)})
 for epoch in range(1,a.epochs+1):
  if epoch in (int(a.epochs*.6),int(a.epochs*.85)):
   for g in opt.param_groups:g['lr']*=.1
  order=torch.randperm(len(h),device=dev,generator=torch.Generator(device=dev).manual_seed(91000+epoch))
  for ix in order.split(512):
   opt.zero_grad();res=c(h[ix])-y[ix][None];loss=(res.square() if 'mse' in arm else res.abs()).mean((1,2)).sum();loss.backward();opt.step()
   if targets is not None and arm.endswith('rownorm'):c.project(targets)
  if epoch==1 and 'fixedtail' not in arm and 'whiten' not in arm:targets=c.canonicalize(h[:512])
  if epoch==1 or epoch%25==0 or epoch==a.epochs:
   met=metrics(c,h,y);emit({'event':'train','arm':arm,'epoch':epoch,'lr':opt.param_groups[0]['lr'],**met})
   torch.save({'model':c.state_dict(),'epoch':epoch,'arm':arm,'metrics':met,'targets':targets},out/f'{arm}_latest.pt')
   if met['missing_with_4_accurate_inf02']==len(free):
    emit({'event':'all_missing_accurate','arm':arm,'epoch':epoch});break
 emit({'event':'train_done','arm':arm,'epoch':epoch,**metrics(c,h,y)})
emit({'event':'done'})
