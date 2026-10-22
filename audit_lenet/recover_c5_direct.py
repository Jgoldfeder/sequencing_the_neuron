"""Direct black-box C5 hyperplane recovery from the attributable fp64 record.
No unknown teacher coefficients enter search, mapback, clustering, or verification.
Teacher rows are accessed only in the separately labelled progress score.
Unoriented hyperplanes are NOT advertised as exact signed recovered neurons.
"""
import argparse,json,sys,time,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
from residual_seed import prefix,residual_subspace,seeds,_prefix_jac
import diag_guessfree_layer as G
from align import cnn_canonicalize_,cnn_align_to_
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--lines',type=int,default=24);p.add_argument('--grid',type=int,default=32001);p.add_argument('--tag',default='direct_c5_20261021_a');p.add_argument('--setup',default='');p.add_argument('--resume',default='');p.add_argument('--first-line',type=int,default=0);p.add_argument('--record',default='recon/_record_lenet_c1c2_c5_105.pt');a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(22501);dev=torch.device(f'cuda:{a.gpu}');D=torch.double;start=time.time();gen=torch.Generator(device=dev).manual_seed(22502)
r=torch.load(a.record,weights_only=False,map_location=dev)
def net():return ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).to(dev).double()
n=net();n.load_state_dict(r['partial_state']);mask=r['partial_mask'][2].to(dev);known=mask.nonzero().flatten().tolist()
# Prefix and pinned rows from the persistent record, with no fp32 round trip.
for l,(W,b,m) in r['frozen'].items():
 with torch.no_grad():n.layers[l].weight[m]=W[m];n.layers[l].bias[m]=b[m]
for pp in n.parameters():pp.requires_grad_(False)
t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',weights_only=True,map_location=dev));t.eval()
for pp in t.parameters():pp.requires_grad_(False)
NQ=0
@torch.no_grad()
def bb(x):
 global NQ
 NQ+=len(x)
 return torch.cat([t(xx) for xx in x.split(1024)])
class Oracle:
 def __call__(self,x):return bb(x)
oracle=Oracle()
Wk=torch.cat([n.layers[2].weight.flatten(1),n.layers[2].bias[:,None]],1)[mask];Wk=Wk/Wk.norm(dim=1,keepdim=True)
# Teacher scoring is isolated: these arrays are never consulted by the solver.
ts=t.clone();cnn_canonicalize_(ts);ns=n.clone();cnn_canonicalize_(ns);cnn_align_to_(ts,ns)
# n's prefix is canonical in this record; handle its gauge explicitly for grading.
def score(row):
 mm=n.clone()
 with torch.no_grad():mm.layers[2].weight[0]=row[:-1].view_as(mm.layers[2].weight[0]);mm.layers[2].bias[0]=row[-1]
 cnn_canonicalize_(mm);rr=torch.cat([mm.layers[2].weight[0].flatten(),mm.layers[2].bias[:1]])
 V=torch.cat([ts.layers[2].weight.flatten(1),ts.layers[2].bias[:,None]],1);cs=V@rr;j=int(cs.abs().argmax());sgn=torch.sign(cs[j]);err=float((rr*sgn-V[j]).abs().max())
 return {'teacher_channel_in_record_order':j,'abs_cosine':float(cs[j].abs()),'max_coefficient_error_up_to_sign':err,'teacher_orientation':float(sgn)}
found=[];accepted=[];stats=[]
Qknown=torch.linalg.qr(n.layers[2].weight.flatten(1)[mask].T).Q
if a.resume:
 old=torch.load(a.resume,map_location='cpu',weights_only=False);found=old['hyperplanes'];stats=old['stats']
def save():
 torch.save({'hyperplanes':found,'accepted_indices':accepted,'known_mask':mask.cpu(),'record_path':a.record,'queries':NQ,'stats':stats},f'audit_lenet/{a.tag}.pt')
 Path(f'audit_lenet/{a.tag}.json').write_text(json.dumps({'queries':NQ,'seconds':time.time()-start,'found':len(found),'accepted':len(accepted),'stats':stats},indent=2)+'\n')
# Find an isolated kink to machine precision by intersecting its two affine pieces.
def locate(x0,d,lo,hi):
 root=G.bisect_kink(oracle,x0,d,float(lo),float(hi))
 delta=min(float(hi-lo)*.12,2e-6)
 for _ in range(8):
  ss=torch.tensor([root-2*delta,root-delta,root+delta,root+2*delta],device=dev,dtype=D);yy=bb(x0+ss[:,None]*d)
  sl=(yy[1]-yy[0])/delta;sr=(yy[3]-yy[2])/delta;diff=sl-sr
  if float(diff.norm())<1e-9:return None
  bl=yy[1]-sl*ss[1];br=yy[2]-sr*ss[2];rr=float(((br-bl)*diff).sum()/diff.square().sum())
  if abs(rr-root)>delta*2:return None
  residual=(bl+sl*rr-br-sr*rr).norm()
  root=rr
  if float(residual)<1e-10:return root,delta
  delta*=.25
 return None
# Recover a full frontier normal from an isolated rank-one output-gradient jump.
def recover(x0,d,rt,delta):
 xstar=x0+rt*d;P=_prefix_jac(n,xstar,2);pinv=torch.linalg.pinv(P,rtol=1e-10)
 # Use full image finite differences; reject any step crossing a known prefix boundary.
 for dt in [delta*.25,delta*.0625]:
  xc=torch.stack([xstar-dt*d,xstar+dt*d]);_,zz=prefix(n,xc,2)
  if bool(((zz[0]>0)!=(zz[1]>0)).any()):continue
  eps=min(1e-7,dt*.1);eye=torch.eye(784,device=dev,dtype=D)*eps;xs=torch.cat([xc[0:1],xc[0]+eye,xc[1:2],xc[1]+eye])
  # Prefix-region equality rules out contaminated finite-difference columns.
  with torch.no_grad():
   patterns=torch.cat([prefix(n,xx,2)[1]>0 for xx in xs.split(256)])
   if bool((patterns[:785]!=patterns[0]).any()) or bool((patterns[785:]!=patterns[785]).any()):continue
  ys=bb(xs);jl=((ys[1:785]-ys[0])/eps).T;jr=((ys[786:]-ys[785])/eps).T;dj=jr-jl
  U,S,V=torch.linalg.svd(dj,full_matrices=False)
  if float(S[0])<1e-7 or float(S[1]/S[0])>1e-4:continue
  w=pinv.T@V[0];err=float((P.T@w-V[0]).norm())
  if err>1e-4:continue
  h=prefix(n,xstar[None],2)[0][0];row=torch.cat([w,(-w@h)[None]]);row/=row.norm()
  return row,xstar,float(S[1]/S[0]),err
 return None
# Verify the candidate on fresh points on its hyperplane, far from the discovery point.
def verify(row,xstar,tests=6):
 points=[];hits=0
 for trial in range(tests):
  x=(xstar+torch.randn(784,device=dev,dtype=D,generator=gen)*(.25+.15*trial)).requires_grad_(True)
  opt=torch.optim.Adam([x],lr=.03)
  for it in range(70):
   opt.zero_grad();val=prefix(n,x[None],2)[0][0]@row[:-1]+row[-1];val.square().backward();opt.step()
  # Newton projection onto the candidate using the exact attacker prefix.
  for it in range(15):
   val=prefix(n,x[None],2)[0][0]@row[:-1]+row[-1];grad=torch.autograd.grad(val,x)[0]
   with torch.no_grad():x.sub_(val*grad/grad.square().sum().clamp_min(1e-30))
   if abs(float(val))<1e-13:break
  xx=x.detach();v=grad.detach();v=v/v.norm();ep=2e-5
  # Wide enough window for numeric candidate error, then test a much smaller exact jump.
  loc=locate(xx,v,-ep,ep)
  if loc is None:continue
  rt,dd=loc
  if abs(rt)>2e-6:continue
  xp=xx+rt*v;yy=bb(xp[None]+torch.tensor([-2e-6,-1e-6,0,1e-6,2e-6],device=dev,dtype=D)[:,None]*v)
  ddv=(yy[2:]-2*yy[1:-1]+yy[:-2]).norm(dim=1)
  if float(ddv[1])>100*max(float(ddv[0]),float(ddv[2]),1e-13):hits+=1;points.append(prefix(n,xp[None],2)[0][0].detach())
 return hits,points
print('start direct output-only extraction; pinned',len(known),flush=True)
if a.setup:
 cached=torch.load(a.setup,map_location=dev,weights_only=False);V=cached['V'];ends=cached['ends'];NQ=cached['queries']
else:
 V,S,nq=residual_subspace(bb,n,2,mask,n_points=40,gen=gen)
 if not len(V):raise RuntimeError('Invalid residual subspace')
 print('constructing reachable residual endpoints',flush=True)
 ends=seeds(n,2,V,256,s_max=12,steps=400,box=4,gen=gen)
torch.save({'V':V,'ends':ends,'queries':NQ},f'audit_lenet/{a.tag}_setup.pt')
starts=torch.randn(256,784,device=dev,dtype=D,generator=gen)*.5
for li in range(a.first_line,a.lines):
 x0=(starts[li]+ends[li])/2;d=(ends[li]-starts[li])/2
 unk,count,step=G.scan_line(oracle,n,2,x0,d,1,a.grid,known)
 print('line',li,'unexplained',len(unk),'total_grid_kinks',count,'queries',NQ,flush=True)
 new=0
 for rr in unk:
  loc=locate(x0,d,rr-float(step),rr+float(step))
  if loc is None:continue
  result=recover(x0,d,*loc)
  if result is None:continue
  row,xstar,ratio,err=result
  if bool(((Wk@row).abs()>.99999).any()):continue
  if found and bool(((torch.stack([f['row'] for f in found]).to(dev)@row).abs()>.99999).any()):continue
  w=row[:-1];fraction=float((w-Qknown@(Qknown.T@w)).norm()/w.norm())
  if fraction<.4:continue
  hit,pts=0,[]
  entry={'row':row.detach().cpu(),'xstar':xstar.detach().cpu(),'verification_points':torch.stack(pts).cpu() if pts else None,'verification_hits':hit,'rank_ratio':ratio,'mapback_residual':err}
  found.append(entry);new+=1
  # Grade AFTER the attacker's accept/reject decision.
  grade=score(row);entry['score_only']=grade
  if hit>=5:accepted.append(len(found)-1)
  print('candidate',len(found),'verified',hit,'/6','grade_only',grade,flush=True);save()
 stats.append({'line':li,'unexplained':len(unk),'new_candidates':new,'accepted_total':len(accepted),'queries':NQ,'seconds':time.time()-start});save()
 print('line complete',stats[-1],flush=True)
 if len(accepted)>=17:break
save();print('DONE',len(accepted),'verified hyperplanes;',NQ,'queries;',time.time()-start,'seconds',flush=True)
