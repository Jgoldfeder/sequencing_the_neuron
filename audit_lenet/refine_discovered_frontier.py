"""Refine independently discovered C5 hyperplanes by native surface tracking.
Selection uses only known-row span; teacher matching is post-fit scoring only.
"""
import argparse,sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
from residual_seed import prefix
from align import cnn_canonicalize_,cnn_align_to_
import kink_solve as K
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--gpu',type=int,default=2);p.add_argument('--tag',default='direct_c5_polish_20261021_a');p.add_argument('--fraction',type=float,default=.4);p.add_argument('--step',type=float,default=2.0);p.add_argument('--need',type=int,default=1200);p.add_argument('--record',default='recon/_record_lenet_c1c2_c5_105.pt');p.add_argument('--frontier',type=int,default=2);p.add_argument('--only',default='');a=p.parse_args();fr=a.frontier
torch.set_num_threads(4);torch.manual_seed(22601);dev=torch.device(f'cuda:{a.gpu}');D=torch.double;start=time.time();g=torch.Generator(device=dev).manual_seed(22602)
r=torch.load(a.record,map_location=dev,weights_only=False)
def net():return ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).to(dev).double()
n=net();n.load_state_dict(r['partial_state'])
for l,(W,b,m) in r['frozen'].items():
 with torch.no_grad():n.layers[l].weight[m]=W[m];n.layers[l].bias[m]=b[m]
for pp in n.parameters():pp.requires_grad_(False)
t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',map_location=dev,weights_only=True))
for pp in t.parameters():pp.requires_grad_(False)
z=torch.load(a.source,map_location=dev,weights_only=False);mask=r['partial_mask'][fr].to(dev);free=(~mask).nonzero().flatten();W=n.layers[fr].weight.flatten(1)[mask];Q=torch.linalg.qr(W.T).Q
selected=[]
for i,f in enumerate(z['hyperplanes']):
 if a.only and i not in {int(v) for v in a.only.split(',')}:continue
 row=f['row'];w=row[:-1];known_aug=torch.cat([n.layers[fr].weight.flatten(1)[mask],n.layers[fr].bias[mask,None]],1);known_aug=known_aug/known_aug.norm(dim=1,keepdim=True).clamp_min(1e-30)
 if len(known_aug) and bool(((known_aug@row).abs()>.99999).any()):continue
 frac=float((w-Q@(Q.T@w)).norm()/w.norm())
 if frac>=a.fraction:selected.append((i,f,frac))
print('selected by known-row complement',[(i,fr) for i,f,fr in selected],flush=True)
NQ=0
@torch.no_grad()
def bb(x):
 global NQ
 NQ+=len(x);return torch.cat([t(xx) for xx in x.split(1024)])
# Grade-only reference, never used in selection, fitting, or validation.
ts=t.clone();ns=n.clone();cnn_canonicalize_(ts);cnn_canonicalize_(ns);cnn_align_to_(ts,ns)
def grade(row):
 mm=n.clone()
 with torch.no_grad():mm.layers[fr].weight[0]=row[:-1].view_as(mm.layers[fr].weight[0]);mm.layers[fr].bias[0]=row[-1]
 cnn_canonicalize_(mm);rr=torch.cat([mm.layers[fr].weight[0].flatten(),mm.layers[fr].bias[:1]]);vv=torch.cat([ts.layers[fr].weight.flatten(1),ts.layers[fr].bias[:,None]],1);cos=vv@rr;j=int(cos.abs().argmax());sg=torch.sign(cos[j]);return {'teacher_channel_in_record_order':j,'abs_cosine':float(cos[j].abs()),'max_error_up_to_sign':float((rr*sg-vv[j]).abs().max()),'orientation_score_only':float(sg)}
# A deep prefix can turn the default uncertainty radius into a bracket with
# many foreign bends. Retry smaller brackets without weakening any light-locator gate.
if fr>=3:
 _base_light=K.locate_light
 @torch.no_grad()
 def adaptive_light(oracle,X0,U,R,tol=1e-12):
  result=X0.clone();good=torch.zeros(len(X0),device=X0.device,dtype=torch.bool)
  for scale in [.1,.03,.01,.003,.001]:
   idx=(~good).nonzero().flatten()
   if not len(idx):break
   out,ok=_base_light(oracle,X0[idx],U[idx],R[idx]*scale,tol)
   result[idx[ok]]=out[ok];good[idx[ok]]=True
  return result,good
 K.locate_light=adaptive_light
results=[]
for seq,(idx,f,frac) in enumerate(selected):
 c=int(free[seq%len(free)]);m=n.clone();row=f['row']
 with torch.no_grad():m.layers[fr].weight[c]=row[:-1].view_as(m.layers[fr].weight[c]);m.layers[fr].bias[c]=row[-1]
 M=K.ConvNetUnits(m,fr);anchor={c:(f['xstar'][None],torch.tensor([c],device=dev))}
 print('track candidate',idx,'queries',NQ,flush=True)
 hs=K.track_layer(bb,M,fr,anchor,g,need=a.need,step=a.step,max_steps=600,par=16,verbose=True)
 if c not in hs:print('not enough tracked points',idx,flush=True);continue
 H=hs[c];fit=K._cnn_solve_points(H,row[:-1],gen=g)
 if fit is None:print('fit failed',idx,flush=True);continue
 w,b,_=fit;quality,stats=K._cnn_fit_quality(H,w,b);out=torch.cat([w,b[None]]);out/=out.norm()
 print('fit',idx,'quality',quality,stats,'score_only',grade(out),flush=True)
 # Independent new points: project onto the fitted candidate and test the output bend.
 x=(f['xstar'][None]+torch.randn(32,784,device=dev,dtype=D,generator=g)*2).requires_grad_(True)
 for it in range(30):
  val=prefix(n,x,fr)[0]@out[:-1]+out[-1];grad=torch.autograd.grad(val.sum(),x)[0]
  with torch.no_grad():x.sub_(val[:,None]*grad/grad.square().sum(1,keepdim=True).clamp_min(1e-30))
 v=grad.detach();v/=v.norm(dim=1,keepdim=True);x=x.detach();e=1e-5;ss=torch.tensor([-2,-1,0,1,2],device=dev,dtype=D)*e
 yy=bb((x[:,None]+ss[None,:,None]*v[:,None]).reshape(-1,784)).reshape(32,5,-1);dd=(yy[:,2:]-2*yy[:,1:-1]+yy[:,:-2]).norm(dim=2);hits=int((dd[:,1]>100*torch.maximum(dd[:,0],dd[:,2]).clamp_min(1e-13)).sum())
 result={'source_index':idx,'row':out.detach().cpu(),'H':H.detach().cpu(),'quality':quality,'quality_stats':stats,'independent_kink_hits':hits,'independent_kink_tests':32,'queries':NQ,'score_only':grade(out)};results.append(result)
 print('VERIFY',idx,hits,'/32',result['score_only'],flush=True)
 torch.save({'results':results,'source':a.source,'queries':NQ},f'audit_lenet/{a.tag}.pt')
 Path(f'audit_lenet/{a.tag}.json').write_text(json.dumps([{k:v for k,v in rr.items() if k not in ['row','H']} for rr in results],indent=2)+'\n')
print('DONE',len(results),'queries',NQ,'seconds',time.time()-start,flush=True)
