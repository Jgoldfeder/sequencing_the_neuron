"""Oracle isolation control: learn each missing C5 row separately with other 119 rows
and downstream layers exact. Eight independent fits per row. Output labels only.
This is a diagnostic, not an extraction algorithm.
"""
import argparse,json,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from nets import ConvNet
from align import cnn_canonicalize_,cnn_align_to_
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:2');p.add_argument('--epochs',type=int,default=3000);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
torch.set_num_threads(4);torch.manual_seed(21022);dev=torch.device(a.device);torch.backends.cuda.matmul.allow_tf32=False
conv=[(1,6,5,1,2,2),(6,16,5,1,0,2),(16,120,5,1,0,0)]
def net():return ConvNet((1,28,28),conv,(84,),10,'leaky_relu').to(dev).double()
r=torch.load('recon/_record_lenet_c1c2_c5_105.pt',map_location=dev,weights_only=False);rec=net();rec.load_state_dict(r['partial_state']);cnn_canonicalize_(rec)
t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',map_location=dev,weights_only=True));cnn_canonicalize_(t);cnn_align_to_(t,rec)
for q in t.parameters():q.requires_grad_(False)
d=torch.load(Path(a.data)/'dataset.pt',weights_only=True,map_location=dev);H=d['boundary_H'];Y=d['boundary_Y'];free=(~r['partial_mask'][2]).nonzero().flatten();ids=free.repeat_interleave(8);count=len(ids)
with torch.no_grad():
 WT=t.layers[2].weight.flatten(1).float();bt=t.layers[2].bias.float();W6=t.layers[3].weight.float();b6=t.layers[3].bias.float();WO=t.layers[4].weight.float();bo=t.layers[4].bias.float()
 A=F.leaky_relu(H@WT.T+bt,.01);Z6=A@W6.T+b6;old=A[:,ids];cols=W6[:,ids].T
 mean=H.double().mean(0);hc=H.double()-mean;ev,U=torch.linalg.eigh(hc.T@hc/len(H));root=(U*ev.sqrt()[None])@U.T;inv=(U*ev.rsqrt()[None])@U.T;X=((H.double()-mean)@inv.T).float()
 Winit=torch.randn(count,400,device=dev)*(2/520)**.5;binit=torch.zeros(count,device=dev)
 W=torch.nn.Parameter((Winit.double()@root).float());b=torch.nn.Parameter((binit.double()+Winit.double()@mean).float())
 trueR=F.normalize(torch.cat([WT[ids],bt[ids,None]],1).double(),dim=1)
start=time.time()
def emit(d):
 d={'seconds':round(time.time()-start,2),**d};print(json.dumps(d),flush=True)
 with (out/'events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
(out/'config.json').write_text(json.dumps({**vars(a),'pid':os.getpid(),'missing':free.tolist(),'oracle_control':'119 other C5 rows and complete suffix exact; 8 independent fits per missing row; no hidden-label loss'},indent=2))
opt=torch.optim.Adam([W,b],lr=.003)
@torch.no_grad()
def metrics():
 wr=W.double()@inv;bb=b.double()-wr@mean;R=F.normalize(torch.cat([wr,bb[:,None]],1),dim=1);cos=(R*trueR).sum(1).reshape(-1,8);err=(R-trueR).abs().amax(1).reshape(-1,8)
 return {'best_cosines':cos.max(1).values.tolist(),'best_inf_errors':err.min(1).values.tolist(),'rows_any_cos99':int((cos.max(1).values>.99).sum()),'rows_4_cos99':int(((cos>.99).sum(1)>=4).sum()),'rows_any_inf02':int((err.min(1).values<.02).sum()),'rows_4_inf02':int(((err<.02).sum(1)>=4).sum())}
emit({'event':'start',**metrics()})
for epoch in range(1,a.epochs+1):
 if epoch in (int(a.epochs*.6),int(a.epochs*.85)):
  for g in opt.param_groups:g['lr']*=.1
 order=torch.randperm(len(X),device=dev)
 lossacc=0.
 for ix in order.split(2048):
  z=X[ix]@W.T+b;delta=F.leaky_relu(z,.01)-old[ix];z6=Z6[ix,None,:]+delta[:,:,None]*cols[None];y=F.leaky_relu(z6,.01)@WO.T+bo
  loss=(y-Y[ix,None,:]).square().mean((0,2)).sum();opt.zero_grad();loss.backward();opt.step();lossacc+=float(loss)/count
 if epoch==1 or epoch%50==0 or epoch==a.epochs:
  met=metrics();emit({'event':'train','epoch':epoch,'mse':lossacc/((len(X)+2047)//2048),**met});torch.save({'W':W.detach(),'b':b.detach(),'inv':inv,'mean':mean,'epoch':epoch,'metrics':met},out/'latest.pt')
  if met['rows_4_inf02']==len(free) and met['rows_4_cos99']==len(free):break
emit({'event':'done','epoch':epoch,**metrics()})
