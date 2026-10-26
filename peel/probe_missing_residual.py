"""Forward-only tail fit to expose the missing frontier neuron's residual.
Diagnostic experiment, never overwrites the recovered prefix checkpoint.
"""
import sys,copy,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import align_clone_to
from gn_polish import gn_polish
torch.set_num_threads(2);device='cuda'
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
r=torch.load('recon/_local_mlp9_L6_canonical_repair.pt',map_location='cpu',weights_only=False)
old=torch.load('recon/_local_mlp9_L6.pt.L6.partial.pt',map_location='cpu',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
T=model(d['teacher_state']);net=model(r['state_dict']);l=5;ok=r['refined_mask'].cuda();new=ok&~old['refined_mask'].cuda();missing=(~ok).nonzero().flatten()
gen=torch.Generator(device=device).manual_seed(447)
with torch.no_grad():
 scales=torch.tensor([.5,1.,2.,4.],device=device,dtype=torch.float64)
 X=torch.randn(24096,d['dims'][0],device=device,dtype=torch.float64,generator=gen)*scales[torch.arange(24096,device=device)%4,None]
 H=[];Y=[]
 for x in X.split(1024):
  Y.append(T(x));h=x
  for mod in net.layers[:l]:h=net.act(mod(h))
  H.append(h)
 H=torch.cat(H);Y=torch.cat(Y)
 print('ORACLE_QUERIES',len(X),flush=True)
 def rows(m):
  z=torch.cat([m.layers[l].weight,m.layers[l].bias[:,None]],1);return z/z.norm(dim=1,keepdim=True)
 ref=rows(net);candidates=[]
 for i,s in enumerate(d['pop_states']):
  _,aligned=align_clone_to(model(s),net)
  cos=(rows(aligned)*ref).sum(1)
  eligible=bool((cos[new]>.98).all())
  tail=MLP(d['dims'][l:],act=d['act']).double().cuda()
  for j,mod in enumerate(tail.layers):mod.load_state_dict(aligned.layers[l+j].state_dict())
  tail.layers[0].weight[ok]=net.layers[l].weight[ok];tail.layers[0].bias[ok]=net.layers[l].bias[ok]
  err=float((tail(H[-4096:])-Y[-4096:]).square().mean())
  print('MEMBER',i,'new-row cosine',cos[new].tolist(),'eligible',eligible,'MSE',err,flush=True)
  if eligible:candidates.append((err,i,tail))
 _,member,tail=min(candidates,key=lambda c:c[0]);print('SELECTED',member,flush=True)
 # Fit the output linear layer exactly before changing nonlinear tail weights.
 hh=H[:20000]
 for mod in tail.layers[:-1]:hh=tail.act(mod(hh))
 A=torch.cat([hh,torch.ones(len(hh),1,device=device,dtype=hh.dtype)],1)
 sol=torch.linalg.lstsq(A,Y[:20000]).solution
 tail.layers[-1].weight.copy_(sol[:-1].T);tail.layers[-1].bias.copy_(sol[-1])
tail.requires_grad_(True);tail.layers[0].requires_grad_(False)
print('VAL_BEFORE',float((tail(H[-4096:])-Y[-4096:]).square().mean()),flush=True)
tail,info=gn_polish(tail,H[:20000],Y[:20000],iters=4,cg_iters=40,chunk=4096,device=device)
with torch.no_grad():
 pred=torch.cat([tail(h) for h in H.split(2048)])
 residual=(pred-Y).square().mean(1)
 print('VAL_AFTER',float(residual[-4096:].mean()),flush=True)
 # Post-hoc diagnostic only: not used for initialization, sampling or fitting.
 h=X
 for mod in T.layers[:l]:h=T.act(mod(h))
 active=(T.layers[l](h)[:,172]>0)
 for frac in [.01,.05,.1]:
  idx=residual.topk(int(frac*len(residual))).indices
  print('DIAGNOSTIC_TOP',frac,'positive_fraction',float(active[idx].double().mean()),'base_rate',float(active.double().mean()),flush=True)
 torch.save(dict(X=X.cpu(),H=H.cpu(),Y=Y.cpu(),residual=residual.cpu(),tail_state=tail.state_dict(),member=member,queries=len(X),info=info), 'recon/_missing_residual_probe.pt')
