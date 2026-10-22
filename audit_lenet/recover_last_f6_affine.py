"""Output-only last-unit recovery, using known kink amplitudes and affine remainder.
Reuses the direct solver's setup/primitives; teacher grading remains separate.
"""
from pathlib import Path
_setup=Path('audit_lenet/recover_frontier_direct.py').read_text().split("print('start direct output-only extraction; pinned'")[0]
exec(_setup)
assert fr==3 and len(known)==83
# Amplitude of an absolute-value kink is independent of its unknown orientation.
head_cols=[];amp_checks=[]
for j in known:
 xx=r['unsigned_anchors'][j].to(dev).clone().requires_grad_(True);ww=n.layers[fr].weight[j].flatten();bias=n.layers[fr].bias[j]
 val=prefix(n,xx[None],fr)[0][0]@ww+bias;direction=torch.autograd.grad(val,xx)[0];direction/=direction.norm();xx=xx.detach();amp=None
 for dt in [2e-3,5e-4,1e-4,3e-5,1e-5,3e-6]:
  coord=torch.tensor([-2.,-1.5,-1.,1.,1.5,2.],device=dev,dtype=D)*dt;pts=xx+coord[:,None]*direction
  with torch.no_grad():
   hh,pre=prefix(n,pts,fr);zg=hh@n.layers[fr].weight.flatten(1).T+n.layers[fr].bias;others=[k for k in known if k!=j]
   patterns=torch.cat([pre>0,zg[:,others]>0],1)
   if bool((patterns!=patterns[0]).any()) or not bool(zg[0,j]*zg[-1,j]<0):continue
  yy=bb(pts);tl=coord[:3];tr=coord[3:];vl=tl-tl.mean();vr=tr-tr.mean();sl=(yy[:3]*vl[:,None]).sum(0)/vl.square().sum();sr=(yy[3:]*vr[:,None]).sum(0)/vr.square().sum()
  gl=(zg[2,j]-zg[0,j])/(coord[2]-coord[0]);amp=(sr-sl)/(.99*gl);break
 if amp is None:raise RuntimeError(f'No isolated kink amplitude for {j}')
 head_cols.append(amp);amp_checks.append({'slot':j,'step':dt,'amplitude_norm':float(amp.norm())})
AA=torch.stack(head_cols,1) # out x 83, independent of the hidden-row signs
print('known output kink amplitudes recovered',len(head_cols),'queries',NQ,flush=True)
# On this region the last unit is affine. Wrong orientations of known rows
# contribute linear terms, since phi(-z)-phi(z) = -(1+.01) z.
XX=torch.randn(4096,784,device=dev,dtype=D,generator=gen)*.5
with torch.no_grad():HH=torch.cat([prefix(n,q,fr)[0] for q in XX.split(256)]);YY=bb(XX)
WK=n.layers[fr].weight.flatten(1)[mask];BK=n.layers[fr].bias[mask];GG=HH@WK.T+BK;AK=torch.nn.functional.leaky_relu(GG,.01);design=torch.cat([HH,torch.ones(len(HH),1,device=dev,dtype=D)],1)
rem=YY-AK@AA.T;linear=torch.linalg.lstsq(design.cpu(),rem.cpu(),driver='gelsd').solution.to(dev);UU=linear[:-1].T
qknown=torch.linalg.qr(WK.T).Q;RP=UU-(UU@qknown)@qknown.T;uu,ss,vv=torch.linalg.svd(RP,full_matrices=False);outdir=uu[:,0];Pu=torch.eye(10,device=dev,dtype=D)-outdir[:,None]*outdir[None,:]
coeff=(Pu@UU)@torch.linalg.pinv(WK);headproj=Pu@AA;flip=-((coeff*headproj).sum(0))/(1.01*headproj.square().sum(0));binary=flip.round();fliperr=float((flip-binary).abs().max());print('orientation binary error',fliperr,'range',float(flip.min()),float(flip.max()),'residual singulars',ss.tolist(),flush=True)
assert fliperr<.01 and bool(((binary==0)|(binary==1)).all())
signs=1-2*binary
trueAK=torch.nn.functional.leaky_relu(GG*signs,.01);remaining=YY-trueAK@AA.T;fit=torch.linalg.lstsq(design.cpu(),remaining.cpu(),driver='gelsd').solution.to(dev);lu,ls,lv=torch.linalg.svd(fit[:-1].T,full_matrices=False);lastw=lv[0];print('last full direction singulars',ls.tolist(),'affine fit max error',float((design@fit-remaining).abs().max()),flush=True)
assert float(ls[1]/ls[0])<.01
# Save fitted decisions before scoring or searching on the fitted direction.
torch.save({'head_amplitudes':AA.cpu(),'known_slots':known,'known_signs':signs.cpu(),'last_weight_direction':lastw.cpu(),'orientation_binary_error':fliperr,'queries':NQ},f'audit_lenet/{a.tag}_decomposition.pt')
# Attacker-side optimization of the estimated last normal, both directions.
count=64;signvec=torch.cat([torch.ones(count//2,device=dev,dtype=D),-torch.ones(count//2,device=dev,dtype=D)])
x=(torch.rand(count,784,device=dev,dtype=D,generator=gen)*2-1).requires_grad_(True);opt=torch.optim.Adam([x],lr=.05)
for it in range(500):
 opt.zero_grad();obj=(prefix(n,x,fr)[0]@lastw)*signvec;(-obj.mean()).backward();opt.step()
 with torch.no_grad():x.clamp_(-4,4)
ends=x.detach()
for li in range(count//2):
 x0=(ends[li]+ends[li+count//2])/2;d=(ends[li]-ends[li+count//2])/2;unk,countk,step=G.scan_line(oracle,n,fr,x0,d,1,a.grid,known)
 print('targeted line',li,'unexplained',len(unk),'queries',NQ,flush=True)
 for rr in unk:
  loc=locate(x0,d,rr-float(step),rr+float(step))
  if loc is None:continue
  result=recover(x0,d,*loc)
  if result is None:continue
  row,xstar,ratio,err=result
  if bool(((Wk@row).abs()>.99999).any()):continue
  found.append({'row':row.cpu(),'xstar':xstar.cpu(),'verification_hits':0,'rank_ratio':ratio,'mapback_residual':err,'score_only':score(row)})
  print('LAST CANDIDATE grade only',found[-1]['score_only'],flush=True);save();break
 if found:break
save();print('DONE candidates',len(found),'queries',NQ,flush=True)
