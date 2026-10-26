import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
import kink_solve as K
torch.set_num_threads(2)
r=torch.load('recon/_precision_L5.pt',map_location='cpu',weights_only=False)
m=MLP(r['dims'],act=r['act']).double().cuda();m.load_state_dict(r['state_dict']);m.requires_grad_(False)
@torch.no_grad()
def invert(net,h,depth):
 z=h
 for l in reversed(range(depth)):
  z=torch.where(z>=0,z,z/.01)-net.layers[l].bias
  w=net.layers[l].weight
  if w.shape[0]==w.shape[1]:z=torch.linalg.solve(w,z.T).T
  else:z=torch.linalg.lstsq(w,z.T).solution.T
 return z
with torch.no_grad():
 torch.manual_seed(425)
 for scale in [.1,1.,10.]:
  h=torch.rand(256,200,device='cuda',dtype=torch.float64)*scale
  x=invert(m,h,5);actual=K._phi(m,x,5)
  e=(actual-h).abs()
  print('INVERSE','scale',scale,'x_rms',float(x.square().mean().sqrt()),'error_max',float(e.max()),'error_median',float(e.median()),flush=True)
 torch.save(dict(X=x.cpu(),H=actual.cpu(),target=h.cpu()),'recon/_inverse_probe.pt')
