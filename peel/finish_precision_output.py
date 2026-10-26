"""Fit and independently verify the output after every hidden row is recovered."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from threadpoolctl import threadpool_limits
from nets import MLP
from align import param_errors
import kink_solve as K
torch.set_num_threads(2)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
r=torch.load('recon/_precision_L8_complete.pt',map_location='cpu',weights_only=False)
if not bool(r['mask'].all()):raise RuntimeError('Hidden layer incomplete')
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
t=model(d['teacher_state']);m=model(r['state_dict']);gen=torch.Generator(device='cuda').manual_seed(988)
@torch.no_grad()
def sample(n):
 x=torch.randn(n,3072,device='cuda',dtype=torch.float64,generator=gen)
 scale=torch.tensor([.25,.5,1.,2.,4.,8.],device='cuda',dtype=torch.float64)
 return x*scale[torch.arange(n,device='cuda')%len(scale),None]
with torch.no_grad(),threadpool_limits(limits=1):
 x=sample(4096);h=K._phi(m,x,8);y=t(x)
 A=torch.cat([h,torch.ones(len(h),1,device='cuda',dtype=h.dtype)],1).cpu();Y=y.cpu()
 scale=A.square().mean(0).sqrt().clamp_min(1e-30)
 sol=torch.linalg.lstsq(A/scale,Y,driver='gelsd').solution/scale[:,None]
 m.layers[-1].weight.copy_(sol[:-1].T);m.layers[-1].bias.copy_(sol[-1])
 x=sample(2048);truth=t(x);pred=m(x);error=pred-truth
 report=dict(queries=6144,max_output_error=float(error.abs().max()),relative_rms=float(error.square().mean().sqrt()/truth.square().mean().sqrt()),parameter_diagnostics=param_errors(m,t))
 print('OUTPUT_VALIDATION',report,flush=True)
 report['passed']=report['relative_rms']<1e-10
 torch.save(dict(dims=d['dims'],act=d['act'],state_dict=m.state_dict(),hidden_complete=True,validation=report,assumption='float32_teacher_parameters'), 'recon/_precision_mlp9_final.pt')
 if not report['passed']:raise RuntimeError('Fresh output validation failed; final checkpoint is not certified')
