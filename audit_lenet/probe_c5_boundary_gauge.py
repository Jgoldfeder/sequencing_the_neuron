"""Actual C5 sampler must be invariant to function-preserving prefix gauges."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
import method
from cnn_balance import scale_channels_
from residual_seed import prefix

torch.set_num_threads(4);torch.manual_seed(7721);dev=torch.device('cuda:1')
conv=[(1,6,5,1,2,2),(6,16,5,1,0,2),(16,120,5,1,0,0)]
t=ConvNet((1,28,28),conv,(84,),10,'leaky_relu').to(dev).double();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',weights_only=True,map_location=dev))
pop=[t.clone() for _ in range(8)]
for i,m in enumerate(pop):
 if i:
  for l in (0,1):scale_channels_(m,l,torch.exp(torch.randn(m.layers[l].weight.shape[0],device=dev,dtype=torch.double)*.9))
 for p in m.parameters():p.requires_grad_(False)
q=960;cfg=method.Cfg(q=q,p=8,qg_init_std=.5)
X=torch.randn(32,784,device=dev,dtype=torch.double)
with torch.no_grad():function_error=max(float((m(X)-t(X)).abs().max()) for m in pop)
# Sampler initializes fp32 image tensors; make the otherwise equivalent models fp32.
pop=[m.float() for m in pop]
I=method._cnn_frontier_boundary_queries(pop,2,(1,28,28),cfg,dev,torch.Generator(device=dev).manual_seed(191),invert_steps=250,invert_lr=.1)
idx=torch.arange(q,device=dev);neuron=idx%120;member=(idx//120)%8
res=[]
with torch.no_grad():
 for i,m in enumerate(pop):
  sel=member==i;h,_=prefix(m,I[sel],2);z=(h*m.layers[2].weight.flatten(1)[neuron[sel]]).sum(1)+m.layers[2].bias[neuron[sel]]
  res.append({'member':i,'median_abs_target_preact':float(z.abs().median()),'mean_abs_target_preact':float(z.abs().mean()),'fraction_under_1e3':float((z.abs()<1e-3).float().mean())})
report={'function_error_fp64':function_error,'targets':res}
Path('audit_lenet/c5_boundary_gauge_probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

assert function_error < 1e-10
assert all(item["fraction_under_1e3"] >= .99 for item in res)
print("PASS: C5 boundary sampler respects member input gauges")
