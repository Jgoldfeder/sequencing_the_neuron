"""Paired L5 learning ablation: identical inputs/init/batches, different labels.
True-tail access is used only as a diagnostic control, not in production.
"""
import argparse,copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from method import _mlp_prefix_inverse
from align import param_errors
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=120)
p.add_argument('--samples',type=int,default=16384);p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3])
p.add_argument('--output',default='peel/direct_vs_simulated.json');a=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(19);dev=a.device
full_dims=[3072]+[200]*7+[100];dims=full_dims[1:];frontier=4
full=MLP(full_dims).double();full.load_state_dict(torch.load('teachers/teacher_'+'x'.join(map(str,full_dims))+'_e25_s1.pt',map_location='cpu',weights_only=True))
teacher=MLP(dims).double().to(dev);tail=MLP(dims[frontier:]).double().to(dev)
with torch.no_grad():
 for dst,src in zip(teacher.layers,full.layers[1:]):dst.weight.copy_(src.weight);dst.bias.copy_(src.bias)
 for dst,src in zip(tail.layers,teacher.layers[frontier:]):dst.weight.copy_(src.weight);dst.bias.copy_(src.bias)
frozen={i:(L.weight,L.bias,torch.ones(200,device=dev,dtype=torch.bool)) for i,L in enumerate(teacher.layers[:frontier])}
inverse=_mlp_prefix_inverse(frozen,frontier,'leaky_relu',dev)
gen=torch.Generator(device=dev).manual_seed(417)
H=2*torch.rand(a.samples+2048,200,device=dev,generator=gen)-1
with torch.no_grad():
 direct=[];simulated=[];errors=[];largest=0.
 for h in H.split(512):
  x=inverse(h.double());actual=x
  for L in teacher.layers[:frontier]:actual=teacher.act(L(actual))
  errors.append((actual-h).square().sum());largest=max(largest,float(x.abs().max()))
  direct.append(tail(h.double()).float());simulated.append(teacher(x).float())
 direct=torch.cat(direct);simulated=torch.cat(simulated)
 label_gap=float((direct-simulated).norm()/direct.norm());h_error=float(torch.stack(errors).sum().sqrt()/H.norm())
Hv=H[a.samples:];Dval=direct[a.samples:];Sval=simulated[a.samples:]
H=H[:a.samples];targets={'direct':direct[:a.samples],'simulated':simulated[:a.samples]}
report=dict(config=vars(a),frontier=frontier,teacher_seed=1,exact_prefix=True,input_distribution='uniform [-1,1]',label_relative_error=label_gap,hidden_relative_error=h_error,max_original_input=largest,runs=[])
print('ORACLE',json.dumps({k:v for k,v in report.items() if k not in ['config','runs']}),flush=True)
@torch.no_grad()
def evaluate(net,mode):
 y=net(Hv);own=Dval if mode=='direct' else Sval
 return dict(true_mae=float((y-Dval).abs().mean()),true_relative=float((y-Dval).norm()/Dval.norm()),own_mae=float((y-own).abs().mean()))
for seed in a.seeds:
 torch.manual_seed(seed);base=MLP(dims[frontier:]).to(dev)
 nets={mode:copy.deepcopy(base) for mode in targets};opts={mode:torch.optim.Adam(net.parameters(),lr=.001) for mode,net in nets.items()}
 bg=torch.Generator(device=dev).manual_seed(1000+seed)
 run=dict(seed=seed,initial=evaluate(base,'direct'),history=[]);start=time.perf_counter()
 for epoch in range(a.epochs):
  if epoch in [int(.6*a.epochs),int(.85*a.epochs)]:
   for opt in opts.values():
    for group in opt.param_groups:group['lr']/=10
  perm=torch.randperm(a.samples,device=dev,generator=bg)
  for idx in perm.split(512):
   # Same minibatch order and update count in both arms.
   for mode,net in nets.items():
    opt=opts[mode];opt.zero_grad(set_to_none=True)
    loss=(net(H[idx])-targets[mode][idx]).abs().mean();loss.backward();opt.step()
  if (epoch+1)%10==0 or epoch+1==a.epochs:
   row=dict(epoch=epoch+1,**{mode:evaluate(net,mode) for mode,net in nets.items()});run['history'].append(row)
   print('EPOCH',seed,json.dumps(row),flush=True)
 run['seconds']=time.perf_counter()-start
 run['parameters']={mode:param_errors(net.double(),tail) for mode,net in nets.items()}
 report['runs'].append(run);Path(a.output).write_text(json.dumps(report,indent=2))
print('SAVED',a.output,flush=True)
