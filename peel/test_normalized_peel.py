"""Bounded benchmark of reachable normalized query search and peeled Adam training."""
import argparse,copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from peel.normalized_peel import NormalizedPeel
p=argparse.ArgumentParser();p.add_argument('--arch',default='128,32,32,32,8');p.add_argument('--frontier',type=int,default=1);p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=0);p.add_argument('--rounds',type=int,default=3);p.add_argument('--output',default='peel/normalized_peel_results.json');a=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(a.seed);dims=list(map(int,a.arch.split(',')))
t=MLP(dims).double().to(a.device);g=copy.deepcopy(t)
with torch.no_grad():
 for l in g.layers[a.frontier:]:
  l.weight.add_(.03*torch.randn_like(l.weight));l.bias.add_(.03*torch.randn_like(l.bias))
# Fixture construction uses teacher weights; optimizer below sees only labels.
method=NormalizedPeel(g,a.frontier,calibration=max(1024,4*dims[a.frontier]),seed=a.seed)
peer=copy.deepcopy(g)
with torch.no_grad():
 for l in peer.layers[a.frontier:]:l.weight.add_(.02*torch.randn_like(l.weight))
peer.requires_grad_(False)
xval=4*torch.tanh(torch.randn(512,dims[0],device=a.device,dtype=torch.float64)/4)
with torch.no_grad():
 yval=t(xval)
 preservation=float((method.predict(xval).double()-g(xval)).abs().max())
 assert preservation<1e-5,preservation
 before=float((method.predict(xval)-yval).square().mean())
rows=[];xs=[];ys=[];start=time.perf_counter()
for i in range(a.rounds):
 x,stats=method.sample([peer],count=256,steps=30)
 with torch.no_grad():y=t(x)
 xs.append(x);ys.append(y)
 stats.update(method.fit(torch.cat(xs),torch.cat(ys),epochs=30))
 with torch.no_grad():stats['validation_mse']=float((method.predict(xval)-yval).square().mean())
 assert stats['disagreement_after']>=stats['disagreement_before']-1e-12
 assert stats['normalized_hidden_movement']>1e-5
 assert stats['training_mse_after']<stats['training_mse_before']
 rows.append(stats);print(i,stats,flush=True)
with torch.no_grad():
 exported=method.export();export_error=float((exported(xval)-method.predict(xval)).abs().max())
assert export_error<1e-5,export_error
assert rows[-1]['validation_mse']<before,(before,rows[-1])
assert all(torch.equal(exported.layers[i].weight,g.layers[i].weight) for i in range(a.frontier))
result=dict(config=vars(a),calibration=method.calibration_stats,initial_validation_mse=before,preservation_max=preservation,export_max=export_error,seconds=time.perf_counter()-start,teacher_training_queries=256*a.rounds,teacher_validation_queries=512,rounds=rows)
Path(a.output).write_text(json.dumps(result,indent=2));torch.save(dict(student=exported.state_dict(),dims=dims),str(Path(a.output).with_suffix('.pt')))
print('PASS',a.output,flush=True)
