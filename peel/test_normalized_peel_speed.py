"""Cached training and shared-prefix sampling preserve the original computation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from peel.normalized_peel import NormalizedPeel,TrainingView
torch.set_num_threads(1);torch.manual_seed(17)
pop=[MLP([24,12,8,3]) for _ in range(3)]
for m in pop[1:]:m.layers[0].load_state_dict(pop[0].layers[0].state_dict())
stages=[NormalizedPeel(m,1,calibration=64,seed=3) for m in pop]
a=stages[0];assert a.same_features(stages[1])
x=torch.randn(25,24,dtype=torch.float64,requires_grad=True)
y=torch.stack([s.predict(x) for s in stages]);g=torch.autograd.grad(y.square().sum(),x)[0]
z=a.features(x);y2=torch.stack([s.predict_features(z) for s in stages]);g2=torch.autograd.grad(y2.square().sum(),x)[0]
torch.testing.assert_close(y,y2,rtol=0,atol=0);torch.testing.assert_close(g,g2,rtol=1e-12,atol=1e-12)
uncached=TrainingView(a);cached=TrainingView(a,cached=True)
loss=uncached(x).square().sum();grad=torch.autograd.grad(loss,list(a.tail.parameters()))
loss2=cached(a.features(x).detach().float()).square().sum();grad2=torch.autograd.grad(loss2,list(a.tail.parameters()))
for u,v in zip(grad,grad2):torch.testing.assert_close(u,v,rtol=0,atol=0)
# Reference original two-evaluations-per-step sampler, with exactly same RNG.
state=a.gen.get_state();count=16;steps=5;lr=.05
z=a.randn(count,a.width).requires_grad_(True);residual=a.randn(count,a.din);opt=torch.optim.Adam([z],lr=lr)
def score(x):return torch.stack([s.predict(x) for s in stages]).var(0,unbiased=False).mean(1)
with torch.no_grad():
 start=a.propose(z,residual);best=score(start);best_x=start.clone()
for _ in range(steps):
 opt.zero_grad();s=score(a.propose(z,residual));z.grad=torch.autograd.grad(-s.mean(),z)[0];opt.step()
 with torch.no_grad():
  z.clamp_(-3,3);xx=a.propose(z,residual);ss=score(xx);mask=ss>best;best[mask]=ss[mask];best_x[mask]=xx[mask]
a.gen.set_state(state)
actual,_=a.sample([s.predict for s in stages[1:]],count=count,steps=steps,lr=lr)
torch.testing.assert_close(actual,best_x,rtol=1e-9,atol=1e-9)
with torch.no_grad():stages[1].base.layers[0].weight[0,0]+=1
assert not a.same_features(stages[1])
a.sample([s.predict for s in stages[1:]],count=4,steps=1)
print('PASS predictions, query gradients, cached training gradients, original sampler equivalence, unequal-prefix fallback')
