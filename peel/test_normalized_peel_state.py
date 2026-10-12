"""Check optimizer persistence, protected rows and rebuild after a new pin."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from peel.normalized_peel import PopulationStages,TrainingView

torch.set_num_threads(1);torch.manual_seed(8)
m=MLP([12,8,6,3]);pop=[m];opts=[torch.optim.Adam(m.parameters())]
masks={1:torch.tensor([True,False,False,False,False,False])}
def mask(m,l):return masks.get(l,torch.zeros(len(m.layers[l].bias),dtype=torch.bool))
manager=PopulationStages();stages=manager.prepare(pop,1,opts,mask,3);stage=stages[0]
old=m.layers[1].weight[0].clone();prefix=m.layers[0].weight.clone()
x=torch.randn(32,12);y=torch.randn(32,3)
view=TrainingView(stage);stage.optimizer.zero_grad();(view(x)-y).square().mean().backward();stage.optimizer.step()
manager.write_back(pop,stages,mask)
assert torch.equal(m.layers[1].weight[0],old)
assert torch.equal(m.layers[0].weight,prefix)
assert manager.prepare(pop,1,opts,mask,3)[0] is stage
assert len(stage.optimizer.state)>0
masks[1][1]=True
assert manager.prepare(pop,1,opts,mask,3)[0] is not stage
print('PASS optimizer persistence, solved rows/prefix unchanged, rebuild on new pin')
