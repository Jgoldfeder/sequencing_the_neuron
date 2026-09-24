"""Paired timing of one design round, with/without scoped CPU thread limits."""
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from threadpoolctl import threadpool_info
from nets import MLP
from align import scale_normalize_
from peel import informative_kinks as I

def pools():
    return {p['filepath']:p['num_threads'] for p in threadpool_info()}

torch.manual_seed(0)
t=MLP([200]*7+[100]).double()
with torch.no_grad():
    for layer in t.layers: layer.bias.normal_(0,.1)
scale_normalize_(t); t=t.cuda(); g=t.clone()
with torch.no_grad():
    g.layers[0].weight.add_(torch.randn_like(g.layers[0].weight)*.01/200**.5)
    g.layers[0].bias.add_(torch.randn_like(g.layers[0].bias)*.005)
    t(torch.zeros(8,200,device='cuda',dtype=torch.float64))
original_pools=pools(); raw=I.recover_layer.__wrapped__.__wrapped__
results=[]
for label,fn in [('limited_warmup',I.recover_layer),('original',raw),('limited',I.recover_layer)]:
    diag={}; gen=torch.Generator(device='cuda').manual_seed(1)
    torch.cuda.synchronize(); start=time.perf_counter()
    with torch.no_grad():
        _,_,_,queries=fn(t,g,0,'cuda',only_channels=list(range(8)),gen=gen,max_rounds=1,verbose=False,diagnostics=diag)
    torch.cuda.synchronize()
    result=dict(mode=label,seconds=time.perf_counter()-start,queries=queries,points=[diag[c]['points'] for c in range(8)])
    assert pools()==original_pools,'Thread settings were not restored'
    results.append(result); print(json.dumps(result),flush=True)
assert results[1]['queries']==results[2]['queries'] and results[1]['points']==results[2]['points']
# Restoration must also happen if a teacher query fails.
def failing_teacher(x): raise RuntimeError('test teacher failure')
try:
    I.recover_layer(failing_teacher,g,0,'cuda',only_channels=[0],max_rounds=1,verbose=False)
except RuntimeError as e:
    assert str(e)=='test teacher failure'
else: raise AssertionError('Expected teacher failure')
assert pools()==original_pools
Path('peel/refinement_threads_check.json').write_text(json.dumps(dict(pools=original_pools,results=results),indent=2))
print('PASS: identical query/point counts; thread settings restored after success and failure',flush=True)
