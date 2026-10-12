"""Saved-committee query coverage audit; teacher weights only score coverage.
Subsampled independent queries retain production q=48000 gradient normalization.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import method
from nets import MLP
torch.set_num_threads(4)
device='cuda:1'
paths=['recon/_pop__mergedbest512_cheatpeel1__200x200x200x100__s0.pt','recon/_pop__mergedbest512_snapB__3072x200x200x200x100__s0_it005_prepeel.pt']
results=[]
original=method.disagreement
for path in paths:
 d=torch.load(path,map_location='cpu',weights_only=False)
 def model(s):
  m=MLP(d['dims'],act=d['act']).to(device);m.load_state_dict(s);m.requires_grad_(False);return m
 pop=[model(s) for s in d['pop_states']];teacher=model(d['teacher_state'])
 W=teacher.layers[0].weight;b=teacher.layers[0].bias
 def stats(x):
  with torch.no_grad():
   z=(x@W.T+b)/W.norm(dim=1)
   pos=(z>0).float().mean(0);near=(z.abs()<.1).float().mean(0)
   out=torch.stack([m(x) for m in pop]);err=((out-teacher(x))**2).mean((0,2))
   return dict(positive=pos.tolist(),near=near.tolist(),rms=x.square().mean().sqrt().item(),disagreement=-original(out,'mean_pair').item(),mse=err.mean().item(),minority_min=torch.minimum(pos,1-pos).min().item())
 entry=dict(path=path,iteration=d['iter'],runs=[])
 for seed in range(3):
  gen=torch.Generator(device=device).manual_seed(seed)
  x=torch.rand(2048,d['dims'][0],device=device,generator=gen)*2-1
  cfg=method.Cfg(q=2048,qg_init='uniform',qg_chunk=256)
  method.disagreement=lambda *a,**kw:original(*a,**kw)*(2048/48000)
  active=method.gen_queries(pop,cfg,d['dims'][0],device,gen,init_override=x)
  r=dict(seed=seed,random=stats(x),active=stats(active),wide_random=stats(x*2))
  entry['runs'].append(r)
  print(path,seed,[(k,{j:r[k][j] for j in ['rms','mse','minority_min']},[round(r[k]['positive'][i],4) for i in [122,139,199]]) for k in ['random','active','wide_random']],flush=True)
 results.append(entry)
Path('peel/sampler_stragglers_audit.json').write_text(json.dumps(results,indent=2))
