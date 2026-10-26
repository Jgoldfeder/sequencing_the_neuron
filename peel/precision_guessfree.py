"""Bounded missing-row discovery after precision correction of the prefix."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from peel.input_subspace import project_first_rowspace
from guessfree import recover_frontier_rows
from align import align_clone_to
torch.set_num_threads(2)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
r=torch.load('recon/_precision_L6.pt',map_location='cpu',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
t=model(d['teacher_state']);m=model(r['state_dict'])
oracle,net,basis=project_first_rowspace(t,m)
cache=torch.load('recon/_missing_residual_probe.pt',map_location='cpu',weights_only=False)
# Residual-based seeds use only cached forward queries, never true activations.
ids=cache['residual'].topk(64).indices
base=cache['X'][ids].cuda()@basis
archive={}
rows,q,info=recover_frontier_rows(oracle,net,5,r['mask'].nonzero().flatten().tolist(),lines=2,N=20001,T=1.,x0_std=.5,max_kinks=160,seed=91,base_points=base,archive=archive)
torch.save(dict(rows=rows,queries=q,info=info,archive=archive,basis=basis.cpu()),'recon/_precision_guessfree_L6.pt')
print('RESULT',len(rows),'QUERIES',q,info,flush=True)
# Score only after sampling and candidate acceptance have completed.
ref,truth=align_clone_to(t,m);target=torch.cat([truth.layers[5].weight[138],truth.layers[5].bias[138:139]]);target/=target.norm()
for i,c in enumerate(archive.get('candidates',[])):
 cos=float((c['row'].cuda()@target).abs())
 if cos>.8:print('DIAGNOSTIC_CANDIDATE',i,'cos',cos,flush=True)
