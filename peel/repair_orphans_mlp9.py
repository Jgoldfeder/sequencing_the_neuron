"""Try actual committee rows in a canonical gauge; teacher used only forward and scoring."""
import sys,ast,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import align_clone_to,scale_normalize_,param_errors
from peel.informative_kinks import recover_layer
from peel.row_geometry import row_angle,input_scales
torch.set_num_threads(2)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
r=torch.load('recon/_local_mlp9_L6.pt.L6.partial.pt',map_location='cpu',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);m.requires_grad_(False);return m
T=model(d['teacher_state']);net=model(r['state_dict']);l=5;ok=r['refined_mask'].cuda();wr=r['refined_weight'].cuda();br=r['refined_bias'].cuda()
source=ast.parse(Path('refine_snapshot_mlp.py').read_text());names=['rows','orphans','distinct_candidate','install'];ns={'torch':torch,'net':net}
exec(compile(ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'helpers','exec'),ns)
with torch.no_grad():ns['install'](l,ok.nonzero().flatten(),wr,br)
scale_normalize_(net);nz=net.clone();mems=[model(s) for s in d['pop_states']];alt=[align_clone_to(m,nz)[1] for m in mems]
ns.update(nz=nz,alt=alt,col_s=[input_scales(net,i) for i in range(len(net.layers)-1)])
wr=net.layers[l].weight.detach().clone();br=net.layers[l].bias.detach().clone();left=(~ok).nonzero().flatten().tolist()
candidates=ns['orphans'](l,wr,br,ok,cos=.98)[:18]
print('CANDIDATES',len(candidates),[c[0] for c in candidates],flush=True)
q=0
for j in left:
 oldw=net.layers[l].weight[j].clone();oldb=net.layers[l].bias[j].clone()
 for ci,(support,row) in enumerate(candidates):
  with torch.no_grad():net.layers[l].weight[j]=row[:-1];net.layers[l].bias[j]=row[-1]
  diagnostics={};proposals={}
  w,b,mask,n=recover_layer(T,net,l,'cuda',only_channels=[j],max_rounds=16,diagnostics=diagnostics,candidate_rows=proposals,verbose=False);q+=n
  print('TRY',j,ci,'support',support,'q',n,'stats',diagnostics,flush=True)
  if bool(mask[j]) and ns['distinct_candidate'](j,w[j],b[j],wr,br,ok):
   wr[j]=w[j];br[j]=b[j];ok[j]=True
   with torch.no_grad():ns['install'](l,torch.tensor([j],device='cuda'),wr,br)
   print('ACCEPT',j,ci,'solved',int(ok.sum()),'q',q,flush=True);break
 else:
  with torch.no_grad():net.layers[l].weight[j]=oldw;net.layers[l].bias[j]=oldb
 torch.save(dict(dims=d['dims'],act=d['act'],state_dict=net.state_dict(),refined_mask=ok,queries=r['queries']+q,repair_queries=q,frontier=l), 'recon/_local_mlp9_L6_canonical_repair.pt')
print('RESULT',int(ok.sum()),'NEW_Q',q,'ERRORS',param_errors(net,T)['max_eps_per_matrix'],flush=True)
