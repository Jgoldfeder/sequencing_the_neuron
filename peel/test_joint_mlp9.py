import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import MLP
from align import param_errors
from peel.joint_kink_polish import polish
torch.set_num_threads(2)
ap=argparse.ArgumentParser();ap.add_argument('--last',type=int,default=5);ap.add_argument('--iters',type=int,default=3);ap.add_argument('--cg',type=int,default=40);a=ap.parse_args()
r=torch.load(f'recon/_local_chain_L{a.last}.pt',map_location='cpu',weights_only=False)
d=torch.load('recon/_pop__mergedbest512_mlp9_nopeel_q200k__3072x200x200x200x200x200x200x200x200x100__s0.pt',map_location='cpu',weights_only=False)
def model(s):
 m=MLP(d['dims'],act=d['act']).double().cuda();m.load_state_dict(s);return m
teacher=model(d['teacher_state']);guess=model(r['state_dict'])
archives={l:torch.load(f'peel/local_chain_L{l+1}_points.pt',map_location='cpu',weights_only=False) for l in range(1,a.last)}
def score(m):
 e=param_errors(m,teacher);print('[joint-kinks] PREFIX_ERRORS',e['max_eps_per_matrix'][:2*a.last],flush=True);return e
out,info=polish(guess,archives,iters=a.iters,cg_iters=a.cg,score=score)
r['state_dict']=out.state_dict();r['joint_info']=info
torch.save(r,f'recon/_joint_local_L{a.last}.pt');score(out);print(info,flush=True)
