"""Polish the already recovered signed C5 prefix through output queries only."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
from align import cnn_canonicalize_,cnn_align_to_
import kink_solve as K
torch.set_num_threads(4);dev='cuda:0';r=torch.load('audit_lenet/unsigned_f6_round4.pt',weights_only=False,map_location=dev)
def net():return ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).to(dev).double()
n=net();n.load_state_dict(r['partial_state']);t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',weights_only=True,map_location=dev))
for m in [n,t]:
 for p in m.parameters():p.requires_grad_(False)
NQ=0
@torch.no_grad()
def bb(X):
 global NQ
 NQ+=len(X);return torch.cat([t(x) for x in X.split(512)])
g=torch.Generator(device=dev).manual_seed(22901)
_base_seed=K.seed_brackets
def fixed_width_seed(*args,**kwargs):
 kwargs['eps']=1e-4
 return _base_seed(*args,**kwargs)
K.seed_brackets=fixed_width_seed
out=K.polish_layer(bb,n,2,list(range(120)),g,eps0=1e-4,need=700,max_rounds=16,max_rows=12000,verbose=True,full=False,scales=(.5,1.,2.))
accepted=[];details=[]
for j,(w,b,H) in out.items():
 ok,stats=K._cnn_fit_quality(H,w,b);details.append({'slot':j,'quality':ok,**stats})
 if not ok:continue
 row=torch.cat([w,b[None]]);row/=row.norm()
 with torch.no_grad():n.layers[2].weight[j]=row[:-1].view_as(n.layers[2].weight[j]);n.layers[2].bias[j]=row[-1]
 accepted.append(j)
r['partial_state']={k:v.cpu() for k,v in n.state_dict().items()};r['state_dict']=r['partial_state'];r['frozen'][2]=(n.layers[2].weight.cpu(),n.layers[2].bias.cpu(),torch.ones(120,dtype=torch.bool));r['c5_polish_queries']=NQ
# Score only after acceptance decisions.
ns=n.clone();ts=t.clone();cnn_canonicalize_(ns);cnn_canonicalize_(ts);cnn_align_to_(ts,ns);errors={l:max(float((ns.layers[l].weight-ts.layers[l].weight).abs().max()),float((ns.layers[l].bias-ts.layers[l].bias).abs().max())) for l in range(3)}
torch.save(r,'audit_lenet/unsigned_f6_precise_prefix_fixed.pt');Path('audit_lenet/c5_prefix_polish_fixed.json').write_text(json.dumps({'accepted':accepted,'details':details,'queries':NQ,'prefix_errors_score_only':errors},indent=2));print('DONE',len(accepted),'C5 rows repolished; errors',errors,'queries',NQ,flush=True)
