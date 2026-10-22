"""Orient fully recovered C5 hyperplanes using output-only slope ratios.
A dual input direction moves one C5 preactivation and holds the other 119 fixed.
The two output slopes have ratio 100 or .01 for leaky slope .01.
Teacher weights are consulted ONLY after every orientation decision, for scoring.
"""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
from residual_seed import prefix,_prefix_jac
from align import cnn_canonicalize_,cnn_align_to_
torch.set_num_threads(4);dev='cuda:2';r=torch.load(sys.argv[1],map_location=dev,weights_only=False);fr=int(sys.argv[3]) if len(sys.argv)>3 else 2
def net():return ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).to(dev).double()
n=net();n.load_state_dict(r['partial_state'])
for l,(W,b,m) in r['frozen'].items():
 with torch.no_grad():n.layers[l].weight[m]=W[m];n.layers[l].bias[m]=b[m]
assert bool(r['partial_mask'][fr].all()),'Need all 120 hyperplanes for dual orientation'
for p in n.parameters():p.requires_grad_(False)
t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',weights_only=True,map_location=dev))
for p in t.parameters():p.requires_grad_(False)
NQ=0
@torch.no_grad()
def bb(x):
 global NQ
 NQ+=len(x);return t(x)
report=[]
for j in r['unsigned_slots']:
 x=r['unsigned_anchors'][j].clone().requires_grad_(True);w=n.layers[fr].weight[j].flatten();b=n.layers[fr].bias[j]
 for it in range(8):
  val=prefix(n,x[None],fr)[0][0]@w+b;gr=torch.autograd.grad(val,x)[0]
  with torch.no_grad():x.sub_(val*gr/gr.square().sum())
 x=x.detach();P=_prefix_jac(n,x,fr);A=n.layers[fr].weight.flatten(1)@P;target=torch.zeros(n.layers[fr].weight.shape[0],device=dev,dtype=torch.double);target[j]=1
 d=torch.linalg.pinv(A,rtol=1e-11)@target;d/=d.norm();actual=A@d;other=actual.clone();other[j]=0
 accepted=False
 for dt in [1e-3,3e-4,1e-4,3e-5,1e-5,3e-6]:
  ss=torch.tensor([-2,-1,1,2],device=dev,dtype=torch.double)*dt;xx=x+ss[:,None]*d;_,z=prefix(n,xx,fr)
  if bool(((z>0)!=(z[0]>0)).any()):continue
  Y=bb(xx);left=(Y[1]-Y[0])/dt;right=(Y[3]-Y[2])/dt;ratio=float(right.norm()/left.norm());cos=float(torch.nn.functional.cosine_similarity(left,right,dim=0))
  if cos>.999 and (90<ratio<110 or .009<ratio<.011):
   sign=1 if ratio>1 else -1;accepted=True;break
 if not accepted:raise RuntimeError(f'No reliable slope ratio for channel {j}: {ratio if "ratio" in locals() else None}')
 with torch.no_grad():n.layers[fr].weight[j].mul_(sign);n.layers[fr].bias[j].mul_(sign)
 row={'slot':j,'sign':sign,'right_over_left_slope_norm':ratio,'slope_cosine':cos,'max_other_frontier_directional_derivative':float(other.abs().max()),'moved_frontier_derivative':float(actual[j]),'step':dt};report.append(row);print(json.dumps(row),flush=True)
# Score ONLY after all signs have been selected from output queries.
ns=n.clone();ts=t.clone();cnn_canonicalize_(ns);cnn_canonicalize_(ts);cnn_align_to_(ts,ns)
errors={l:max(float((ns.layers[l].weight-ts.layers[l].weight).abs().max()),float((ns.layers[l].bias-ts.layers[l].bias).abs().max())) for l in range(fr+1)}
r['partial_state']={k:v.detach().cpu() for k,v in n.state_dict().items()};r['state_dict']=r['partial_state'];r['unsigned_search_only']=False;r['signed_c5_complete']=True;r['signed_frontier_complete']=fr;r['orientation_output_queries']=NQ;r['orientation_report']=report;r['prefix_errors_score_only']=errors;r['note']='C1/C2/C5 recovered; new C5 signs selected using output slopes, not teacher coefficients. F6/head remain learned guesses.'
r['frozen']={l:(n.layers[l].weight.detach().cpu(),n.layers[l].bias.detach().cpu(),torch.ones(n.layers[l].weight.shape[0],dtype=torch.bool)) for l in range(fr+1)}
torch.save(r,sys.argv[2]);Path(sys.argv[2]+'.json').write_text(json.dumps({'orientation':report,'queries':NQ,'prefix_errors_score_only':errors},indent=2));print('SIGNED PREFIX COMPLETE',errors,'queries',NQ,flush=True)
