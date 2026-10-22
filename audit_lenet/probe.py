"""Small CPU audit reproductions; no training or snapshot writes. Run from code directory."""
import copy, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nets import ConvNet
from align import cnn_canonicalize_, cnn_align_to_
import residual_seed as rs
import cnn_balance as cb
from method import _cnn_frontier_layer

torch.set_num_threads(4)
torch.manual_seed(20261021)
conv=[(1,6,5,1,2,2),(6,16,5,1,0,2),(16,120,5,1,0,0)]
def net(): return ConvNet((1,28,28),conv,(84,),10,'leaky_relu').double()
t=net(); t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',weights_only=True)); cnn_canonicalize_(t)
for p in t.parameters(): p.requires_grad_(False)
report={}
# Isolate fp32 rounding: same function and gauge, only the attacker's prefix/pinned rows rounded.
pinned=torch.arange(120)<103
for name,m in [('exact_teacher_prefix',t.clone()),('fp32_rounded_prefix',t.clone().float().double())]:
 for p in m.parameters(): p.requires_grad_(False)
 V,S,nq=rs.residual_subspace(t,m,2,pinned,n_points=8,gen=torch.Generator().manual_seed(43),verbose=False)
 report[name]={'rank':len(V),'threshold_rank':int((S>1e-4*S[0]).sum()),'queries':nq,'singular_values':S[:24].tolist()}
 print(name,report[name],flush=True)
# The projected residual span is not the full unknown-row span.
W=rs.frontier_rows(t,2); Q=torch.linalg.qr(W[pinned].T).Q; P=torch.eye(400,dtype=torch.double)-Q@Q.T
U=torch.linalg.svd(W[~pinned]@P,full_matrices=False).Vh[:17]
projected=(W[~pinned]@U.T)@U
cos=torch.nn.functional.cosine_similarity(W[~pinned],projected)
report['unknown_rows_vs_projected_span_cosines']=cos.tolist()
# Actual record, initial canonicalization changes raw pinned coefficients despite frozen mask.
r=torch.load('recon/_record_lenet_c1c2_c5_105.pt',weights_only=False,map_location='cpu')
m=net(); m.load_state_dict(r['partial_state']); cnn_canonicalize_(m)
pre=m.clone(); x=torch.randn(32,784,dtype=torch.double)*.5
cb.canonicalize_(m,x,layers=list(range(4)))
report['initial_balance_raw_pinned_change']={str(l):float((m.layers[l].weight[mask]-pre.layers[l].weight[mask]).abs().max()) for l,mask in r['partial_mask'].items()}
report['initial_balance_function_error']=float((m(x)-pre(x)).abs().max())
a=pre.clone(); b=m.clone(); cnn_canonicalize_(a); cnn_canonicalize_(b)
report['initial_balance_canonical_error']=max(float((pa-pb).abs().max()) for pa,pb in zip(a.parameters(),b.parameters()))
# Sampler consults frozen, whereas partial pinning consults partial_mask too.
report['sampler_frontier_empty_frozen']=_cnn_frontier_layer({},4) if {} else None
report['sampler_frontier_frozen_only_conv1']=_cnn_frontier_layer({0:(None,None,torch.ones(6,dtype=torch.bool))},4)
# Summarize every residual-rank observation in the best log.
s=Path('pool_logs/lenet_c5_resseed_noreinit.log').read_text()
report['logged_ranks']=[{'line':i,'pinned':int(k),'missing':120-int(k),'rank':int(rank)} for i,line in enumerate(s.splitlines(),1) for k,rank in re.findall(r'frontier L3: (\d+) pinned rows, .*residual rank (\d+)',line)]
report['record_counts']={str(l):int(mask.sum()) for l,mask in r['partial_mask'].items()}
Path('audit_lenet/probe_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('exact_teacher_prefix','fp32_rounded_prefix')},indent=2),flush=True)

# Match the actual 103-row record, whose missing set consists of difficult channels.
rnet=net(); rnet.load_state_dict(r['partial_state']); cnn_canonicalize_(rnet)
ta=t.clone(); cnn_align_to_(ta,rnet)
report['actual_record_canonical_errors']={str(l):max(float((ta.layers[l].weight[mask]-rnet.layers[l].weight[mask]).abs().max()),float((ta.layers[l].bias[mask]-rnet.layers[l].bias[mask]).abs().max())) for l,mask in r['partial_mask'].items()}
for name,m in [('actual_record_fp64',rnet),('actual_record_fp32_roundtrip',rnet.clone().float().double())]:
 for p in m.parameters(): p.requires_grad_(False)
 V,S,nq=rs.residual_subspace(t,m,2,r['partial_mask'][2],n_points=8,gen=torch.Generator().manual_seed(43),verbose=False)
 report[name]={'rank':len(V),'threshold_rank':int((S>1e-4*S[0]).sum()),'queries':nq,'singular_values':S[:24].tolist()}
 print(name,report[name],flush=True)
W=rs.frontier_rows(ta,2); mask=r['partial_mask'][2]; Q=torch.linalg.qr(W[mask].T).Q; P=torch.eye(400,dtype=torch.double)-Q@Q.T
U=torch.linalg.svd(W[~mask]@P,full_matrices=False).Vh[:int((~mask).sum())]
report['actual_missing_projected_span_cosines']=torch.nn.functional.cosine_similarity(W[~mask],(W[~mask]@U.T)@U).tolist()
Path('audit_lenet/probe_results.json').write_text(json.dumps(report,indent=2)+'\n')
print('actual_record_canonical_errors',report['actual_record_canonical_errors'],flush=True)
print('actual_missing_projected_span_cosines',report['actual_missing_projected_span_cosines'],flush=True)
# Simulate successfully pinning the most active remaining channel: dominant signal vanishes,
# exposing the fp32 noise floor to a threshold tied only to the largest singular value.
with torch.no_grad():
 x=torch.randn(512,784,dtype=torch.double,generator=torch.Generator().manual_seed(81))*.5
 h,_=rs.prefix(ta,x,2)
 z=h@rs.frontier_rows(ta,2).T+ta.layers[2].bias
 free=(~mask).nonzero().flatten(); added=int(free[(z[:,free]>0).double().mean(0).argmax()])
 promoted=rnet.clone(); promoted.layers[2].weight[added]=ta.layers[2].weight[added]; promoted.layers[2].bias[added]=ta.layers[2].bias[added]
 mask104=mask.clone(); mask104[added]=True
report['controlled_promoted_channel']=added
for name,m in [('promoted104_fp64',promoted),('promoted104_fp32_roundtrip',promoted.clone().float().double())]:
 for p in m.parameters(): p.requires_grad_(False)
 V,S,nq=rs.residual_subspace(t,m,2,mask104,n_points=8,gen=torch.Generator().manual_seed(43),verbose=False)
 report[name]={'rank':len(V),'threshold_rank':int((S>1e-4*S[0]).sum()),'queries':nq,'singular_values':S[:24].tolist()}
 print(name,report[name],flush=True)
Path('audit_lenet/probe_results.json').write_text(json.dumps(report,indent=2)+'\n')
assert report['record_counts'] == {'0':6,'1':16,'2':103}
assert max(report['actual_record_canonical_errors'].values()) < 3.2e-10
assert report['promoted104_fp64']['rank'] == 16
assert report['promoted104_fp32_roundtrip']['threshold_rank'] > 16
assert report['promoted104_fp32_roundtrip']['rank'] == 0  # repaired guard abstains
assert max(report['actual_missing_projected_span_cosines']) < .84
print('PASS: record accuracy, rank-contamination reproduction, and projected-span limitation',flush=True)
