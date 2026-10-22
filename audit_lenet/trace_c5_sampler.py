"""Instrument the actual reconstruct_cnn sampler. No training, no refinement, no snapshot writes."""
import json,sys,contextlib,io
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import method
from nets import ConvNet

torch.set_num_threads(4);dev=torch.device('cuda:1')
r=torch.load('recon/_record_lenet_c1c2_c5_105.pt',map_location='cpu',weights_only=False)
t=ConvNet(r['input_shape'],r['conv_cfgs'],r['fc_dims'],r['out_dim'],r['act']).to(dev)
t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',map_location=dev,weights_only=True))
original_gen=method.gen_queries;original_boundary=method._cnn_frontier_boundary_queries
out=[]
for mode in ['default_boundary','disagreement','l1_default','l2_default']:
 stage_masks = ({0:r['partial_mask'][0]} if mode=='l2_default' else r['partial_mask'])
 trace={'mode':mode,'gen_queries_calls':0,'boundary_calls':0}
 def gen(*args,**kw):trace['gen_queries_calls']+=1;return original_gen(*args,**kw)
 def boundary(pop,fr,input_shape,cfg,device,generator,*args,**kw):
  trace['boundary_calls']+=1
  ids=torch.arange(cfg.q)%pop[0].layers[fr].weight.shape[0]
  pinned=stage_masks.get(fr,torch.zeros(pop[0].layers[fr].weight.shape[0],dtype=torch.bool))
  trace['frontier']=fr;trace['known_row_targets']=int(pinned[ids].sum());trace['unknown_row_targets']=int((~pinned[ids]).sum());trace['total_targets']=len(ids)
  result=original_boundary(pop,fr,input_shape,cfg,device,generator,*args,**kw)
  trace['boundary_returns_none']=result is None
  return result
 method.gen_queries=gen;method._cnn_frontier_boundary_queries=boundary
 cfg=method.Cfg(p=8,q=960,outer=1,epochs=0,balance=0,act='leaky_relu',layer_space_init=(mode!='disagreement'),qg_steps=3,log_every=1)
 torch.manual_seed(871)
 buf=io.StringIO()
 with contextlib.redirect_stdout(buf):
  _,_,result=method.reconstruct_cnn(t,r['input_shape'],r['conv_cfgs'],r['out_dim'],cfg,dev,fc_dims=r['fc_dims'],resume_state=(None if mode=='l1_default' else r['state_dict']),resume_record=(None if mode=='l1_default' else (r['partial_state'],stage_masks)))
 trace['queries']=result['queries'];out.append(trace)
 Path(f'audit_lenet/c5_sampler_trace_{mode}.log').write_text(buf.getvalue())
method.gen_queries=original_gen;method._cnn_frontier_boundary_queries=original_boundary
Path('audit_lenet/c5_sampler_trace.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
assert out[0]['gen_queries_calls']==0 and out[0]['known_row_targets']==824
assert out[1]['gen_queries_calls']==1

assert out[2]['gen_queries_calls']==1 and out[2]['boundary_calls']==0
assert out[3]['gen_queries_calls']==1 and out[3]['boundary_returns_none']
print('PASS: the default production path switches from disagreement at L1/L2 to boundary-only at C5')
