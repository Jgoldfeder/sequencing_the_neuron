"""Compare C5 matching rules on attributable original disagreement snapshots. Scoring only."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scipy.optimize import linear_sum_assignment
from nets import ConvNet
from align import cnn_canonicalize_,cnn_align_to_
torch.set_num_threads(4)
conv=[(1,6,5,1,2,2),(6,16,5,1,0,2),(16,120,5,1,0,0)]
def net():return ConvNet((1,28,28),conv,(84,),10,'leaky_relu').double()
t=net();t.load_state_dict(torch.load('teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt',weights_only=True));cnn_canonicalize_(t)
paths=[Path('recon/_record_lenet_c1c2_c5_105.pt')]+[p for p in Path('recon').glob('_pop__mergedbest*') if any(tag in p.name for tag in ['lenet_c5_cheat_cnn','lenet_nopeel_q200k_c_cnn'])]
rows=[]
for path in paths:
 ck=torch.load(path,weights_only=False,map_location='cpu');m=net();m.load_state_dict(ck.get('state_dict',ck));cnn_canonicalize_(m)
 # Align prefix only using the existing complete routine, then undo only C5+ suffix permutations.
 orig=m.clone();perms=cnn_align_to_(m,t)
 # Prefix gauges/positions now agree. C5 matching is recomputed on its pre-permutation rows.
 inv=torch.argsort(perms[2]);W=m.layers[2].weight.flatten(1)[inv];b=m.layers[2].bias[inv]
 Tw=t.layers[2].weight.flatten(1);Tb=t.layers[2].bias
 R=torch.cat([W,b[:,None]],1);T=torch.cat([Tw,Tb[:,None]],1);C=torch.nn.functional.normalize(T,dim=1)@torch.nn.functional.normalize(R,dim=1).T
 rules={'old_abs_filter_L1':torch.cdist(Tw.abs(),W.abs(),p=1),'signed_augmented_L1':torch.cdist(T,R,p=1),'signed_cosine':1-C}
 result={'snapshot':str(path),'iteration':ck.get('iter'),'rules':{}}
 for name,cost in rules.items():
  _,jj=linear_sum_assignment(cost.detach().numpy());ii=torch.arange(120);jj=torch.as_tensor(jj);cos=C[ii,jj];err=(T-R[jj]).abs().amax(1)
  result['rules'][name]={'above_cos99':int((cos>.99).sum()),'above_cos95':int((cos>.95).sum()),'coefficient_error_under02':int((err<.02).sum()),'cosines_sorted':cos.sort().values.tolist(),'mean_cosine':float(cos.mean())}
 rows.append(result)
Path('audit_lenet/original_c5_matching_results.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:
 print(row['snapshot'],row['iteration'])
 for k,v in row['rules'].items(): print(k,{kk:vv for kk,vv in v.items() if kk!='cosines_sorted'})
