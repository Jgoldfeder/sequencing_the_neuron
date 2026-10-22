"""Build search-only known hyperplane bank. Does NOT claim signs or a functional recovery."""
import sys,torch,json
r=torch.load(sys.argv[1],map_location='cpu',weights_only=False);r['partial_state']={k:v.clone() for k,v in r['partial_state'].items()};r['partial_mask']={k:v.clone() for k,v in r['partial_mask'].items()}
mask=r['partial_mask'][3];W=r['partial_state']['layers.3.weight'];b=r['partial_state']['layers.3.bias'];added=[];anchors=dict(r.get('unsigned_anchors',{}))
for path in sys.argv[3:]:
 z=torch.load(path,map_location='cpu',weights_only=False);source=torch.load(z['source'],map_location='cpu',weights_only=False)
 for f in z['results']:
  if not f['quality'] or f['independent_kink_hits']<30:continue
  row=f['row'];known=torch.cat([W.flatten(1)[mask],b[mask,None]],1);known/=known.norm(dim=1,keepdim=True)
  if bool(((known@row).abs()>.99999).any()):continue
  free=(~mask).nonzero().flatten()
  if not len(free):break
  j=int(free[0]);W[j]=row[:-1].view_as(W[j]);b[j]=row[-1];mask[j]=True;added.append(j);anchors[j]=source['hyperplanes'][f['source_index']]['xstar'].clone()
r['unsigned_anchors']=anchors;r['unsigned_search_only']=True;r['unsigned_slots']=r.get('unsigned_slots',[])+added;r['state_dict']=r['partial_state'];r['note']='SEARCH ONLY: newly added rows have arbitrary sign; not a signed extraction checkpoint.'
torch.save(r,sys.argv[2]);print(json.dumps({'known_hyperplanes':int(mask.sum()),'new_slots':added,'all_unsigned_slots':r['unsigned_slots']}))
