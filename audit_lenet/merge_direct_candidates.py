import sys,torch
r=torch.load('recon/_record_lenet_c1c2_c5_105.pt',map_location='cpu',weights_only=False);mask=r['partial_mask'][2];W=r['partial_state']['layers.2.weight'].flatten(1)[mask];Q=torch.linalg.qr(W.T).Q
found=[];stats=[]
for path in sys.argv[2:]:
 z=torch.load(path,map_location='cpu',weights_only=False);stats.extend(z['stats'])
 for f in z['hyperplanes']:
  row=f['row'];w=row[:-1]
  if float((w-Q@(Q.T@w)).norm()/w.norm())<.4:continue
  if found and bool(((torch.stack([ff['row'] for ff in found])@row).abs()>.99999).any()):continue
  found.append(f)
torch.save({'hyperplanes':found,'accepted_indices':[],'stats':stats},sys.argv[1]);print('merged candidates',len(found))
