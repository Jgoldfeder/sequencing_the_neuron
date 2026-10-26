"""Query-free joint prefix correction from archived kink constraints.

Teacher is not an argument. Initial recovered first layer is held fixed.
Only points passing a robust current-plane screen enter the least-squares fit.
"""
import copy
import torch
from gn_polish import gn_polish

class KinkConstraints(torch.nn.Module):
    def __init__(self,model,last):
        super().__init__()
        self.layers=torch.nn.ModuleList(copy.deepcopy(list(model.layers[1:last+1])))
        self.requires_grad_(True)
    def forward(self,x):
        h=x[:,:-2];layer=x[:,-2].long();channel=x[:,-1].long()
        result=torch.zeros(len(x),1,device=x.device,dtype=x.dtype)
        for j,mod in enumerate(self.layers,1):
            z=mod(h)
            row=channel.clamp(max=z.shape[1]-1)
            v=z.gather(1,row[:,None])
            scale=mod.weight.norm(dim=1)[row,None]*(1+h.norm(dim=1,keepdim=True))
            result=torch.where((layer==j)[:,None],v/scale.clamp_min(1e-30),result)
            h=torch.nn.functional.leaky_relu(z,.01)
        return result

@torch.no_grad()
def prepare(model,archives,per_row=256):
    """Return cached first-layer features plus layer/channel, using no oracle."""
    device=next(model.parameters()).device
    data=[];counts={}
    for l,a in archives.items():
        count=0
        for c,raw in a['points'].items():
            raw=raw.to(device=device,dtype=torch.float64)
            first=model.act(model.layers[0](raw));h=first
            for j in range(1,l):h=model.act(model.layers[j](h))
            w=model.layers[l].weight[c];b=model.layers[l].bias[c]
            res=(h@w+b).abs()/(w.norm()*(1+h.norm(dim=1)))
            # Never include a row whose majority of samples is not a kink of
            # the recovered plane. Keep up to 80% and cap per-row work.
            if float(res.median())>1e-7:continue
            n=min(per_row,int(.8*len(raw)))
            idx=res.argsort()[:n]
            keep=res[idx]<1e-7;idx=idx[keep]
            if len(idx)<model.layers[l].in_features+8:continue
            ids=torch.empty(len(idx),2,device=device,dtype=torch.float64)
            ids[:,0]=l;ids[:,1]=c
            data.append(torch.cat([first[idx],ids],1).cpu());count+=len(idx)
        counts[l]=count
    if not data:raise ValueError('No reliable kink constraints')
    return torch.cat(data),counts

def polish(model,archives,iters=3,cg_iters=40,chunk=8192,per_row=256,score=None):
    last=max(archives)
    x,counts=prepare(model,archives,per_row)
    print('[joint-kinks] constraints',counts,'queries=0',flush=True)
    constraints=KinkConstraints(model,last)
    def diagnostic(m):
        candidate=copy.deepcopy(model)
        for j,layer in enumerate(m.layers,1):candidate.layers[j].load_state_dict(layer.state_dict())
        return score(candidate)
    constraints,info=gn_polish(constraints,x,torch.zeros(len(x),1,dtype=x.dtype),
        iters=iters,cg_iters=cg_iters,chunk=chunk,device=next(model.parameters()).device,
        score=diagnostic if score else None,tag='[joint-kinks]')
    result=copy.deepcopy(model)
    for j,layer in enumerate(constraints.layers,1):result.layers[j].load_state_dict(layer.state_dict())
    info['constraints']=counts;info['queries']=0
    return result,info
