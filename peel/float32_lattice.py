"""Optional float32-parameter prior: recover a row up to a power-of-two gauge.

No oracle or teacher weights are used here. Enumerate the mantissa of the
largest coordinate, constrain it using other coordinates, then check the full
row. This is inapplicable to arbitrary float64 parameter networks.
"""
import torch
@torch.no_grad()
def recover_row(w,atol=1e-10,probes=24):
    w=w.double();pivot=int(w.abs().argmax());vmax=w[pivot].abs()
    if vmax==0:return None,{'reason':'zero_row'}
    # Positive scales taking the largest absolute entry into [.5,1).
    scale=(.5+torch.arange(1<<23,device=w.device,dtype=torch.float64)*(2.**-24))/vmax
    order=w.abs().argsort(descending=True)
    history=[]
    for idx in order[1:probes+1]:
        x=scale*w[idx];err=(x.float().double()-x).abs()
        scale=scale[err<=scale*atol]
        history.append(len(scale))
        if len(scale)==0:return None,dict(reason='no_lattice_match',survivors=history)
        if len(scale)<16:break
    if len(scale)>1024:return None,dict(reason='ambiguous',survivors=history)
    candidate=(scale[:,None]*w[None]).float().double()
    # A pivot estimate has noise: compare after the best positive projection.
    fitted=(candidate@w)/(w@w)
    residual=(candidate/fitted[:,None]-w).abs().amax(1)
    good=residual<=atol
    if int(good.sum())!=1:return None,dict(reason='ambiguous_full_row',survivors=history,
                                           matches=int(good.sum()),best_residual=float(residual.min()))
    i=int(good.nonzero()[0])
    x=fitted[i]*w
    bound=fitted[i]*atol*(1+w.abs()/vmax)
    uncertain=((x-bound).float()!=(x+bound).float()).nonzero().flatten().tolist()
    return candidate[i],dict(scale=float(fitted[i]),residual=float(residual[i]),survivors=history,uncertain=uncertain)
