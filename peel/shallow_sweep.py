"""Forward-only shallow-net refiner with isolated affine Jacobian sweeps.

The guess's right inverse places other guessed neurons away from their kinks.
Coordinate sweeps recover a rank-one jump without storing kink point clouds or
factoring an input-dimension-square matrix. Restricted to one hidden layer.
"""
import copy,time,math
import torch
from threadpoolctl import threadpool_limits
from verify_layer1 import _Oracle

@torch.no_grad()
@threadpool_limits.wrap(limits=1)
def recover_layer(teacher,cons,device,only_channels=None,gen=None,verbose=True,diagnostics=None,attempts=6,angle_gate=12.):
    if len(cons.layers)!=2:raise ValueError('shallow sweep requires one hidden layer')
    original_dtype=cons.layers[0].weight.dtype
    model=copy.deepcopy(cons).double().to(device)
    teacher=copy.deepcopy(teacher).double().to(device) if isinstance(teacher,torch.nn.Module) else teacher
    oracle=_Oracle(teacher);gen=gen or torch.Generator(device=device).manual_seed(0)
    W=model.layers[0].weight.detach();b=model.layers[0].bias.detach()
    norm=W.norm(dim=1);W=W/norm[:,None];b=b/norm
    m,d=W.shape;channels=list(range(m)) if only_channels is None else list(only_channels)
    outputW=W.clone();outputb=b.clone();mask=torch.zeros(m,device=device,dtype=torch.bool)
    if not channels or attempts == 0:
        return cons.layers[0].weight.detach().clone(),cons.layers[0].bias.detach().clone(),mask,0
    start=time.perf_counter()
    P=torch.linalg.lstsq(W.cpu(),torch.eye(m,dtype=torch.float64),driver='gelsd').solution.to(device)
    if (W@P-torch.eye(m,device=device)).abs().max()>1e-8:
        if verbose:print('[shallow-sweep] guess rows not independent; abstaining',flush=True)
        return cons.layers[0].weight.detach().clone(),cons.layers[0].bias.detach().clone(),mask,0
    def sweep(x,step):
        y0=oracle(x[None])[0];J=[]
        for first in range(0,d,256):
            idx=torch.arange(first,min(first+256,d),device=device)
            X=x[None].repeat(len(idx),1);X[torch.arange(len(idx),device=device),idx]+=step
            J.append((oracle(X)-y0)/step)
        return torch.cat(J),y0
    for c in channels:
        estimates=[];records=[];q0=oracle.n
        for attempt in range(attempts):
            z=(torch.randint(0,2,(m,),device=device,generator=gen)*2-1).double()*2
            z[c]=0;x=P@(z-b);u=P[:,c]
            step=2.0 if attempt%2==0 else 1.0
            xl=x-u;xr=x+u
            JL,fL=sweep(xl,step);JR,fR=sweep(xr,step)
            jump=(JR-JL).cpu();U,S,V=torch.linalg.svd(jump,full_matrices=False)
            signal_floor=1000*torch.finfo(W.dtype).eps*max(float(JL.norm()),float(JR.norm()),1e-300)
            if not torch.isfinite(S).all() or float(S[0]) <= signal_floor:
                records.append(dict(reason='no_resolved_jump'))
                continue
            rank_error=float(S[1:].norm()/S[0].clamp_min(1e-300))
            w=U[:,0].to(device)
            if w@W[c]<0:w=-w
            angle=math.degrees(math.acos(min(1.,float(w@W[c]))))
            # Intersect independently queried side lines along u for kink offset.
            Y=oracle(torch.stack([x-1.5*u,x-.5*u,x+.5*u,x+1.5*u]))
            left=Y[1]-Y[0];right=Y[3]-Y[2];delta=right-left
            aleft=Y[1]+.5*left;aright=Y[2]-.5*right
            t=float(((aleft-aright)*delta).sum()/delta.square().sum().clamp_min(1e-300))
            bias=-w@(x+t*u)
            records.append(dict(rank_error=rank_error,angle=angle,offset=t))
            if rank_error>1e-9 or angle>angle_gate or abs(t)>.4:continue
            estimates.append(torch.cat([w,bias[None]]))
            if len(estimates)>=2:
                disagreement=float((estimates[-1]-estimates[-2]).abs().max())
                records[-1]["independent_disagreement"]=disagreement
                if disagreement<1e-10:
                    v=torch.stack(estimates[-2:]).mean(0);v/=v[:-1].norm()
                    outputW[c]=v[:-1];outputb[c]=v[-1];mask[c]=True;break
        if diagnostics is not None:diagnostics[c]=dict(accepted=bool(mask[c]),queries=oracle.n-q0,attempts=records)
        if verbose:print(f'[shallow-sweep] c={c} accepted={bool(mask[c])} queries={oracle.n-q0} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    # Preserve guesses exactly when abstaining. Accepted rows use unit [w,b] gauge.
    n=(outputW.square().sum(1)+outputb.square()).sqrt()
    outputW[mask]/=n[mask,None];outputb[mask]/=n[mask]
    outputW[~mask]=cons.layers[0].weight[~mask].double();outputb[~mask]=cons.layers[0].bias[~mask].double()
    return outputW.to(original_dtype),outputb.to(original_dtype),mask,oracle.n
