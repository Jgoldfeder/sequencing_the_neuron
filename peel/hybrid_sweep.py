"""First-layer affine sweeps with downstream-kink rejection.

Forward-only teacher access. Uses a wide input / narrow first layer to construct
candidate crossings. Locates actual teacher kinks, shrinks coordinate sweeps
until their jump is rank one, then checks agreement across independent regions.
"""
import copy,math,time
import torch
from threadpoolctl import threadpool_limits
import kink_solve as K
from verify_layer1 import _Oracle

@torch.no_grad()
@threadpool_limits.wrap(limits=1)
def recover_first(teacher,cons,device,only_channels=None,gen=None,diagnostics=None,verbose=True,max_regions=8,angle_gate=12.):
    if max_regions < 0:raise ValueError("max_regions must be nonnegative")
    model=copy.deepcopy(cons).double().to(device).eval()
    oracle=_Oracle(teacher if not isinstance(teacher,torch.nn.Module) else copy.deepcopy(teacher).double().to(device).eval())
    gen=gen or torch.Generator(device=device).manual_seed(1234)
    W=model.layers[0].weight.detach();b=model.layers[0].bias.detach();norm=W.norm(dim=1);W=W/norm[:,None];b=b/norm
    m,d=W.shape;channels=list(range(m)) if only_channels is None else list(only_channels)
    Wr=cons.layers[0].weight.detach().double().to(device).clone();br=cons.layers[0].bias.detach().double().to(device).clone();mask=torch.zeros(m,device=device,dtype=torch.bool)
    if not channels or max_regions == 0:return Wr.to(cons.layers[0].weight.dtype),br.to(cons.layers[0].bias.dtype),mask,0
    start=time.perf_counter();P=torch.linalg.lstsq(W.cpu(),torch.eye(m,dtype=torch.float64),driver='gelsd').solution.to(device)
    if (W@P-torch.eye(m,device=device)).abs().max()>1e-8:
        return Wr.to(cons.layers[0].weight.dtype),br.to(cons.layers[0].bias.dtype),mask,0
    def sweep(x,step):
        base=oracle(x[None]);out=[]
        for first in range(0,d,256):
            idx=torch.arange(first,min(first+256,d),device=device);X=x[None].repeat(len(idx),1)
            X[torch.arange(len(idx),device=device),idx]+=step
            out.append((oracle(X)-base)/step)
        return torch.cat(out)
    for c in channels:
        q0=oracle.n;records=[];estimates=[]
        z=(2*torch.randint(0,2,(48,m),device=device,generator=gen)-1).double()*2;z[:,c]=0
        X=(z-b)@P.T;u=P[:,c];u=u/u.norm();U=u[None].expand_as(X);R=torch.full((len(X),),.3/float(W[c]@u),device=device,dtype=torch.float64)
        xs,ok=K.locate(oracle,X,U,R,gen,model,0,W[c],K=49)
        xs=xs[ok]
        # Favor predicted downstream margins, but teacher checks decide acceptance.
        if len(xs):
            a=xs;quality=torch.full((len(xs),),float('inf'),device=device)
            for l,L in enumerate(model.layers[:-1]):
                a=L(a)
                if l:quality=torch.minimum(quality,a.abs().amin(1))
                a=model.act(a)
            xs=xs[quality.argsort(descending=True)]
        for x in xs[:max_regions]:
            for step in [.5,.125,.03125,.0078125,.001953125]:
                e=.2*step
                JL=sweep(x-e*W[c],step);JR=sweep(x+e*W[c],step)
                jump=(JR-JL).cpu();A,S,V=torch.linalg.svd(jump,full_matrices=False)
                rank=float(S[1:].norm()/S[0].clamp_min(1e-300));w=A[:,0].to(device)
                if w@W[c]<0:w=-w
                angle=math.degrees(math.acos(min(1.,float(w@W[c]))))
                rec=dict(step=step,rank_error=rank,angle=angle);records.append(rec)
                if rank>1e-9 or angle>angle_gate or float(S[0])<1e-10:continue
                bias=-w@x
                # Estimate bias at the plane's low-norm foot, avoiding anchor-norm
                # amplification of the small normal-estimation error.
                foot=(-bias*w)[None]
                xf,found=K.locate(oracle,foot,w[None],torch.tensor([.01],device=device,dtype=torch.float64),gen,model,0,w,K=49,tol=1e-11)
                rec["bias_foot_found"]=bool(found[0])
                if bool(found[0]):bias=-w@xf[0]
                v=torch.cat([w,bias[None]]);estimates.append(v)
                if len(estimates)>=2:
                    diff=float((v-estimates[-2]).abs().max());rec['independent_difference']=diff
                    if diff<1e-10:
                        avg=(v+estimates[-2])/2;avg/=avg.norm();Wr[c]=avg[:-1];br[c]=avg[-1];mask[c]=True
                break
            if mask[c]:break
        if diagnostics is not None:diagnostics[c]=dict(accepted=bool(mask[c]),anchors=len(xs),queries=oracle.n-q0,records=records)
        if verbose:print(f'[hybrid] c={c} accepted={bool(mask[c])} queries={oracle.n-q0} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    return Wr.to(cons.layers[0].weight.dtype),br.to(cons.layers[0].bias.dtype),mask,oracle.n
