"""Optional float32-prior cleanup with sparse kink constraints; forward only."""
import copy
import torch
from threadpoolctl import threadpool_limits
import kink_solve as K
from verify_layer1 import _Oracle
from peel.float32_lattice import recover_row

@torch.no_grad()
@threadpool_limits.wrap(limits=1)
def sparse_fit(h,w,b,indices):
    device=h.device;h=h.cpu();w=w.cpu();b=b.cpu()
    ids=torch.tensor(indices,dtype=torch.long)
    A=torch.cat([h[:,ids],torch.ones(len(h),1,dtype=h.dtype)],1)
    target=-(h@w+b)
    hold=torch.arange(len(h))%5==0
    keep=~hold
    # Initial close plane screens foreign surfaces before regression.
    gate=max(float(target[keep].abs().median())*50,1e-11)
    keep &= target.abs()<=gate
    if int(keep.sum())<len(ids)+12:return None
    best=None
    for _ in range(8):
        sol=torch.linalg.lstsq(A[keep],target[keep],driver='gelsd').solution
        res=(A@sol-target).abs();med=float(res[keep].median())
        if best is None or med<best[0]:best=(med,sol.clone())
        updated=(~hold)&(res<max(20*med,1e-12))
        if int(updated.sum())<len(ids)+12:break
        keep=updated
    sol=best[1];v=w.clone();v[ids]+=sol[:-1];bias=b+sol[-1]
    v=v.float().double();bias=bias.float().double()
    res=(h@v+bias).abs()/(v.norm()*(1+h.norm(dim=1)))
    fraction=float((res[hold]<1e-10).double().mean())
    return v.to(device),bias.to(device),dict(validation=fraction,residual=float(res[hold].median()),uncertain=len(ids))

@torch.no_grad()
@threadpool_limits.wrap(limits=1)
def cleanup_layer(oracle,model,l,points=None,channels=None,seed=82,frozen_mask=None):
    net=copy.deepcopy(model).double();dev=next(net.parameters()).device
    channels=list(range(net.layers[l].out_features)) if channels is None else channels
    gen=torch.Generator(device=dev).manual_seed(seed+l);bb=_Oracle(oracle)
    accepted=torch.zeros(net.layers[l].out_features,dtype=torch.bool,device=dev)
    archive={};stats={}
    for c in channels:
        oldw=net.layers[l].weight[c].clone();oldb=net.layers[l].bias[c].clone()
        raw=None;h=None;w=oldw;b=oldb
        if points is not None and c in points:
            raw=points[c].to(device=dev,dtype=torch.float64);h=K._phi(net,raw,l)
            if len(h)>=len(w)+16:
                try:w,b,_,_=K.solve_from_points(h,w,rounds=32,drop=.04)
                except torch.linalg.LinAlgError:pass
        # Lattice test operates at unit weight norm for a gauge-independent tolerance.
        norm=w.norm();w=w/norm;b=b/norm
        row=None
        for tolerance in (3e-12,1e-11,3e-11,1e-10):
            row,info=recover_row(w,atol=tolerance)
            if row is not None:break
        if row is None:
            stats[c]=info;print('[precision]',l+1,c,'no lattice',info.get('reason'),flush=True);continue
        bias=b*info['scale'];ids=info['uncertain'];need=max(96,2*(len(ids)+16))
        if h is None:
            if l!=0:raise ValueError('Deep cleanup requires previously collected raw kink points')
            samples=[]
            direction=row/row.norm()
            for rnd in range(12):
                x=torch.randn(128,len(row),device=dev,dtype=torch.float64,generator=gen)
                x-=((x@row+bias)/row.square().sum())[:,None]*row
                u=direction[None].expand_as(x);radius=torch.full((len(x),),(.01,.03,.1,.003)[rnd%4],device=dev,dtype=x.dtype)
                xs,ok=K.locate_light(bb,x,u,radius)
                ok &= (xs@row+bias).abs()<1e-6
                if ok.any():samples.append(xs[ok])
                if sum(len(v) for v in samples)>=need:break
            if not samples:stats[c]=dict(reason='no_points');continue
            raw=torch.cat(samples);h=raw
        archive[c]=raw.cpu()
        result=sparse_fit(h,row,bias,ids)
        if result is None:stats[c]=dict(reason='insufficient_inliers');continue
        wr,br,quality=result;stats[c]=dict(lattice=info,**quality)
        if quality['validation']<.9:continue
        solved=accepted.clone()
        if frozen_mask is not None:solved|=frozen_mask.to(device=dev)
        solved[c]=False
        if solved.any():
            bank=torch.cat([net.layers[l].weight[solved],net.layers[l].bias[solved,None]],1)
            bank=bank/bank.norm(dim=1,keepdim=True)
            row=torch.cat([wr,br.reshape(1)]);row=row/row.norm()
            if torch.minimum((bank-row).norm(dim=1),(bank+row).norm(dim=1)).min()<1e-8:
                stats[c]['reason']='duplicate_solved_row';continue
        factor=wr.norm()/oldw.norm()
        net.layers[l].weight[c]=wr;net.layers[l].bias[c]=br
        net.layers[l+1].weight[:,c]/=factor
        accepted[c]=True;archive[c]=raw.cpu()
        print('[precision]',l+1,c,'accepted','uncertain',len(ids),'residual',quality['residual'],'q',bb.n,flush=True)
    return net,accepted,bb.n,archive,stats
