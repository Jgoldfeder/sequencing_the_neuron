"""Bounded, forward-only local refinement from independently seeded kinks.

Unlike tracking, points start in independent input regions. Each attempted
bracket costs eight oracle inputs, plus eight for fingerprinting each hit. Raw kink coordinates are retained for refits.
Acceptance is a heuristic held-out check, not a machine-precision certificate.
"""
import copy, math, time
import torch
from threadpoolctl import threadpool_limits
import kink_solve as K
from peel.informative_kinks import candidate_pool, select_candidates
from verify_layer1 import _Oracle

@torch.no_grad()
def locate_target(oracle,model,layer,c,x,u,r,gen):
    xs,ok=K.locate_light(oracle,x,u,r)
    idx=ok.nonzero().flatten()
    if not len(idx):return xs,ok
    xx=xs[idx];uu=u[idx];rr=r[idx];n,d=xx.shape
    vs,dhv,dhu=K._probe_dirs(model,layer,xx,uu,model.layers[layer].weight[c],gen,n_probe=2)
    e=rr/1000;dl=rr/20000
    minus=xx-e[:,None]*uu;plus=xx+e[:,None]*uu
    directions=torch.cat([uu[:,None],vs],1)
    probes=torch.cat([minus[:,None],plus[:,None],minus[:,None]+dl[:,None,None]*directions,
                      plus[:,None]+dl[:,None,None]*directions],1)
    y=oracle(probes.reshape(-1,d)).reshape(n,8,-1)
    jump=(y[:,5:]-y[:,1,None]-y[:,2:5]+y[:,0,None])/dl[:,None,None]
    ju=jump[:,0];jv=jump[:,1:]
    ratio=(jv.norm(dim=2)/dhv.clamp_min(1e-30))/(ju.norm(dim=1)/dhu.clamp_min(1e-30))[:,None].clamp_min(1e-30)
    strength=ju.norm(dim=1)*rr
    valid=(ratio<.1).all(1)&(strength>1e-6*y.abs().amax(dim=(1,2)))
    ok[idx]=valid
    return xs,ok


@torch.no_grad()
@threadpool_limits.wrap(limits=1)
def recover_layer(teacher, guess, layer, channels=None, seed=123, need=None,
                  rounds=16, pool=384, batch=96, verbose=True, archive=None, initial_points=None, widths=None):
    model=copy.deepcopy(guess).double().eval()
    W=model.layers[layer].weight.detach().clone();b=model.layers[layer].bias.detach().clone()
    device=W.device;din=W.shape[1];need=need or 2*din+64
    channels=list(range(len(W))) if channels is None else list(channels)
    gen=torch.Generator(device=device).manual_seed(seed);oracle=_Oracle(teacher)
    points={c:[] for c in channels};counts={c:0 for c in channels}
    if initial_points is not None:
        for c in channels:
            if c in initial_points and len(initial_points[c]):
                points[c]=[initial_points[c].to(device=device,dtype=torch.float64)]
                counts[c]=len(points[c][0])
    widths=tuple(widths) if widths is not None else (.0005,.001,.00025,.002,.0001,.005,.00005,.01,.02,.04)
    if not widths or min(widths)<=0:raise ValueError("Positive bracket widths required")
    yields={c:[0.]*len(widths) for c in channels}
    start=time.time()
    for rnd in range(rounds):
        todo=[c for c in channels if counts[c]<need]
        if not todo:break
        for begin in range(0,len(todo),8):
            group=todo[begin:begin+8]
            choice={c:(rnd%len(widths) if rnd<len(widths) or rnd%4==0
                       else max(range(len(widths)),key=lambda k:yields[c][k])) for c in group}
            eps={c:widths[choice[c]] for c in group}
            pools=candidate_pool(model,layer,group,gen,pool,(1.,4.,16.),eps)
            for c in group:
                hist=K._phi(model,torch.cat(points[c]),layer) if points[c] else None
                x,u,r,_,_=select_candidates(model,layer,c,gen,hist,pool=pool,batch=batch,
                    candidates=pools[c])
                if not len(x):continue
                xs,ok=locate_target(oracle,model,layer,c,x,u,r,gen)
                yields[c][choice[c]]=float(ok.double().mean())
                if ok.any():points[c].append(xs[ok]);counts[c]+=int(ok.sum())
        if verbose:print(f'[local] L{layer+1} round={rnd+1} enough={sum(v>=need for v in counts.values())}/{len(channels)} points_min={min(counts.values(),default=0)} q={oracle.n} t={time.time()-start:.1f}s',flush=True)
    accepted=torch.zeros(len(W),dtype=torch.bool,device=device);stats={}
    for c in channels:
        if counts[c]<din+64:
            stats[c]={'points':counts[c],'accepted':False};continue
        x=torch.cat(points[c]);h=K._phi(model,x,layer)
        # Independent random seeds throughout; interleave holdout before any fit.
        hold=torch.arange(len(h),device=device)%5==0;train=h[~hold];test=h[hold]
        if len(train)<din+16:continue
        try:w,bc,kept,gap=K.solve_from_points(train,W[c])
        except torch.linalg.LinAlgError:continue
        v=torch.cat([w,bc[None]]);v=v/v.norm();g=torch.cat([W[c],b[c:c+1]]);g=g/g.norm()
        angle=math.degrees(math.acos(float((v@g).abs().clamp(max=1))))
        residual=(test@w+bc).abs()/(1+test.norm(dim=1))
        valid=float((residual<1e-8).double().mean())
        fresh_count=0
        if angle<12 and (gap>=1e-6 or valid<.9):
            # More aggressive trimming is only a proposal. Require fresh
            # oracle-confirmed points before accepting a previously rejected fit.
            try:
                ww,bb,kk,gg=K.solve_from_points(train,W[c],rounds=32,drop=.04)
                vv=torch.cat([ww,bb[None]]);vv/=vv.norm()
                aa=math.degrees(math.acos(float((vv@g).abs().clamp(max=1))))
                if aa<12 and gg<1e-6:
                    oldw=model.layers[layer].weight[c].clone();oldb=model.layers[layer].bias[c].clone()
                    model.layers[layer].weight[c]=vv[:-1];model.layers[layer].bias[c]=vv[-1]
                    fresh=[]
                    try:
                        for epsv in (.0001,)*6+(.0005,)*2:
                            pp=candidate_pool(model,layer,[c],gen,pool,(1.,4.,16.),{c:epsv})[c]
                            xx,uu,rr,_,_=select_candidates(model,layer,c,gen,None,pool=pool,batch=batch,candidates=pp)
                            if not len(xx):continue
                            xx,hit=locate_target(oracle,model,layer,c,xx,uu,rr,gen)
                            if bool(hit.any()):fresh.append(xx[hit])
                            if sum(len(vf) for vf in fresh)>=need:break
                    finally:
                        model.layers[layer].weight[c]=oldw;model.layers[layer].bias[c]=oldb
                    if fresh:
                        xf=torch.cat(fresh);hf=K._phi(model,xf,layer)
                        hfhold=torch.arange(len(hf),device=device)%5==0
                        if len(hf[~hfhold])>=din+16:
                            ww,bb,kk,gg=K.solve_from_points(hf[~hfhold],W[c])
                            vv=torch.cat([ww,bb[None]]);vv/=vv.norm()
                            aa=math.degrees(math.acos(float((vv@g).abs().clamp(max=1))))
                            check=hf[hfhold]
                        else:check=hf
                        rf=(check@ww+bb).abs()/(1+check.norm(dim=1));vf=float((rf<1e-8).double().mean())
                        fresh_count=len(xf)
                        if fresh_count>=24 and vf>=.9 and gg<1e-6 and aa<12:
                            w,bc,kept,gap,v,angle,valid,residual=ww,bb,kk,gg,vv,aa,vf,rf
                            points[c].append(xf)
            except torch.linalg.LinAlgError:
                pass
        ok=angle<12 and gap<1e-6 and valid>=.9
        if ok and bool(accepted.any()):
            bank=torch.cat([W[accepted],b[accepted,None]],1)
            bank=bank/bank.norm(dim=1,keepdim=True)
            ok=bool((torch.minimum((bank-v).norm(dim=1),(bank+v).norm(dim=1))>1e-5).all())
        stats[c]=dict(points=len(h),accepted=ok,angle=angle,gap=gap,validation=valid,
                      median_residual=float(residual.median()),fresh_validation_points=fresh_count)
        if ok:W[c]=v[:-1];b[c]=v[-1];accepted[c]=True
    if archive is not None:
        archive.update(points={c:torch.cat(p).cpu() for c,p in points.items() if p},stats=stats,
                       queries=oracle.n,layer=layer,seed=seed,width_yields=yields)
    return W,b,accepted,oracle.n
