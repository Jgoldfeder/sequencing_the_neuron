"""Matrix-free Gauss-Newton with a stochastic Jacobian-column preconditioner.
Uses already collected outputs, never calls an oracle.
"""
import time,math
import torch
from torch.func import functional_call,jvp,vjp
from gn_polish import _cg

def polish(net,X,Y,iters=3,cg_iters=60,chunk=4096,state=None,probes=12,tag='[pcgn]'):
    start=time.time();device=next(net.parameters()).device
    theta={k:p.detach().clone() for k,p in net.named_parameters() if p.requires_grad};keys=list(theta)
    xs=list(X.split(chunk));ys=list(Y.split(chunk))
    def f(p,x):return functional_call(net,p,(x,))
    def flat(p):return torch.cat([p[k].reshape(-1) for k in keys])
    def unflat(v):
        out={};i=0
        for k in keys:
            n=theta[k].numel();out[k]=v[i:i+n].reshape_as(theta[k]);i+=n
        return out
    def loss(p):
        with torch.no_grad():return sum(float((f(p,x)-y).square().sum()) for x,y in zip(xs,ys))
    def grad(p):
        g=None
        for x,y in zip(xs,ys):
            out,fn=vjp(lambda pp:f(pp,x),p);v=flat(fn(out-y)[0]);g=v if g is None else g+v
        return g
    def normal(p,v):
        dv=unflat(v);result=None
        for x in xs:
            _,jv=jvp(lambda pp:f(pp,x),(p,),(dv,));_,fn=vjp(lambda pp:f(pp,x),p);g=flat(fn(jv)[0]);result=g if result is None else result+g
        return result
    gen=torch.Generator(device=device).manual_seed(981)
    diagonal=torch.zeros_like(flat(theta))
    for x in xs:
        out,fn=vjp(lambda pp:f(pp,x),theta)
        for _ in range(probes):
            signs=(torch.randint(0,2,out.shape,device=device,generator=gen)*2-1).to(out.dtype)
            g=flat(fn(signs)[0]);diagonal+=g.square()/probes
    positive=diagonal[diagonal>0];floor=positive.median()*1e-8
    inv=diagonal.clamp_min(floor).rsqrt()
    state={} if state is None else state;lam=state.get('lam',1e-3);initial=current=loss(theta)
    print(tag,'diagonal range',float(positive.min()),float(positive.max()),'loss',current,flush=True)
    for it in range(iters):
        g=grad(theta);b=-inv*g
        if float(b.norm())<1e-14:break
        accepted=False
        for _ in range(4):
            dz,ncg=_cg(lambda v:inv*normal(theta,inv*v)+lam*v,b,cg_iters,1e-6)
            delta=unflat(inv*dz)
            for alpha in (1.,.5,.25,.125):
                cand={k:theta[k]+alpha*delta[k] for k in keys};value=loss(cand)
                if math.isfinite(value) and value<current:
                    theta=cand;current=value;accepted=True;break
            if accepted:lam=max(lam/3,1e-12);break
            lam*=10
        print(tag,'iter',it+1,'loss',current,'lambda',lam,'accepted',accepted,'seconds',round(time.time()-start,1),flush=True)
        if not accepted:break
    with torch.no_grad():
        for k,p in net.named_parameters():
            if k in theta:p.copy_(theta[k])
    state['lam']=lam
    return net,dict(loss_init=initial,loss_final=current,wall_s=time.time()-start)
