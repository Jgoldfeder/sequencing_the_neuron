"""Experimental multi-region Jacobian jumps; teacher access is forward-only.

Measurements use full coordinate sweeps in input space. The joint system uses
prefix products, never a prefix inverse. Dense fitting is a diagnostic reference;
LSMR is the memory-bounded alternative. Neither is a production accuracy gate.
"""
import time
import numpy as np
import torch
from scipy.linalg import lstsq
from scipy.sparse.linalg import LinearOperator, lsmr
import kink_solve as K
from peel.informative_kinks import candidate_pool
from verify_layer1 import _Oracle


@torch.no_grad()
def measure(oracle, x, u, radius, nscan):
    """Check two step sizes and rank one before accepting a local jump."""
    d = x.numel()
    eye = torch.eye(d, device=x.device, dtype=x.dtype)
    e = radius * 1.2 / (nscan-1) * 2.0
    for attempt in range(7):
        step = e * .2
        centers = torch.stack([x-e*u, x+e*u])
        offsets = torch.cat([eye*step, -eye*step, eye*step/2, -eye*step/2])
        y = oracle((centers[:,None]+offsets[None]).reshape(-1,d)).reshape(2,4,d,-1)
        jc = (y[:,0]-y[:,1])/(2*step)
        jf = (y[:,2]-y[:,3])/step
        delta = jf[1]-jf[0]
        # CPU SVD avoids the GPU small-matrix solver overhead.
        a,s,vh = np.linalg.svd(delta.cpu().numpy(), full_matrices=False)
        norm = float(s[0])
        agreement = float((jc-jf).norm().cpu()) / max(norm,1e-300)
        rank_error = float(np.linalg.norm(s[1:])) / max(norm,1e-300)
        if agreement < 1e-6 and rank_error < 1e-6:
            return a[:,0], dict(step=float(step), agreement=agreement, rank_error=rank_error, attempts=attempt+1)
        e *= .25
    return None, dict(agreement=agreement, rank_error=rank_error, attempts=7)


def prefix_masks(guess, x, layer):
    a=x[None]
    masks=[]
    with torch.no_grad():
        for i in range(layer):
            z=guess.layers[i](a)
            masks.append(torch.where(z[0]>=0,torch.ones_like(z[0]),torch.full_like(z[0],guess.act.negative_slope)).cpu().numpy())
            a=guess.act(z)
    return masks


def joint_fit(weights, regions, wguess, dense=False):
    """Pin one weight coordinate, jointly impose parallel prefix normals."""
    d=len(wguess); pin=int(np.argmax(abs(wguess))); cols=np.delete(np.arange(d),pin)
    # One scalar bias unknown accompanies d-1 unpinned weights.
    def jt(w,masks):
        for weight,mask in reversed(list(zip(weights,masks))): w=weight.T@(mask*w)
        return w
    def j(v,masks):
        for weight,mask in zip(weights,masks): v=mask*(weight@v)
        return v
    scales=[]
    for masks,n,h in regions:
        scales.append(max(np.linalg.norm(jt(wguess,masks)),1e-12))
    def forward(z):
        w=z[:d]; b=z[d]; out=[]
        for (masks,n,h),scale in zip(regions,scales):
            p=jt(w,masks); out.extend(((p-n*np.dot(n,p))/scale, np.array([(h@w+b)/(1+np.linalg.norm(h))])))
        return np.concatenate(out)
    def transpose(y):
        w=np.zeros(d); b=0.; off=0
        for (masks,n,h),scale in zip(regions,scales):
            v=y[off:off+len(n)]; off+=len(n)
            w+=j(v-n*np.dot(n,v),masks)/scale
            r=y[off]/(1+np.linalg.norm(h)); off+=1
            w+=h*r; b+=r
        return np.r_[w,b]
    free=np.r_[cols,d]; fixed=np.zeros(d+1); fixed[pin]=wguess[pin]
    def mv(z):
        full=np.zeros(d+1); full[free]=z
        return forward(full)
    op=LinearOperator((len(forward(fixed)),d),matvec=mv,rmatvec=lambda y:transpose(y)[free],dtype=np.float64)
    rhs=-forward(fixed)
    start=time.perf_counter()
    if dense:
        matrix=np.column_stack([mv(e) for e in np.eye(d)])
        sol,_,rank,s=lstsq(matrix,rhs,cond=1e-13)
        info=dict(rank=int(rank),condition=float(s[0]/s[-1]))
    else:
        result=lsmr(op,rhs,atol=1e-14,btol=1e-14,conlim=1e14,maxiter=6*d)
        sol=result[0]; info=dict(stop=int(result[1]),iterations=int(result[2]),residual=float(result[3]))
    full=fixed.copy(); full[free]=sol
    full/=np.linalg.norm(full[:d])
    info['seconds']=time.perf_counter()-start
    return full,info


@torch.no_grad()
def recover_neuron(teacher, guess, layer, channel, gen, regions=8, rounds=12):
    oracle=_Oracle(teacher); records=[]; measurement=[]; rejected=0
    wg=guess.layers[layer].weight[channel]
    weights=[L.weight.detach().cpu().numpy() for L in guess.layers[:layer]]
    nscan=25+12*layer
    for rnd in range(rounds):
        X,U,R,H,cross=candidate_pool(guess,layer,[channel],gen,96,(1.,4.,16.),{channel:.02})[channel]
        if not len(X): continue
        idx=cross.argsort()[:24]; X,U,R=X[idx],U[idx],R[idx]
        xs,ok=K.locate(oracle,X,U,R,gen,guess,layer,wg,K=nscan)
        for k in ok.nonzero()[:,0].tolist():
            n,info=measure(oracle,xs[k],U[k],R[k],nscan)
            if n is None: rejected+=1; continue
            h=K._phi(guess,xs[k:k+1],layer)[0].cpu().numpy()
            records.append((prefix_masks(guess,xs[k],layer),n,h)); measurement.append(info)
            if len(records)>=regions: break
        if len(records)>=regions: break
    diagnostics=dict(regions=len(records),rejected=rejected,queries=oracle.n,measurements=measurement)
    if len(records)<2: return None,None,diagnostics
    dense,di=joint_fit(weights,records,wg.cpu().numpy(),dense=True)
    iterative,ii=joint_fit(weights,records,wg.cpu().numpy(),dense=False)
    diagnostics.update(dense=di,iterative=ii)
    return dense,iterative,diagnostics
