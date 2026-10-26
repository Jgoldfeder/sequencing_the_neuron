#!/usr/bin/env python3
"""One entry point for the leaky-ReLU LeNet committee + output-only recovery.

    python lenet_hybrid.py --device cuda:0 --output runs/lenet_new
    python lenet_hybrid.py --record recon/_record_lenet_c1c2_c5_105.pt --output runs/lenet_resume
    python lenet_hybrid.py --continue-run runs/lenet_resume

A fresh run invokes the original committee until its recorded prefix reaches the
handoff gate. Fresh committee convergence is budget-limited, not guaranteed.
No historical audit artifacts or recovered final-layer weights are loaded by default.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
TEACHER = ROOT / 'teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_leaky_relu.pt'
SHAPE = (1, 28, 28)
CONVS = [(1,6,5,1,2,2), (6,16,5,1,0,2), (16,120,5,1,0,0)]


def atomic_json(path, data):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n'); os.replace(tmp, path)


def load_dependencies():
    global torch, ConvNet, canonicalize, prefix, residual_subspace, seeds, prefix_jac, G, K
    import torch
    from nets import ConvNet
    from align import cnn_canonicalize_ as canonicalize
    from residual_seed import prefix, residual_subspace, seeds, _prefix_jac as prefix_jac
    import diag_guessfree_layer as G
    import kink_solve as K


def new_net(device):
    return ConvNet(SHAPE, CONVS, (84,), 10, 'leaky_relu').to(device).double()


def load_teacher(path, device):
    model = new_net(device)
    data = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(data.get('state_dict', data))
    for p in model.parameters(): p.requires_grad_(False)
    return model.eval()


class OutputOracle:
    """Only callable outputs are passed to the direct recovery engine."""
    def __init__(self, model): self._model, self.queries = model, 0
    def __call__(self, x):
        self.queries += len(x)
        with torch.no_grad():
            return torch.cat([self._model(xx) for xx in x.split(512)])


@contextlib.contextmanager
def adaptive_light(enabled=True):
    original = K.locate_light
    def locate(bb, x, u, radius, tol=1e-12):
        result = x.clone(); good = torch.zeros(len(x), device=x.device, dtype=torch.bool)
        for scale in (.1, .03, .01, .003, .001):
            idx = (~good).nonzero().flatten()
            if not len(idx): break
            out, ok = original(bb, x[idx], u[idx], radius[idx]*scale, tol)
            result[idx[ok]] = out[ok]; good[idx[ok]] = True
        return result, good
    if enabled: K.locate_light = locate
    try: yield
    finally: K.locate_light = original


class Recovery:
    def __init__(self, bb, args, record):
        self.bb, self.args, self.device = bb, args, torch.device(args.device)
        self.started=time.monotonic()
        self.net = new_net(self.device)
        self.gen = torch.Generator(device=self.device).manual_seed(args.seed)
        self.masks = {l: torch.zeros(self.net.layers[l].weight.shape[0], device=self.device, dtype=torch.bool) for l in range(4)}
        self.anchors = {l:{} for l in range(4)}; self.unsigned = {l:[] for l in range(4)}
        self.pending = {2:[],3:[]}; self.rounds = {2:0,3:0}
        self.polished = False; self.events = []; self.source = str(record)
        raw = torch.load(record, map_location=self.device, weights_only=False)
        if raw.get('format') == 'lenet-hybrid-v1':
            self.net.load_state_dict(raw['state_dict'])
            self.masks = {int(l):m.to(self.device) for l,m in raw['masks'].items()}
            self.anchors = {int(l):{int(j):x.to(self.device) for j,x in d.items()} for l,d in raw['anchors'].items()}
            self.unsigned = raw['unsigned']; self.pending = raw['pending']; self.rounds = raw['rounds']
            self.polished = raw['polished']; self.events = raw['events']
            self.gen.set_state(raw['generator_state'].cpu()); self.bb.queries = raw['queries']
        else:
            if raw.get('unsigned_search_only'):
                raise ValueError('Historical unsigned banks are not resume checkpoints; use this script\'s state.pt.')
            if tuple(raw.get('input_shape', ())) != SHAPE or raw.get('act') != 'leaky_relu':
                raise ValueError('Expected the 28x28 leaky-ReLU LeNet record, not tanh LeNet-5.')
            self.net.load_state_dict(raw['state_dict'])
            from method import _cnn_checkpoint_record
            normalized = _cnn_checkpoint_record(self.net, raw)
            if normalized is None: raise ValueError('Record has no certified rows.')
            state, masks = normalized; self.net.load_state_dict(state)
            for l,m in masks.items():
                if l<4: self.masks[l] = m.to(self.device)
            # Includes the compensating next-layer column scales. Do not just
            # normalize a prefix and leave the following rows in the old frame.
            canonicalize(self.net)
            self.polished = bool(self.masks[3].any())
        if not bool(self.masks[0].all() and self.masks[1].all()):
            raise ValueError('Direct phase requires fully recorded C1/C2. Committee stage did not reach its gate.')
        for p in self.net.parameters(): p.requires_grad_(False)

    def emit(self, event, **kw):
        row = dict(event=event, queries=self.bb.queries, elapsed_seconds=round(time.monotonic()-self.started,3), **kw); self.events.append(row)
        print(json.dumps(row), flush=True)

    def save(self, stage):
        # CPU conversion recursively avoids device-bound resume files.
        def cpu(v):
            if torch.is_tensor(v): return v.detach().cpu()
            if isinstance(v,dict): return {k:cpu(x) for k,x in v.items()}
            if isinstance(v,list): return [cpu(x) for x in v]
            return v
        data = cpu(dict(format='lenet-hybrid-v1', stage=stage, state_dict=self.net.state_dict(),
                        masks=self.masks, anchors=self.anchors, unsigned=self.unsigned,
                        pending=self.pending, rounds=self.rounds, polished=self.polished,
                        generator_state=self.gen.get_state(), queries=self.bb.queries,
                        source=self.source, events=self.events))
        tmp = self.args.output/'state.pt.tmp'; torch.save(data,tmp); os.replace(tmp,self.args.output/'state.pt')
        atomic_json(self.args.output/'progress.json', dict(stage=stage, counts={l:int(m.sum()) for l,m in self.masks.items()}, queries=self.bb.queries, events=self.events))

    def rows(self, fr):
        lay = self.net.layers[fr]
        return torch.cat([lay.weight.flatten(1),lay.bias[:,None]],1)

    def locate(self, x0, d, lo, hi):
        root = G.bisect_kink(self.bb,x0,d,float(lo),float(hi))
        delta = min(float(hi-lo)*.12,2e-6)
        for _ in range(8):
            ss = torch.tensor([root-2*delta,root-delta,root+delta,root+2*delta],device=self.device,dtype=torch.double)
            yy = self.bb(x0+ss[:,None]*d); sl=(yy[1]-yy[0])/delta; sr=(yy[3]-yy[2])/delta; diff=sl-sr
            if float(diff.norm())<1e-9: return None
            bl=yy[1]-sl*ss[1]; br=yy[2]-sr*ss[2]; rr=float(((br-bl)*diff).sum()/diff.square().sum())
            if abs(rr-root)>delta*2: return None
            residual=(bl+sl*rr-br-sr*rr).norm(); root=rr
            if float(residual)<1e-10: return root,delta
            delta*=.25
        return None

    def mapback(self, fr, x0, d, rt, delta):
        xstar=x0+rt*d; P=prefix_jac(self.net,xstar,fr); pinv=torch.linalg.pinv(P,rtol=1e-10)
        dim=x0.numel()
        for dt in (delta*.25,delta*.0625):
            xc=torch.stack([xstar-dt*d,xstar+dt*d]); _,zz=prefix(self.net,xc,fr)
            if bool(((zz[0]>0)!=(zz[1]>0)).any()): continue
            eps=min(1e-7,dt*.1); eye=torch.eye(dim,device=self.device,dtype=torch.double)*eps
            xs=torch.cat([xc[:1],xc[0]+eye,xc[1:],xc[1]+eye])
            with torch.no_grad(): patterns=torch.cat([prefix(self.net,x,fr)[1]>0 for x in xs.split(256)])
            count=dim+1
            if bool((patterns[:count]!=patterns[0]).any()) or bool((patterns[count:]!=patterns[count]).any()): continue
            y=self.bb(xs); jl=((y[1:count]-y[0])/eps).T; jr=((y[count+1:]-y[count])/eps).T
            _,s,v=torch.linalg.svd(jr-jl,full_matrices=False)
            if float(s[0])<1e-7 or float(s[1]/s[0])>1e-4: continue
            w=pinv.T@v[0]
            if float((P.T@w-v[0]).norm())>1e-4: continue
            h=prefix(self.net,xstar[None],fr)[0][0]
            row=torch.cat([w,(-w@h)[None]]); row/=row.norm()
            return dict(row=row.detach().cpu(), anchor=xstar.detach().cpu(), attempts=0)
        return None

    def duplicate(self, fr, row, include_pending=True):
        known=self.rows(fr)[self.masks[fr]]; known=known/known.norm(dim=1,keepdim=True).clamp_min(1e-30)
        if len(known) and bool(((known@row).abs()>.99999).any()): return True
        return include_pending and any(float((c['row'].to(self.device)@row).abs())>.99999 for c in self.pending[fr])

    def scan(self, fr, ends, starts):
        known=self.masks[fr].nonzero().flatten().tolist()
        Q=torch.linalg.qr(self.net.layers[fr].weight.flatten(1)[self.masks[fr]].T).Q
        for li,(left,right) in enumerate(zip(starts,ends)):
            x0=(left+right)/2; d=(right-left)/2
            unk,_,step=G.scan_line(self.bb,self.net,fr,x0,d,1,self.args.grid,known)
            for root in unk:
                loc=self.locate(x0,d,root-float(step),root+float(step))
                candidate=self.mapback(fr,x0,d,*loc) if loc else None
                if candidate is None: continue
                row=candidate['row'].to(self.device)
                if self.duplicate(fr,row): continue
                w=row[:-1]
                if fr==2 and float((w-Q@(Q.T@w)).norm()/w.norm())<.4: continue
                self.pending[fr].append(candidate)
                self.emit('candidate',layer=fr+1,pending=len(self.pending[fr]))
                self.save(f'discover_L{fr+1}')
            if li%4==0: self.emit('scan',layer=fr+1,line=li,unexplained=len(unk))
            if len(self.pending[fr])>=min(8 if fr==2 else 64,int((~self.masks[fr]).sum())): break

    def discover(self, fr):
        V,_,_=residual_subspace(self.bb,self.net,fr,self.masks[fr],n_points=40,gen=self.gen)
        if len(V)==0: raise RuntimeError('Residual span invalid: checkpoint retained; no rank truncation was applied.')
        ends=seeds(self.net,fr,V,256,s_max=12,steps=400,box=4,gen=self.gen)
        starts=torch.randn(256,784,device=self.device,dtype=torch.double,generator=self.gen)*.5
        offset=(self.rounds[fr]*self.args.lines)%256
        ids=[(offset+i)%256 for i in range(self.args.lines)]
        self.scan(fr,ends[ids],starts[ids])

    def verify_row(self, fr, row, anchor):
        x=(anchor[None]+torch.randn(32,784,device=self.device,dtype=torch.double,generator=self.gen)*2).requires_grad_(True)
        for _ in range(30):
            val=prefix(self.net,x,fr)[0]@row[:-1]+row[-1]; grad=torch.autograd.grad(val.sum(),x)[0]
            with torch.no_grad(): x.sub_(val[:,None]*grad/grad.square().sum(1,keepdim=True).clamp_min(1e-30))
        v=grad.detach(); v/=v.norm(dim=1,keepdim=True); x=x.detach()
        ss=torch.tensor([-2,-1,0,1,2],device=self.device,dtype=torch.double)*1e-5
        y=self.bb((x[:,None]+ss[None,:,None]*v[:,None]).reshape(-1,784)).reshape(32,5,10)
        dd=(y[:,2:]-2*y[:,1:-1]+y[:,:-2]).norm(dim=2)
        return int((dd[:,1]>100*torch.maximum(dd[:,0],dd[:,2]).clamp_min(1e-13)).sum())

    def refine(self, fr):
        remaining=[]
        for candidate in self.pending[fr]:
            row=candidate['row'].to(self.device); anchor=candidate['anchor'].to(self.device)
            if self.duplicate(fr,row,False): continue
            free=(~self.masks[fr]).nonzero().flatten()
            if not len(free): break
            c=int(free[0]); m=self.net.clone()
            with torch.no_grad(): m.layers[fr].weight[c]=row[:-1].view_as(m.layers[fr].weight[c]); m.layers[fr].bias[c]=row[-1]
            attempt=candidate.get('attempts',0); need=(1200 if fr==2 else 800)*(1+attempt)
            step=min(2+attempt,6)
            with adaptive_light(fr==3):
                hs=K.track_layer(self.bb,K.ConvNetUnits(m,fr),fr,
                    {c:(anchor[None],torch.tensor([c],device=self.device))},self.gen,
                    need=need,step=step,max_steps=600,par=16,verbose=False)
            fit=K._cnn_solve_points(hs[c],row[:-1],gen=self.gen) if c in hs else None
            if fit is None: quality=False; stats={}; hits=0
            else:
                w,b,H=fit; quality,stats=K._cnn_fit_quality(H,w,b)
                result=torch.cat([w,b[None]]); result/=result.norm()
                hits=self.verify_row(fr,result,anchor)
            self.emit('refine',layer=fr+1,quality=quality,hits=hits,attempt=attempt,stats=stats)
            if quality and hits>=30 and self.duplicate(fr,result,False):
                self.emit('duplicate_refined_row',layer=fr+1)
                continue
            if quality and hits>=30:
                with torch.no_grad():
                    self.net.layers[fr].weight[c]=result[:-1].view_as(self.net.layers[fr].weight[c])
                    self.net.layers[fr].bias[c]=result[-1]
                self.masks[fr][c]=True; self.anchors[fr][c]=anchor; self.unsigned[fr].append(c)
                self.emit('accepted',layer=fr+1,count=int(self.masks[fr].sum()))
                self.save(f'accepted_L{fr+1}')
            else:
                candidate['attempts']=attempt+1
                if candidate['attempts']<4: remaining.append(candidate)
            # Persist each accepted row, with the still-unprocessed queue.
        self.pending[fr]=remaining
        self.save(f'refined_L{fr+1}')

    def affine_last(self):
        fr=3; known=self.masks[fr].nonzero().flatten().tolist()
        if len(known)!=83 or any(j not in self.anchors[fr] for j in known):
            raise RuntimeError('Last-F6 decomposition requires anchors for all 83 known surfaces.')
        cols=[]
        for j in known:
            x=self.anchors[fr][j].clone().requires_grad_(True)
            val=prefix(self.net,x[None],fr)[0][0]@self.net.layers[fr].weight[j]+self.net.layers[fr].bias[j]
            d=torch.autograd.grad(val,x)[0]; d/=d.norm(); x=x.detach(); amp=None
            for dt in (2e-3,5e-4,1e-4,3e-5,1e-5,3e-6):
                co=torch.tensor([-2,-1.5,-1,1,1.5,2],device=self.device,dtype=torch.double)*dt
                pts=x+co[:,None]*d
                with torch.no_grad():
                    h,pre=prefix(self.net,pts,fr); z=h@self.net.layers[fr].weight.T+self.net.layers[fr].bias
                    others=[k for k in known if k!=j]; pat=torch.cat([pre>0,z[:,others]>0],1)
                    if bool((pat!=pat[0]).any()) or not bool(z[0,j]*z[-1,j]<0): continue
                yy=self.bb(pts); vl=co[:3]-co[:3].mean(); vr=co[3:]-co[3:].mean()
                sl=(yy[:3]*vl[:,None]).sum(0)/vl.square().sum(); sr=(yy[3:]*vr[:,None]).sum(0)/vr.square().sum()
                gl=(z[2,j]-z[0,j])/(co[2]-co[0]); amp=(sr-sl)/(.99*gl); break
            if amp is None: raise RuntimeError(f'No isolated kink amplitude for F6 row {j}')
            cols.append(amp)
        AA=torch.stack(cols,1)
        X=torch.randn(4096,784,device=self.device,dtype=torch.double,generator=self.gen)*.5
        with torch.no_grad(): H=torch.cat([prefix(self.net,x,fr)[0] for x in X.split(256)]); Y=self.bb(X)
        W=self.net.layers[fr].weight[self.masks[fr]]; b=self.net.layers[fr].bias[self.masks[fr]]
        Z=H@W.T+b; design=torch.cat([H,torch.ones_like(H[:,:1])],1)
        def linear(y): return torch.linalg.lstsq(design.cpu(),y.cpu(),driver='gelsd').solution.to(self.device)
        U=linear(Y-torch.nn.functional.leaky_relu(Z,.01)@AA.T)[:-1].T
        Q=torch.linalg.qr(W.T).Q; u,_,_=torch.linalg.svd(U-(U@Q)@Q.T,full_matrices=False)
        Pu=torch.eye(10,device=self.device,dtype=torch.double)-u[:,:1]@u[:,:1].T
        coeff=(Pu@U)@torch.linalg.pinv(W); hp=Pu@AA
        flip=-(coeff*hp).sum(0)/(1.01*hp.square().sum(0)); bits=flip.round()
        error=float((flip-bits).abs().max())
        if error>.01 or not bool(((bits==0)|(bits==1)).all()): raise RuntimeError('Last-unit orientation decomposition failed its binary check.')
        remainder=Y-torch.nn.functional.leaky_relu(Z*(1-2*bits),.01)@AA.T
        fit=linear(remainder); _,sv,v=torch.linalg.svd(fit[:-1].T,full_matrices=False)
        if float(sv[1]/sv[0])>.01: raise RuntimeError('Last-unit affine remainder is not rank one.')
        w=v[0]; count=64; sign=torch.cat([torch.ones(32,device=self.device),-torch.ones(32,device=self.device)])
        x=(torch.rand(count,784,device=self.device,dtype=torch.double,generator=self.gen)*2-1).requires_grad_(True)
        opt=torch.optim.Adam([x],lr=.05)
        for _ in range(500):
            opt.zero_grad(); loss=-((prefix(self.net,x,fr)[0]@w)*sign).mean(); loss.backward(); opt.step()
            with torch.no_grad(): x.clamp_(-4,4)
        self.emit('affine_last',binary_error=error,rank_ratio=float(sv[1]/sv[0]))
        self.scan(fr,x.detach()[:32],x.detach()[32:])

    def orient(self, fr):
        if not bool(self.masks[fr].all()): raise RuntimeError('Cannot orient an incomplete bank.')
        for j in list(self.unsigned[fr]):
            x=self.anchors[fr][j].clone().requires_grad_(True); w=self.net.layers[fr].weight[j].flatten(); b=self.net.layers[fr].bias[j]
            for _ in range(8):
                val=prefix(self.net,x[None],fr)[0][0]@w+b; gr=torch.autograd.grad(val,x)[0]
                with torch.no_grad(): x.sub_(val*gr/gr.square().sum().clamp_min(1e-30))
            x=x.detach(); A=self.net.layers[fr].weight.flatten(1)@prefix_jac(self.net,x,fr)
            target=torch.zeros(len(A),device=self.device,dtype=torch.double); target[j]=1
            d=torch.linalg.pinv(A,rtol=1e-11)@target; d/=d.norm(); chosen=None
            for dt in (1e-3,3e-4,1e-4,3e-5,1e-5,3e-6):
                co=torch.tensor([-2,-1,1,2],device=self.device,dtype=torch.double)*dt; pts=x+co[:,None]*d
                _,z=prefix(self.net,pts,fr)
                if bool(((z>0)!=(z[0]>0)).any()): continue
                y=self.bb(pts); left=(y[1]-y[0])/dt; right=(y[3]-y[2])/dt
                ratio=float(right.norm()/left.norm()); cosine=float(torch.nn.functional.cosine_similarity(left,right,dim=0))
                if cosine>.999 and (90<ratio<110 or .009<ratio<.011): chosen=1 if ratio>1 else -1; break
            if chosen is None: raise RuntimeError(f'Cannot orient layer {fr+1} row {j}; no reliable slope ratio.')
            with torch.no_grad(): self.net.layers[fr].weight[j].mul_(chosen); self.net.layers[fr].bias[j].mul_(chosen)
            self.unsigned[fr].remove(j); self.emit('oriented',layer=fr+1,slot=j,ratio=ratio)
            self.save(f'orient_L{fr+1}')

    def polish_c5(self):
        if self.polished: return
        original=K.seed_brackets
        def fixed(*a,**kw): kw['eps']=1e-4; return original(*a,**kw)
        K.seed_brackets=fixed
        try:
            out=K.polish_layer(self.bb,self.net,2,list(range(120)),self.gen,eps0=1e-4,
                need=700,max_rounds=16,max_rows=12000,verbose=True,full=False,scales=(.5,1.,2.))
        finally: K.seed_brackets=original
        accepted=[]
        for j,(w,b,H) in out.items():
            ok,_=K._cnn_fit_quality(H,w,b)
            if not ok: continue
            row=torch.cat([w,b[None]]); row/=row.norm()
            old=self.rows(2)[j]; scale=old.norm()
            # Keep the existing C5 output gauge. F6 input columns remain valid.
            if float((row-old/scale) .abs().max())>1e-6: continue
            with torch.no_grad(): self.net.layers[2].weight[j]=(row[:-1]*scale).view_as(self.net.layers[2].weight[j]); self.net.layers[2].bias[j]=row[-1]*scale
            accepted.append(j)
        if len(accepted)!=120: raise RuntimeError(f'C5 polish accepted {len(accepted)}/120; state retained for retry.')
        self.polished=True; self.emit('polished_C5',count=120); self.save('polished_C5')

    def finish(self):
        for fr in (2,3):
            while not bool(self.masks[fr].all()):
                if self.rounds[fr]>=self.args.max_rounds: raise RuntimeError(f'Layer {fr+1} budget exhausted at {int(self.masks[fr].sum())}; resume state saved.')
                if self.pending[fr]: self.refine(fr)
                if bool(self.masks[fr].all()): break
                self.emit('round',layer=fr+1,round=self.rounds[fr],known=int(self.masks[fr].sum()))
                if fr==3 and int(self.masks[fr].sum())==83: self.affine_last()
                else: self.discover(fr)
                self.rounds[fr]+=1; self.refine(fr)
            self.orient(fr)
            if fr==2: self.polish_c5()
        self.fit_head()

    def fit_head(self):
        if any(not bool(m.all()) for m in self.masks.values()) or any(self.unsigned.values()): raise RuntimeError('Head fit requires a complete signed prefix.')
        X=torch.cat([torch.randn(2048,784,device=self.device,dtype=torch.double,generator=self.gen)*1.5,
                     (torch.rand(2048,784,device=self.device,dtype=torch.double,generator=self.gen)*2-1)*3])
        with torch.no_grad(): H=torch.cat([prefix(self.net,x,4)[0] for x in X.split(256)]); Y=self.bb(X)
        design=torch.cat([H,torch.ones_like(H[:,:1])],1); s=torch.linalg.svdvals(design)
        if float(s[-1])<1e-9: raise RuntimeError('Head design is rank deficient.')
        coef=torch.linalg.lstsq(design.cpu(),Y.cpu(),driver='gelsd').solution.to(self.device)
        with torch.no_grad(): self.net.layers[4].weight.copy_(coef[:-1].T); self.net.layers[4].bias.copy_(coef[-1])
        checks={}
        for scale in (.25,.5,1.,1.5,3.,6.):
            x=torch.randn(self.args.validation_size,784,device=self.device,dtype=torch.double,generator=self.gen)*scale
            with torch.no_grad(): pred=torch.cat([self.net(xx) for xx in x.split(256)]); err=(pred-self.bb(x)).abs()
            checks[str(scale)]=dict(max_output_error=float(err.max()),mean_output_error=float(err.mean()))
        worst=max(v['max_output_error'] for v in checks.values())
        report=dict(complete=worst<=self.args.output_tolerance,counts=[6,16,120,84,10],output_tests=checks,
                    max_output_error=worst,output_tolerance=self.args.output_tolerance,queries=self.bb.queries,oracle_contract='fp64 outputs',teacher_parameters_used=False)
        atomic_json(self.args.output/'validation.json',report)
        self.save('validated' if report['complete'] else 'validation_failed')
        if not report['complete']: raise RuntimeError(f'Fresh output error {worst:g} exceeds {self.args.output_tolerance:g}; no complete checkpoint written.')
        ck=dict(input_shape=SHAPE,conv_cfgs=CONVS,fc_dims=(84,),out_dim=10,act='leaky_relu',
                state_dict={k:v.detach().cpu() for k,v in self.net.state_dict().items()},
                partial_state={k:v.detach().cpu() for k,v in self.net.state_dict().items()},
                partial_mask={l:torch.ones(self.net.layers[l].weight.shape[0],dtype=torch.bool) for l in range(5)},
                frozen={},complete_lenet_recovery=True,validation=report)
        tmp=self.args.output/'complete.pt.tmp'; torch.save(ck,tmp); os.replace(tmp,self.args.output/'complete.pt')
        self.emit('complete',checkpoint=str(self.args.output/'complete.pt'),max_output_error=worst)


def committee_worker(args):
    from method import Cfg, reconstruct_cnn
    from run import VARIANTS
    teacher=load_teacher(args.teacher,args.device).float()
    options=dict(VARIANTS['mergedbest'])
    options.update(act='leaky_relu',epochs=args.committee_epochs,freeze_reinit=True,
                   fast_peel_partial=True,loc_refine=True,f64=True,masked_consensus=True,
                   layer_consensus=True,layer_space_init=False,balance=1,balance_refresh=5,
                   peel_refresh=True,retry=args.committee_retries,window=60,
                   pop_save_every=1,pop_save_path=str(args.output/f'committee_{uuid.uuid4().hex[:8]}.pt'),log_every=1)
    historical = args.committee_recipe == 'historical'
    if historical and args.committee_stage == 'prefix':
        # lenetpeel_fixed.log: row-normalized directional LR for C1/C2.
        options.update(layer_space_init=True,balance_dirlr=.03,balance_dirlr_decay=.85)
    if historical and args.committee_stage == 'c5':
        # lenet_c5_cheat.log is the source of the 103-row record.
        options.update(cheat=True,cheat_bb_ref=True,layer_space_init=False)
        print('[recipe] C5: historical teacher-gradient disagreement (not output-only)',flush=True)
    cfg=Cfg(p=8,q=args.committee_q,outer=args.committee_outer,**options)
    os.environ['POP_SAVE_ALL']='1'
    resume_state=resume_record=None
    if args.committee_resume:
        from method import _cnn_checkpoint_record
        raw=torch.load(args.committee_resume,map_location='cpu',weights_only=False)
        scratch=new_net('cpu');scratch.load_state_dict(raw['state_dict'])
        resume_state=raw['state_dict'];resume_record=_cnn_checkpoint_record(scratch,raw)
        if resume_record is None:resume_record=(resume_state,{})
    reconstruct_cnn(teacher,SHAPE,CONVS,10,cfg,args.device,seed=args.seed,fc_dims=(84,),
                    resume_state=resume_state,resume_record=resume_record)


def committee_snapshot_counts(path):
    try:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        masks={l:m.bool().clone() for l,(_,_,m) in (ck.get('frozen') or {}).items()}
        for l,m in (ck.get('partial_mask') or {}).items():
            masks[l]=m.bool()|masks.get(l,torch.zeros_like(m,dtype=torch.bool))
        return tuple(int(masks.get(l,torch.zeros(n)).sum()) for l,n in enumerate((6,16,120)))
    except (OSError,RuntimeError,EOFError,pickle.UnpicklingError): return None


def best_committee_snapshot(directory):
    candidates=[]
    for path in directory.glob('committee*_it*.pt'):
        counts=committee_snapshot_counts(path)
        if counts is not None: candidates.append((counts,path.stat().st_mtime,path))
    return max(candidates)[2] if candidates else None


def direct_handoff(directory, reason):
    path=best_committee_snapshot(directory)
    counts=committee_snapshot_counts(path) if path else None
    if counts is None or counts[:2]!=(6,16):
        raise RuntimeError(f'{reason}: no complete certified C1/C2 prefix; latest counts {counts}.')
    print(f'[committee] {reason}: TRYING direct phase at C5={counts[2]}/120 from {path}',flush=True)
    atomic_json(directory/'handoff.json',dict(reason=reason,record=str(path),counts=counts))
    return path


def committee_phase(args):
    """Run the recorded prefix/C5 recipes, then attempt direct recovery at any C5 count."""
    import signal
    existing=best_committee_snapshot(args.output)
    counts=committee_snapshot_counts(existing) if existing else None
    if args.handoff_now:
        return direct_handoff(args.output,'explicit handoff now')
    if counts and counts[:2]==(6,16) and counts[2]>=args.handoff_c5:
        return direct_handoff(args.output,'target reached')
    stages=['prefix','c5'] if args.committee_recipe=='historical' else ['ordinary']
    if counts and counts[:2]==(6,16) and 'prefix' in stages: stages.remove('prefix')
    for stage in stages:
        existing=best_committee_snapshot(args.output)
        budget=args.committee_c5_outer if stage=='c5' else args.committee_outer
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--committee-worker',
             '--output',str(args.output),'--teacher',str(args.teacher),'--device',args.device,
             '--seed',str(args.committee_seed),'--committee-retries',str(args.committee_retries),
             '--committee-outer',str(budget),'--committee-q',str(args.committee_q),
             '--committee-epochs',str(args.committee_epochs),'--committee-recipe',args.committee_recipe,
             '--committee-stage',stage]
        if existing: cmd.extend(['--committee-resume',str(existing)])
        log=args.output/'committee.log'
        print(f'[committee] recipe={args.committee_recipe} stage={stage} budget={budget}; log {log}',flush=True)
        atomic_json(args.output/'phase1_recipe.json',dict(recipe=args.committee_recipe,stage=stage,
                    seed=args.committee_seed,budget=budget,q=args.committee_q,retries=args.committee_retries,
                    resumed_record=str(existing) if existing else None,
                    teacher_gradient_disagreement=args.committee_recipe=='historical' and stage=='c5'))
        reached=False
        with log.open('a') as stream:
            stream.write(f'\n[worker] {time.strftime("%Y-%m-%d %H:%M:%S")} isolated {stage} segment\n');stream.flush()
            env=os.environ.copy();env.pop('PEEL_DEBUG_NEAR_TEACHER',None)
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
            try:
                while True:
                    best=best_committee_snapshot(args.output)
                    current=committee_snapshot_counts(best) if best else None
                    prefix_done=current is not None and current[:2]==(6,16)
                    if prefix_done and (stage=='prefix' or current[2]>=args.handoff_c5):
                        reached=True;break
                    if proc.poll() is not None: break
                    time.sleep(2)
            finally:
                if proc.poll() is None:
                    with contextlib.suppress(ProcessLookupError):os.killpg(proc.pid,signal.SIGTERM)
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(ProcessLookupError):os.killpg(proc.pid,signal.SIGKILL)
                        proc.wait()
        if stage=='prefix' and reached:
            continue
        # A target is an early handoff trigger, never a minimum number of C5 rows.
        return direct_handoff(args.output,'target reached' if reached else f'committee ended (exit {proc.returncode}), below target allowed')
    return direct_handoff(args.output,'committee stages ended')


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--record',type=Path,help='Skip committee; load a certified signed C1/C2/partial-C5 or later record.')
    group.add_argument('--continue-run',type=Path,help='Resume this driver\'s own output directory and state.pt.')
    p.add_argument('--output',type=Path,help='New output directory (must not exist). Default: unique directory under runs/.')
    p.add_argument('--teacher',type=Path,default=TEACHER,help='Saved teacher used as a forward-only fp64 oracle in direct recovery.')
    p.add_argument('--device',default='cuda:0'); p.add_argument('--seed',type=int,default=22502)
    p.add_argument('--lines',type=int,default=32); p.add_argument('--grid',type=int,default=32001)
    p.add_argument('--max-rounds',type=int,default=12)
    p.add_argument('--handoff-c5',type=int,default=103,help='Early handoff target, not a required minimum.')
    p.add_argument('--handoff-now',action='store_true',help='On committee resume, try direct recovery immediately with the best saved prefix.')
    p.add_argument('--committee-recipe',choices=('historical','ordinary'),default='historical',help='Historical uses teacher-gradient disagreement for C5, as in the 103-row source run.')
    p.add_argument('--committee-c5-outer',type=int,default=40)
    p.add_argument('--committee-stage',choices=('prefix','c5','ordinary'),default='prefix',help=argparse.SUPPRESS)
    p.add_argument('--committee-outer',type=int,default=60); p.add_argument('--committee-q',type=int,default=20000)
    p.add_argument('--committee-epochs',type=int,default=10)
    p.add_argument('--committee-seed',type=int,default=0,help='First-phase committee seed; direct recovery uses --seed.')
    p.add_argument('--committee-retries',type=int,default=5,help='First-phase retries with solved rows retained.')
    p.add_argument('--validation-size',type=int,default=1024,help='Fresh inputs at each of six validation scales.')
    p.add_argument('--output-tolerance',type=float,default=1e-6)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--plan',action='store_true',help='Print the execution plan without running or allocating a GPU.')
    p.add_argument('--committee-worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--committee-resume',type=Path,help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p=parser(); args=p.parse_args(argv)
    if not 1<=args.lines<=256 or args.grid<101 or not 0<=args.handoff_c5<=120:
        p.error('Require 1<=lines<=256, grid>=101, 0<=handoff-c5<=120.')
    if args.committee_c5_outer<1: p.error('--committee-c5-outer must be positive.')
    if args.committee_retries<0: p.error('--committee-retries must be nonnegative.')
    if args.threads<1 or args.output_tolerance<=0 or args.max_rounds<1 or args.validation_size<1 or args.committee_outer<1 or args.committee_q<1 or args.committee_epochs<0:
        p.error('Budgets and validation size must be positive (epochs may be zero for a smoke test).')
    if args.continue_run and args.output: p.error('--continue-run already identifies the output directory.')
    args.teacher=args.teacher.resolve()
    if args.record: args.record=args.record.resolve()
    args.output=(args.continue_run or args.output or ROOT/'runs'/f'lenet_{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}').resolve()
    if args.plan:
        print(json.dumps(dict(start='resume' if args.continue_run else 'record' if args.record else 'fresh committee',
            teacher=str(args.teacher),record=str(args.record) if args.record else None,output=str(args.output),
            stages=['committee handoff (fresh only)','C5 residual discovery/refinement','C5 output-only signs',
                    'C5 precision polish with preserved gauges','F6 residual discovery/refinement + last-unit affine fallback',
                    'F6 output-only signs','linear head fit','independent output validation'],
            committee_recipe=args.committee_recipe,below_target='attempt direct recovery if C1/C2 complete',
            fresh_committee_convergence='budget limited; not guaranteed',oracle='direct: fp64 outputs; historical C5 committee: teacher gradients'),indent=2)); return 0
    load_dependencies(); torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    if not args.teacher.is_file(): p.error(f'Teacher does not exist: {args.teacher}')
    args.teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest()
    if args.committee_worker:
        committee_worker(args); return 0
    if args.continue_run:
        if not (args.output/'state.pt').is_file() and not list(args.output.glob('committee*_it*.pt')):
            p.error('No recovery or committee checkpoints in --continue-run directory.')
        config=json.loads((args.output/'config.json').read_text())
        if config['teacher']!=str(args.teacher) or config.get('teacher_sha256',args.teacher_sha256)!=args.teacher_sha256:
            p.error('Resume must use the same teacher file contents as the original run.')
        record=args.output/'state.pt' if (args.output/'state.pt').is_file() else None
    else:
        if args.output.exists():p.error('Output directory already exists; use --continue-run or a new --output.')
        args.output.mkdir(parents=True,exist_ok=False)
        atomic_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
        record=args.record
    solver=None
    try:
        if record is None: record=committee_phase(args)
        solver=Recovery(OutputOracle(load_teacher(args.teacher,args.device)),args,record)
        solver.save('loaded'); solver.finish()
    except (Exception,KeyboardInterrupt) as exc:
        if solver is not None: solver.save('interrupted' if isinstance(exc,KeyboardInterrupt) else 'incomplete')
        atomic_json(args.output/'failure.json',dict(error=str(exc),type=type(exc).__name__,complete=False))
        print(f'[incomplete] {exc}\nArtifacts: {args.output}',file=sys.stderr,flush=True)
        return 130 if isinstance(exc,KeyboardInterrupt) else 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
