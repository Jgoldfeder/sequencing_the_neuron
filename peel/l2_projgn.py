"""PROJECTED GAUSS-NEWTON for W2 (reviewer's idea): update W2 only along directions the downstream
CANNOT imitate. Per iter: J=[J_A | J_eta] (our model, no teacher params); Q_eta=orth basis of col(J_eta);
project r and J_A off col(J_eta); solve min||J_A_perp dA + r_perp|| (LSQR, float32-safe); update A only.
eta (downstream) is held at member-0's WRONG value -- NOT updated, just quotiented out.

Decisive test: oracle 5% hull + member-0 wrong downstream (went 5%->13.7% under JOINT). If projected-GN
gives 5%->~1% or lower, downstream compensation was corrupting the W2 gradient. Then try 8%, 16%, 23% inits.
true B + exact oracle jets (isolating); truth only for scoring.
"""
import sys, time, torch
from torch.func import jacrev
torch.set_default_dtype(torch.float32)
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
dev="cuda"; CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
pop=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False)
dims=pop["dims"]
t=MLP(dims,act="sigmoid").to(dev).float(); t.load_state_dict(pop["teacher_state"]); t.eval()
W2t=t.layers[1].weight.detach(); b2t=t.layers[1].bias.detach(); nt=W2t.norm(dim=1)
W3t=t.layers[2].weight.detach(); b3t=t.layers[2].bias.detach(); W4t=t.layers[3].weight.detach(); b4t=t.layers[3].bias.detach()
W5t=t.layers[4].weight.detach(); b5t=t.layers[4].bias.detach()
Uw,Sw,Vh=torch.linalg.svd(W2t,full_matrices=False); B=Vh[:80]
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False)
mem=[{kk:vv.to(dev).float() for kk,vv in sd.items()} for sd in pk["pop_states"]]; M=len(mem)
def match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r);c=torch.tensor(c);return r,c,torch.where(Cm[r,c]<Cp[r,c],-1.,1.)
def align_to(refW):
    out=[]
    for sd in mem:
        W2m=sd["layers.1.weight"];r,c,s=match(W2m,refW);Wa=torch.zeros_like(W2m);Wa[c]=W2m[r]*s[:,None];out.append(Wa)
    return torch.stack(out)
ones=torch.ones(M,device=dev); a0=ones/M; _,_,V8=torch.linalg.svd(ones.reshape(1,M)); Z=V8[1:].t()
def affine(cand,tgt,lam=0.0):
    A=cand.transpose(1,2);AZ=A@Z;rhs=tgt-A@a0;G=AZ.transpose(1,2)@AZ+lam*torch.eye(7,device=dev)[None]
    beta=torch.linalg.solve(G,(AZ.transpose(1,2)@rhs.unsqueeze(-1))).squeeze(-1);return torch.einsum('jdc,jc->jd',A,a0+beta@Z.t())
Wor=align_to(W2t); hull_oracle=affine(Wor.permute(1,0,2),W2t)
Wtf=align_to(mem[0]["layers.1.weight"]); r0,c0,s0=match(mem[0]["layers.1.weight"],W2t); tgt0=torch.zeros_like(W2t); tgt0[r0]=s0[:,None]*W2t[c0]
hull_tf=affine(Wtf.permute(1,0,2),tgt0)
W2cons_tf=Wtf.median(0).values; W2mem0=mem[0]["layers.1.weight"]
def m0_true():
    W2m=mem[0]["layers.1.weight"];b2m=mem[0]["layers.1.bias"];W3m=mem[0]["layers.2.weight"];b3m=mem[0]["layers.2.bias"]
    r,c,s=match(W2m,W2t);b2a=torch.zeros_like(b2m);b2a[c]=b2m[r]*s;W3a=torch.zeros_like(W3m);W3a[:,c]=W3m[:,r]*s[None,:];b3a=b3m+(W3m[:,r][:,s<0]).sum(1)
    return (b2a,W3a,b3a,mem[0]["layers.3.weight"],mem[0]["layers.3.bias"],mem[0]["layers.4.weight"],mem[0]["layers.4.bias"])
eta0=torch.cat([x.reshape(-1) for x in m0_true()])       # member-0 downstream, TRUE order (WRONG values, frozen)
# probes + observables
W2gp=torch.linalg.pinv(W2cons_tf); gg=torch.Generator(device=dev).manual_seed(0); Hs=[]
for j in range(40):
    tt=torch.randn(80,generator=gg,device=dev)*2.0; tt[j%80]=0.0; Hs.append(torch.clamp(W2gp@(tt-mem[0]["layers.1.bias"]),1e-3,1-1e-3))
H=torch.stack(Hs)
sizes=[6400,80,3200,40,1280,32,320,10]; shp=[(80,80),(80,),(40,80),(40,),(32,40),(32,),(10,32),(10,)]
def unpack(th):
    o=0;out=[]
    for n,sh in zip(sizes,shp): out.append(th[o:o+n].reshape(sh)); o+=n
    return out
def obs(th):
    A,b2,W3,b3,W4,b4,W5,b5=unpack(th); W2=A@B
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2)
    z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4)
    out=s4@W5.t()+b5; U=(W2@B.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3
    dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    return torch.cat([out.reshape(-1),dout.reshape(-1)])
th_true=torch.cat([(W2t@B.t()).reshape(-1),b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t])
target=obs(th_true).detach(); vsc=float(target.abs().max())
def wscore(A):
    W2=A@B; Cp=torch.cdist(W2,W2t);Cm=torch.cdist(-W2,W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);rel=e/nt[ci].cpu()
    return float(rel.mean()),float(rel.max()),int((rel<0.01).sum())
def lsqr(Aop,Atop,b,n,damp,iters=80):
    beta=b.norm()
    if float(beta)==0: return torch.zeros(n,device=b.device)
    u=b/beta;v=Atop(u);alfa=v.norm();v=v/alfa.clamp_min(1e-30);w=v.clone();x=torch.zeros(n,device=b.device);phibar=beta;rhobar=alfa
    for _ in range(iters):
        u=Aop(v)-alfa*u;beta=u.norm();u=u/beta.clamp_min(1e-30)
        v=Atop(u)-beta*v;alfa=v.norm();v=v/alfa.clamp_min(1e-30)
        rb1=(rhobar**2+damp**2).sqrt();c1=rhobar/rb1;phibar=c1*phibar
        rho=(rb1**2+beta**2).sqrt();c=rb1/rho;s=beta/rho;theta=s*alfa;rhobar=-c*alfa;phi=c*phibar;phibar=s*phibar
        x=x+(phi/rho)*w;w=v-(theta/rho)*w
    return x
eta_true=torch.cat([b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t])
def proj_gn(A_init, eta, iters=40):
    A=A_init.clone()
    for it in range(iters):
        th=torch.cat([A.reshape(-1),eta])
        J=jacrev(obs,chunk_size=4096)(th)               # Y x 11362 (our model only)
        JA=J[:,:6400]; Jeta=J[:,6400:]
        Q,_=torch.linalg.qr(Jeta)                        # Y x 4962 nuisance basis
        r=(obs(th)-target)/vsc
        rp=r-Q@(Q.t()@r)
        QtJA=Q.t()@JA
        Aop=lambda v: (JA@v)-Q@(QtJA@v)                  # J_A_perp @ v
        Atop=lambda u: (JA.t()@u)-QtJA.t()@(Q.t()@u)     # J_A_perp^T @ u
        damp=1e-3; base=float(rp@rp); ok=False
        for _ in range(10):
            d=lsqr(Aop,Atop,-rp,6400,damp)
            An=A+d.reshape(80,80); thn=torch.cat([An.reshape(-1),eta])
            rn=(obs(thn)-target)/vsc; rpn=rn-Q@(Q.t()@rn); cn=float(rpn@rpn)
            if cn<base: A=An;ok=True;break
            damp*=4
        if not ok: break
    return A
gp=torch.Generator(device=dev).manual_seed(3)
eta_2=eta_true+0.02*eta_true.abs().mean()*torch.randn(eta_true.numel(),generator=gp,device=dev)
eta_5=eta_true+0.05*eta_true.abs().mean()*torch.randn(eta_true.numel(),generator=gp,device=dev)
print(f"[proj-GN control] does projection work as delta-eta grows? init=oracle-hull 5%. probes={H.shape[0]}")
A0=hull_oracle@B.t()
for tag,eta in [("eta=TRUE (dn=0)",eta_true),("eta=true+2%",eta_2),("eta=true+5%",eta_5),("eta=member-0 (far)",eta0)]:
    m0=wscore(A0); t0=time.time(); A=proj_gn(A0,eta); m1=wscore(A)
    print(f"  {tag:22s} {m0[0]:.3e} -> {m1[0]:.3e}/{m1[1]:.2e}  <1%:{m1[2]}/80  [{time.time()-t0:.0f}s]",flush=True)
