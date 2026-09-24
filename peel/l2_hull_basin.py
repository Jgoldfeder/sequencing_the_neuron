"""DECISIVE test (reviewer): does the oracle-best population hull point CONVERGE in the Stage-1 joint
basin experiment? If even 5% oracle-hull stalls, correspondence work is a distraction.
true B + exact oracle jets. Inits: oracle-hull(5%), truth-free-hull(8.2%), reg-hull(10%), each with a
consistent committee-downstream init, joint (A,b2,downstream) refine. Plus controls:
 - near-truth (validates solver), oracle-hull + TRUE downstream FROZEN (isolates W2 quality).
"""
import sys, torch
from torch.func import jvp, vjp
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
ones=torch.ones(M,device=dev); a0=ones/M; _,_,V8=torch.linalg.svd(ones.reshape(1,M)); Z=V8[1:].t()
def affine(cand,tgt,lam=0.0):
    A=cand.transpose(1,2); AZ=A@Z; rhs=tgt-A@a0
    G=AZ.transpose(1,2)@AZ+lam*torch.eye(7,device=dev)[None]
    beta=torch.linalg.solve(G,(AZ.transpose(1,2)@rhs.unsqueeze(-1))).squeeze(-1)
    return torch.einsum('jdc,jc->jd',A,a0+beta@Z.t())
# hull inits
def align_to(refW):
    out=[]
    for sd in mem:
        W2m=sd["layers.1.weight"];r,c,s=match(W2m,refW);Wa=torch.zeros_like(W2m);Wa[c]=W2m[r]*s[:,None];out.append(Wa)
    return torch.stack(out)
Wor=align_to(W2t); hull_oracle=affine(Wor.permute(1,0,2),W2t)                      # 5%, TRUE order
Wtf=align_to(mem[0]["layers.1.weight"]); r0,c0,s0=match(mem[0]["layers.1.weight"],W2t)
tgt0=torch.zeros_like(W2t); tgt0[r0]=s0[:,None]*W2t[c0]
hull_tf=affine(Wtf.permute(1,0,2),tgt0); hull_reg=affine(Wtf.permute(1,0,2),tgt0,lam=0.1)   # member-0 order
# member-0 downstream, in TRUE order (for oracle hull) and member-0 order (for tf/reg)
def m0_aligned_true():
    W2m=mem[0]["layers.1.weight"];b2m=mem[0]["layers.1.bias"];W3m=mem[0]["layers.2.weight"];b3m=mem[0]["layers.2.bias"]
    r,c,s=match(W2m,W2t); b2a=torch.zeros_like(b2m); b2a[c]=b2m[r]*s
    W3a=torch.zeros_like(W3m); W3a[:,c]=W3m[:,r]*s[None,:]; b3a=b3m+(W3m[:,r][:,s<0]).sum(1)
    return (b2a,W3a,b3a,mem[0]["layers.3.weight"],mem[0]["layers.3.bias"],mem[0]["layers.4.weight"],mem[0]["layers.4.bias"])
D0_true=m0_aligned_true()
D0_m0=(mem[0]["layers.1.bias"],mem[0]["layers.2.weight"],mem[0]["layers.2.bias"],mem[0]["layers.3.weight"],mem[0]["layers.3.bias"],mem[0]["layers.4.weight"],mem[0]["layers.4.bias"])
Dtrue=(b2t,W3t,b3t,W4t,b4t,W5t,b5t)
# probes + observables
W2gp=torch.linalg.pinv(torch.stack([align_to(mem[0]["layers.1.weight"])[m] for m in range(M)]).median(0).values)
gg=torch.Generator(device=dev).manual_seed(0); Hs=[]
for j in range(80):
    tt=torch.randn(80,generator=gg,device=dev)*2.0; tt[j]=0.0; Hs.append(torch.clamp(W2gp@(tt-mem[0]["layers.1.bias"]),1e-3,1-1e-3))
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
target=obs(th_true).detach(); vsc=target.abs().max()
def pack(W2,D): b2,W3,b3,W4,b4,W5,b5=D; return torch.cat([(W2@B.t()).reshape(-1),b2,W3.reshape(-1),b3,W4.reshape(-1),b4,W5.reshape(-1),b5])
def wscore(th):
    W2=unpack(th)[0]@B; Cp=torch.cdist(W2,W2t);Cm=torch.cdist(-W2,W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);rel=e/nt[ci].cpu()
    return float(rel.mean()),float(rel.max()),int((rel<0.01).sum())
def lsqr(Aop,Atop,b,n,damp,iters=70):
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
def refine(resid,w0,iters=60):
    wf=w0.clone();r=resid(wf);c=float(r@r);n=wf.numel();damp=1e-2
    for it in range(iters):
        _,vjpf=vjp(resid,wf);Jt=lambda u:vjpf(u)[0];Jv=lambda v:jvp(resid,(wf,),(v,))[1];ok=False
        for _ in range(12):
            dwf=lsqr(Jv,Jt,-r,n,damp);wn=wf+dwf;rn=resid(wn);cn=float(rn@rn)
            if cn<c: wf=wn;r=rn;c=cn;damp=max(damp*0.3,1e-9);ok=True;break
            damp*=4
        if not ok or c<1e-16:break
    return wf,c
def run(tag,th0):
    resid=lambda th:(obs(th)-target)/vsc
    m0=wscore(th0);wf,c=refine(resid,th0);m1=wscore(wf)
    print(f"  {tag:40s} {m0[0]:.3e}/{m0[1]:.2e} -> {m1[0]:.3e}/{m1[1]:.2e}  <1%:{m1[2]}/80  loss {c:.2e}",flush=True)
print("[hull-basin] true B + oracle jets. joint (A,b2,downstream) unless noted.")
gc=torch.Generator(device=dev).manual_seed(7); run("(ctrl) near-truth+3%", th_true+0.03*th_true.abs().mean()*torch.randn(th_true.numel(),generator=gc,device=dev))
run("oracle-hull 5% + mem0-down (joint)", pack(hull_oracle,D0_true))
# oracle hull + TRUE downstream frozen: refine A(+b2) only
def run_frozen(tag,W2h):
    A0=(W2h@B.t()).reshape(-1)
    def resid(a):
        th=torch.cat([a,b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t]); return (obs(th)-target)/vsc
    m0=wscore(torch.cat([A0,b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t]))
    wf,c=refine(resid,A0.clone());
    th=torch.cat([wf,b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t]); m1=wscore(th)
    print(f"  {tag:40s} {m0[0]:.3e}/{m0[1]:.2e} -> {m1[0]:.3e}/{m1[1]:.2e}  <1%:{m1[2]}/80  loss {c:.2e}",flush=True)
run_frozen("oracle-hull 5% + TRUE down FROZEN (W2-only)", hull_oracle)
run("truth-free-hull 8.2% + mem0-down (joint)", pack(hull_tf,D0_m0))
run("reg-hull 10% + mem0-down (joint)", pack(hull_reg,D0_m0))
