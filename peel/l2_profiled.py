"""PROFILED (nuisance-eliminated) LANDSCAPE (reviewer's decisive test).
L_prof(W2)=min_eta ||Phi(W2,eta)-Y*||^2. Along W2(t)=(1-t)W2guess + t W2*, hold W2 fixed and optimize
ONLY eta=(b2,W3,b3,W4,b4,W5,b5) as hard as possible (from BOTH true-eta and member-0 init, take min ->
best-case compensation = true lower bound on L_prof). Do it for value+first jets, then +second jets.
Flat L_prof => observations don't distinguish W2 (escalate order). Rising L_prof => info present ->
build a variable-projection W2 solver. Oracle diagnostic (truth used for path + scoring).
"""
import sys, time, torch
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
mem=[{kk:vv.to(dev).float() for kk,vv in sd.items()} for sd in pk["pop_states"]]
def match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r);c=torch.tensor(c);return r,c,torch.where(Cm[r,c]<Cp[r,c],-1.,1.)
al=[]
for sd in mem:
    W2m=sd["layers.1.weight"];r,c,s=match(W2m,W2t);Wa=torch.zeros_like(W2m);Wa[c]=W2m[r]*s[:,None];al.append(Wa)
W2guess=torch.stack(al).median(0).values
def m0_true():
    W2m=mem[0]["layers.1.weight"];b2m=mem[0]["layers.1.bias"];W3m=mem[0]["layers.2.weight"];b3m=mem[0]["layers.2.bias"]
    r,c,s=match(W2m,W2t);b2a=torch.zeros_like(b2m);b2a[c]=b2m[r]*s;W3a=torch.zeros_like(W3m);W3a[:,c]=W3m[:,r]*s[None,:];b3a=b3m+(W3m[:,r][:,s<0]).sum(1)
    return torch.cat([b2a.reshape(-1),W3a.reshape(-1),b3a,mem[0]["layers.3.weight"].reshape(-1),mem[0]["layers.3.bias"],mem[0]["layers.4.weight"].reshape(-1),mem[0]["layers.4.bias"]])
eta_m0=m0_true(); eta_true=torch.cat([b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t])
W2gp=torch.linalg.pinv(W2guess); gg=torch.Generator(device=dev).manual_seed(0); Hs=[]
for j in range(40):
    tt=torch.randn(80,generator=gg,device=dev)*2.0; tt[j%80]=0.0; Hs.append(torch.clamp(W2gp@(tt-b2t),1e-3,1-1e-3))
H=torch.stack(Hs)
def ueta(e): return e[:80],e[80:3280].reshape(40,80),e[3280:3320],e[3320:4600].reshape(32,40),e[4600:4632],e[4632:4952].reshape(10,32),e[4952:4962]
def observ(W2,e,order2):
    b2,W3,b3,W4,b4,W5,b5=ueta(e)
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2)
    z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4)
    out=s4@W5.t()+b5; U=(W2@B.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3
    dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    o=[out.reshape(-1),dout.reshape(-1)]
    if order2:
        spp2=sp2*(1-2*s2); spp3=sp3*(1-2*s3); spp4=sp4*(1-2*s4)
        dds2=spp2[:,None,:]*(U[None,:,:]**2); ddz3=dds2@W3.t(); dds3=spp3[:,None,:]*(dz3**2)+sp3[:,None,:]*ddz3
        ddz4=dds3@W4.t(); dds4=spp4[:,None,:]*(dz4**2)+sp4[:,None,:]*ddz4; ddout=dds4@W5.t()
        o.append(ddout.reshape(-1))
    return torch.cat(o)
def lsqr(Aop,Atop,b,n,damp,iters=40):
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
def opt_eta(W2,order2,e0,target,vsc,iters=45):
    resid=lambda e:(observ(W2,e,order2)-target)/vsc
    e=e0.clone();r=resid(e);c=float(r@r);n=e.numel();damp=1e-2
    for it in range(iters):
        _,vjpf=vjp(resid,e);Jt=lambda u:vjpf(u)[0];Jv=lambda v:jvp(resid,(e,),(v,))[1];ok=False
        for _ in range(12):
            d=lsqr(Jv,Jt,-r,n,damp);en=e+d;rn=resid(en);cn=float(rn@rn)
            if cn<c:e=en;r=rn;c=cn;damp=max(damp*0.3,1e-9);ok=True;break
            damp*=4
        if not ok or c<1e-18:break
    return c
def Lprof(W2,order2):
    target=observ(W2t,eta_true,order2).detach(); vsc=float(target.abs().max())
    c1=opt_eta(W2,order2,eta_true.clone(),target,vsc); c2=opt_eta(W2,order2,eta_m0.clone(),target,vsc)
    c=min(c1,c2); Y=target.numel(); return (c*(vsc**2)/Y)**0.5/ (float(target.pow(2).mean())**0.5)   # RMS resid / RMS signal
ts=[0.0,0.3,0.6,0.85,1.0]
print("[profiled landscape] L_prof = min_eta ||Phi(W2,eta)-Y*|| along W2(t)=(1-t)guess+t*truth")
print(f"guess W2 err = {float(((W2guess-W2t).norm(dim=1)/nt).mean()):.3f} mean")
print(f"{'t':>5} {'W2 err':>8} {'Lprof value+1st':>16} {'Lprof +2nd':>12}")
for tt in ts:
    W2=(1-tt)*W2guess+tt*W2t; we=float(((W2-W2t).norm(dim=1)/nt).mean())
    l1=Lprof(W2,False); l2=Lprof(W2,True)
    print(f"{tt:>5.1f} {we:>8.3f} {l1:>16.3e} {l2:>12.3e}",flush=True)
