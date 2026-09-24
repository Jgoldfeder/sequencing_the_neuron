"""VARIABLE-PROJECTION W2 solver. Descend L_prof(W2)=min_eta||Phi(W2,eta)-Y*||^2 (monotone, per
l2_profiled). Each OUTER step: (1) optimize eta fully at current W2 (warm-started); (2) at that eta*,
step W2 along the profiled residual with the nuisance tangent projected out (Golub-Pereyra):
  dA = argmin ||(I-P_eta) J_A dA + r||,  P_eta=proj onto col(J_eta at eta*).
Key vs the failed projected-GN: eta is RE-OPTIMIZED each step (never frozen far). true B, oracle jets.
"""
import sys, time, torch
from torch.func import jvp, vjp, jacrev
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
eta_m0=m0_true()
W2gp=torch.linalg.pinv(W2guess); gg=torch.Generator(device=dev).manual_seed(0); Hs=[]
for j in range(30):
    tt=torch.randn(80,generator=gg,device=dev)*2.0; tt[j%80]=0.0; Hs.append(torch.clamp(W2gp@(tt-b2t),1e-3,1-1e-3))
H=torch.stack(Hs)
def ueta(e): return e[:80],e[80:3280].reshape(40,80),e[3280:3320],e[3320:4600].reshape(32,40),e[4600:4632],e[4632:4952].reshape(10,32),e[4952:4962]
def observ(W2,e):
    b2,W3,b3,W4,b4,W5,b5=ueta(e)
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2)
    z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4)
    out=s4@W5.t()+b5; U=(W2@B.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3
    dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    return torch.cat([out.reshape(-1),dout.reshape(-1)])
target=observ(W2t,torch.cat([b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t])).detach(); vsc=float(target.abs().max())
def lsqr(Aop,Atop,b,n,damp,iters=50):
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
def opt_eta(W2,e0,iters):
    resid=lambda e:(observ(W2,e)-target)/vsc; e=e0.clone();r=resid(e);c=float(r@r);n=e.numel();damp=1e-2
    for it in range(iters):
        _,vjpf=vjp(resid,e);Jt=lambda u:vjpf(u)[0];Jv=lambda v:jvp(resid,(e,),(v,))[1];ok=False
        for _ in range(10):
            d=lsqr(Jv,Jt,-r,n,damp);en=e+d;rn=resid(en);cn=float(rn@rn)
            if cn<c:e=en;r=rn;c=cn;damp=max(damp*0.3,1e-9);ok=True;break
            damp*=4
        if not ok or c<1e-20:break
    return e,c
def wscore(A):
    W2=A@B;Cp=torch.cdist(W2,W2t);Cm=torch.cdist(-W2,W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);rel=e/nt[ci].cpu()
    return float(rel.mean()),float(rel.max()),int((rel<0.01).sum())
def varpro(A_init,outer=25):
    A=A_init.clone(); eta=eta_m0.clone()
    for it in range(outer):
        eta,cin=opt_eta(A@B,eta,40 if it==0 else 15)                  # profile eta out (warm)
        r=(observ(A@B,eta)-target)/vsc
        Jeta=jacrev(lambda e:(observ(A@B,e)-target)/vsc,chunk_size=256)(eta)   # Y x 4962
        Q,_=torch.linalg.qr(Jeta)
        fA=lambda a:(observ(a.reshape(80,80)@B,eta)-target)/vsc
        Jv=lambda v:jvp(fA,(A.reshape(-1),),(v,))[1]
        Jt=lambda u:vjp(fA,A.reshape(-1))[1](u)[0]
        Aop=lambda v:(lambda j:j-Q@(Q.t()@j))(Jv(v))
        Atop=lambda u:Jt(u-Q@(Q.t()@u))
        rp=r-Q@(Q.t()@r); base=float(rp@rp); damp=1e-3; ok=False
        for _ in range(10):
            d=lsqr(Aop,Atop,-rp,6400,damp); An=A+d.reshape(80,80)
            rn=(observ(An@B,eta)-target)/vsc; rpn=rn-Q@(Q.t()@rn); cn=float(rpn@rpn)
            if cn<base: A=An;ok=True;break
            damp*=4
        m=wscore(A)
        if it%3==0 or it==outer-1: print(f"    outer {it:2d}: W2 {m[0]:.3e}/{m[1]:.2e} <1%:{m[2]}/80  Lprof {cin:.2e}",flush=True)
        if not ok: print("    (no step)"); break
    return A
print(f"[VarPro] true B, oracle jets (value+1st). Y={target.numel()}")
for tag,W2i in [("consensus 13%",W2guess),("5% hull",None)]:
    if W2i is None:
        Wor=torch.stack(al); # oracle-hull 5%
        cand=Wor.permute(1,0,2); ones=torch.ones(8,device=dev); a0=ones/8; _,_,V8=torch.linalg.svd(ones.reshape(1,8)); Zc=V8[1:].t()
        Am=cand.transpose(1,2); AZ=Am@Zc; rhs=W2t-Am@a0; Gm=AZ.transpose(1,2)@AZ; beta=torch.linalg.solve(Gm,(AZ.transpose(1,2)@rhs.unsqueeze(-1))).squeeze(-1); W2i=torch.einsum('jdc,jc->jd',Am,a0+beta@Zc.t())
    A0=W2i@B.t(); m0=wscore(A0); print(f"  init {tag}: {m0[0]:.3e}/{m0[1]:.2e}",flush=True)
    t0=time.time(); A=varpro(A0); mf=wscore(A); print(f"  DONE {tag}: {mf[0]:.3e}/{mf[1]:.2e} <1%:{mf[2]}/80 [{time.time()-t0:.0f}s]\n",flush=True)
