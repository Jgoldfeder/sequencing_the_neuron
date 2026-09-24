"""BASIN-TRANSFER SWEEP (reviewer). Controlled ACTUAL W2 row-errors; one projected-A step; peel
idealizations one at a time to see which kills the basin. LABELED ORACLE DIAGNOSTIC: truth used to
construct controlled starts + score; the STEP machinery is sealed-equivalent per config.
Configs (cumulative): C1 true B + true eta + analytic target ; C2 +sealed FD target (warp+FD) ;
C3 +estimated B_hat ; C4 +committee eta (member-0, aligned).  [candidate jets analytic, h=q throughout]
"""
import sys, torch
from torch.func import jvp, vjp, jacrev
torch.set_default_dtype(torch.float32); dev="cuda"
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"; PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
pop=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False)
teach=MLP(pop["dims"],act="sigmoid").to(dev).float(); teach.load_state_dict(pop["teacher_state"]); teach.eval()
for p in teach.parameters(): p.requires_grad_(False)
W2t=teach.layers[1].weight.detach(); b2t=teach.layers[1].bias.detach(); nt=W2t.norm(dim=1)
W3t=teach.layers[2].weight.detach(); b3t=teach.layers[2].bias.detach(); W4t=teach.layers[3].weight.detach(); b4t=teach.layers[3].bias.detach()
W5t=teach.layers[4].weight.detach(); b5t=teach.layers[4].bias.detach()
Uw,Sw,Vh=torch.linalg.svd(W2t,full_matrices=False); Btrue=Vh[:80]
eta_true=torch.cat([b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t])
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False); mem=[{k:v.to(dev).float() for k,v in sd.items()} for sd in pk["pop_states"]]
W1c=mem[0]["layers.0.weight"]; b1c=mem[0]["layers.0.bias"]; W1cp=W1c.t()@torch.linalg.inv(W1c@W1c.t())
def xofq(Q): return (torch.log(Q/(1-Q))-b1c)@W1cp.t()
# committee member-0 eta aligned to TRUE L2 order (for C4)
def match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r,device=dev);c=torch.tensor(c,device=dev);return r,c,torch.where(Cm[r.cpu(),c.cpu()]<Cp[r.cpu(),c.cpu()],-1.,1.).to(dev)
W2m=mem[0]["layers.1.weight"]; r,c,s=match(W2m,W2t)
b2a=torch.zeros_like(b2t); b2a[c]=mem[0]["layers.1.bias"][r]*s
W3a=torch.zeros_like(W3t); W3a[:,c]=mem[0]["layers.2.weight"][:,r]*s[None,:]; b3a=mem[0]["layers.2.bias"]+(mem[0]["layers.2.weight"][:,r][:,s<0]).sum(1)
eta_m0=torch.cat([b2a,W3a.reshape(-1),b3a,mem[0]["layers.3.weight"].reshape(-1),mem[0]["layers.3.bias"],mem[0]["layers.4.weight"].reshape(-1),mem[0]["layers.4.bias"]])
gg=torch.Generator(device=dev).manual_seed(0); H=torch.sigmoid(torch.randn(40,128,generator=gg,device=dev)*1.8).clamp(1e-3,1-1e-3)
# sealed B_hat (FD of f(x(q)) in committee-L1 coordinate)
def recoverB(fd=3e-4):
    gq=torch.Generator(device=dev).manual_seed(5); W2gp=torch.linalg.pinv(W2t); rows=[]  # note: probe placement can use truth here (diagnostic); geometry only
    for rep in range(2):
        for j in range(80):
            tt=(torch.randn(80,generator=gq,device=dev)*2.0); tt[j]=0.0; q0=torch.clamp(W2gp@(tt-b2t),1e-3,1-1e-3); E=torch.eye(128,device=dev)
            rows.append(((teach(xofq((q0+fd*E).clamp(1e-4,1-1e-4)))-teach(xofq((q0-fd*E).clamp(1e-4,1-1e-4))))/(2*fd)).t())
    M=torch.cat(rows,0); U,S,Vh2=torch.linalg.svd(M,full_matrices=False); return Vh2[:80]
Bhat=recoverB(); print(f"[Bhat] ||W2t-W2t P||/||W2t|| mean {float(((W2t-W2t@(Bhat.t()@Bhat)).norm(dim=1)/nt).mean()):.3e}",flush=True)
def ueta(e): return e[:80],e[80:3280].reshape(40,80),e[3280:3320],e[3320:4600].reshape(32,40),e[4600:4632],e[4632:4952].reshape(10,32),e[4952:4962]
def observ(A,e,Bu):
    b2,W3,b3,W4,b4,W5,b5=ueta(e); W2=A@Bu
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2)
    z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4)
    out=s4@W5.t()+b5; U=(W2@Bu.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3
    dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    return torch.cat([out.reshape(-1),dout.reshape(-1)])
def target_analytic(Bu): return observ(W2t@Bu.t(),eta_true,Bu).detach()
def target_sealed(Bu,fd=3e-4):
    val=teach(xofq(H)); jt=[]
    for p in range(H.shape[0]):
        jt.append((teach(xofq((H[p][None,:]+fd*Bu).clamp(1e-4,1-1e-4)))-teach(xofq((H[p][None,:]-fd*Bu).clamp(1e-4,1-1e-4))))/(2*fd))
    return torch.cat([val.reshape(-1),torch.stack(jt).reshape(-1)]).detach()
def lsqr(Aop,Atop,b,n,damp,iters=70):
    beta=b.norm()
    if float(beta)==0: return torch.zeros(n,device=b.device)
    u=b/beta;v=Atop(u);alfa=v.norm();v=v/alfa.clamp_min(1e-30);w=v.clone();x=torch.zeros(n,device=b.device);phibar=beta;rhobar=alfa
    for _ in range(iters):
        u=Aop(v)-alfa*u;beta=u.norm();u=u/beta.clamp_min(1e-30)
        v=Atop(u)-beta*v;alfa=v.norm();v=v/alfa.clamp_min(1e-30)
        rb1=(rhobar**2+damp**2).sqrt();c1=rhobar/rb1;phibar=c1*phibar
        rho=(rb1**2+beta**2).sqrt();cc=rb1/rho;s=beta/rho;theta=s*alfa;rhobar=-cc*alfa;phi=cc*phibar;phibar=s*phibar
        x=x+(phi/rho)*w;w=v-(theta/rho)*w
    return x
def werr(A,Bu):
    W2=A@Bu;Cp=torch.cdist(W2,W2t);Cm=torch.cdist(-W2,W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);return float((e/nt[ci].cpu()).mean())
def one_step(A,e,Bu,target):
    r=(observ(A,e,Bu)-target); sc=float(target.abs().max()); r=r/sc
    Jn=jacrev(lambda ee:(observ(A,ee,Bu)-target)/sc,chunk_size=256)(e); Q,_=torch.linalg.qr(Jn)
    fA=lambda a:(observ(a.reshape(80,80),e,Bu)-target)/sc
    Jv=lambda v:jvp(fA,(A.reshape(-1),),(v,))[1]; Jt=lambda u:vjp(fA,A.reshape(-1))[1](u)[0]
    Aop=lambda v:(lambda j:j-Q@(Q.t()@j))(Jv(v)); Atop=lambda u:Jt(u-Q@(Q.t()@u))
    rp=r-Q@(Q.t()@r); base=float(rp@rp); damp=1e-4
    for _ in range(12):
        d=lsqr(Aop,Atop,-rp,6400,damp); An=A+d.reshape(80,80); rn=(observ(An,e,Bu)-target)/sc; rpn=rn-Q@(Q.t()@rn)
        if float(rpn@rpn)<base: return An
        damp*=4
    return A
def make_A(Bu,err,seed):
    A0=W2t@Bu.t(); gp=torch.Generator(device=dev).manual_seed(seed); noise=torch.randn(80,80,generator=gp,device=dev)
    An=A0+noise; cur=werr(An,Bu); s=err/cur; return A0+s*noise
configs=[("C1 trueB+trueEta+analytic",Btrue,eta_true,"analytic"),
         ("C2 +sealed target",Btrue,eta_true,"sealed"),
         ("C3 +estimated Bhat",Bhat,eta_true,"sealed"),
         ("C4 +committee eta",Bhat,eta_m0,"sealed")]
errs=[0.02,0.04,0.06,0.08,0.12,0.16]
for name,Bu,e_use,tk in configs:
    tgt=target_analytic(Bu) if tk=="analytic" else target_sealed(Bu)
    row=[]
    for er in errs:
        A0=make_A(Bu,er,int(er*1000)); A1=one_step(A0,e_use,Bu,tgt); row.append((werr(A0,Bu),werr(A1,Bu)))
    print(f"{name:32s} " + "  ".join(f"{er*100:.0f}%:{b*100:.1f}->{a*100:.1f}" for er,(b,a) in zip(errs,row)),flush=True)
