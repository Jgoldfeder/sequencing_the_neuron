"""ETA-QUALITY sweep (labeled oracle diagnostic). Fix W2 at a controlled error; vary the downstream eta
= true + controlled relative perturbation; take ONE projected-A step against the TRUE function; find the
eta-error threshold where the step flips from reducing W2 to diverging. Tells us how good eta must be.
true B, analytic true-function target (isolates eta)."""
import sys, torch
from torch.func import jvp, vjp, jacrev
torch.set_default_dtype(torch.float32); dev="cuda"
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
pop=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False)
t=MLP(pop["dims"],act="sigmoid").to(dev).float(); t.load_state_dict(pop["teacher_state"]); t.eval()
W2t=t.layers[1].weight.detach(); b2t=t.layers[1].bias.detach(); nt=W2t.norm(dim=1)
W3t=t.layers[2].weight.detach(); b3t=t.layers[2].bias.detach(); W4t=t.layers[3].weight.detach(); b4t=t.layers[3].bias.detach()
W5t=t.layers[4].weight.detach(); b5t=t.layers[4].bias.detach()
Uw,Sw,Vh=torch.linalg.svd(W2t,full_matrices=False); B=Vh[:80]
eta_true=torch.cat([b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t]); ne=float(eta_true.norm())
gg=torch.Generator(device=dev).manual_seed(0); H=torch.sigmoid(torch.randn(40,128,generator=gg,device=dev)*1.8).clamp(1e-3,1-1e-3)
def ueta(e): return e[:80],e[80:3280].reshape(40,80),e[3280:3320],e[3320:4600].reshape(32,40),e[4600:4632],e[4632:4952].reshape(10,32),e[4952:4962]
def observ(A,e):
    b2,W3,b3,W4,b4,W5,b5=ueta(e); W2=A@B
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2)
    z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4)
    out=s4@W5.t()+b5; U=(W2@B.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3
    dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    return torch.cat([out.reshape(-1),dout.reshape(-1)])
target=observ(W2t@B.t(),eta_true).detach()   # TRUE function
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
def werr(A):
    W2=A@B;Cp=torch.cdist(W2,W2t);Cm=torch.cdist(-W2,W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);return float((e/nt[ci].cpu()).mean())
def step(A,e):
    r=(observ(A,e)-target); sc=float(target.abs().max()); r=r/sc
    Jn=jacrev(lambda ee:(observ(A,ee)-target)/sc,chunk_size=256)(e); Q,_=torch.linalg.qr(Jn)
    fA=lambda a:(observ(a.reshape(80,80),e)-target)/sc
    Jv=lambda v:jvp(fA,(A.reshape(-1),),(v,))[1]; Jt=lambda u:vjp(fA,A.reshape(-1))[1](u)[0]
    Aop=lambda v:(lambda j:j-Q@(Q.t()@j))(Jv(v)); Atop=lambda u:Jt(u-Q@(Q.t()@u))
    rp=r-Q@(Q.t()@r); base=float(rp@rp); damp=1e-4
    for _ in range(12):
        d=lsqr(Aop,Atop,-rp,6400,damp); An=A+d.reshape(80,80); rn=(observ(An,e)-target)/sc; rpn=rn-Q@(Q.t()@rn)
        if float(rpn@rpn)<base: return An
        damp*=4
    return A
def make_A(err,seed):
    A0=W2t@B.t(); gp=torch.Generator(device=dev).manual_seed(seed); nz=torch.randn(80,80,generator=gp,device=dev)
    return A0+(err/werr(A0+nz))*nz
def make_eta(rel,seed):
    gp=torch.Generator(device=dev).manual_seed(seed); nz=torch.randn(eta_true.numel(),generator=gp,device=dev)
    ep=eta_true+nz; return eta_true+(rel*ne/float((ep-eta_true).norm()))*(ep-eta_true)
print("one projected-A step vs ETA quality (true function target, true B). W2 error before->after.")
for w2e in [0.04,0.08]:
    A0=make_A(w2e,7); print(f"  W2 start {werr(A0)*100:.1f}%:",flush=True)
    for rel in [0.0,0.005,0.01,0.02,0.05,0.10,0.20]:
        e=eta_true.clone() if rel==0 else make_eta(rel,int(rel*1000)+1)
        A1=step(A0.clone(),e); print(f"    eta err {rel*100:5.1f}% -> W2 {werr(A0)*100:.1f} -> {werr(A1)*100:.1f}",flush=True)
