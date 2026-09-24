"""ORACLE MACHINERY CHECK (labeled diagnostic, NOT a sealed solver claim). Pure test of the projected-A
step: true B, true eta, analytic internal target (no warp, no seal). Init A = A*+noise; one projected-A
step quotienting the true-eta tangent. Does W2 error DROP? If yes -> machinery correct, far-init failure
is basin/curvature. If no -> the projected step itself is buggy."""
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
Uw,Sw,Vh=torch.linalg.svd(W2t,full_matrices=False); B=Vh[:80]; A_true=W2t@B.t()
eta=torch.cat([b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t])
gg=torch.Generator(device=dev).manual_seed(0); H=torch.sigmoid(torch.randn(40,128,generator=gg,device=dev)*1.5).clamp(1e-3,1-1e-3)
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
target=observ(A_true,eta).detach(); vsc=float(target.abs().max())
def lsqr(Aop,Atop,b,n,damp,iters=80):
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
def proj_iter(A,iters):
    for _ in range(iters):
        r=(observ(A,eta)-target)/vsc
        Jn=jacrev(lambda e:(observ(A,e)-target)/vsc,chunk_size=256)(eta); Q,_=torch.linalg.qr(Jn)
        fA=lambda a:(observ(a.reshape(80,80),eta)-target)/vsc
        Jv=lambda v:jvp(fA,(A.reshape(-1),),(v,))[1]; Jt=lambda u:vjp(fA,A.reshape(-1))[1](u)[0]
        Aop=lambda v:(lambda j:j-Q@(Q.t()@j))(Jv(v)); Atop=lambda u:Jt(u-Q@(Q.t()@u))
        rp=r-Q@(Q.t()@r); base=float(rp@rp); damp=1e-4; ok=False
        for _ in range(12):
            d=lsqr(Aop,Atop,-rp,6400,damp); An=A+d.reshape(80,80); rn=(observ(An,eta)-target)/vsc; rpn=rn-Q@(Q.t()@rn)
            if float(rpn@rpn)<base: A=An;ok=True;break
            damp*=4
        if not ok: break
    return A
for pert in [0.03,0.08]:
    gp=torch.Generator(device=dev).manual_seed(int(pert*1000))
    A0=A_true+pert*A_true.abs().mean()*torch.randn(80,80,generator=gp,device=dev)
    A1=proj_iter(A0.clone(),1); A5=proj_iter(A0.clone(),5)
    print(f"pert {pert:.2f}: init {werr(A0):.3e} -> 1 step {werr(A1):.3e} -> 5 steps {werr(A5):.3e}",flush=True)
