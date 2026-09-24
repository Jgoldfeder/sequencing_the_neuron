"""PARTIAL-ORACLE test (user's idea): freeze the WORST K W2 rows at truth ('pretend they're solved'),
then JOINTLY refine the remaining (80-K) rows + the (wrong) downstream against oracle jets. Does
anchoring some rows break the compensation valley so the rest solve? Sweep K.
true B + oracle jets + member-0 downstream init (wrong, refined jointly). Truth only for freeze+scoring.
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
# truth-aligned committee -> consensus guess (true order) + member-0 downstream (true order)
al=[]
for sd in mem:
    W2m=sd["layers.1.weight"];r,c,s=match(W2m,W2t);Wa=torch.zeros_like(W2m);Wa[c]=W2m[r]*s[:,None];al.append(Wa)
W2guess=torch.stack(al).median(0).values                       # ~13% (oracle corr)
def m0_true():
    W2m=mem[0]["layers.1.weight"];b2m=mem[0]["layers.1.bias"];W3m=mem[0]["layers.2.weight"];b3m=mem[0]["layers.2.bias"]
    r,c,s=match(W2m,W2t);b2a=torch.zeros_like(b2m);b2a[c]=b2m[r]*s;W3a=torch.zeros_like(W3m);W3a[:,c]=W3m[:,r]*s[None,:];b3a=b3m+(W3m[:,r][:,s<0]).sum(1)
    return [b2a,W3a,b3a,mem[0]["layers.3.weight"],mem[0]["layers.3.bias"],mem[0]["layers.4.weight"],mem[0]["layers.4.bias"]]
eta0=torch.cat([x.reshape(-1) for x in m0_true()])
A_true=W2t@B.t(); A_guess=W2guess@B.t()
rowerr_guess=(W2guess-W2t).norm(dim=1)/nt
order=torch.argsort(rowerr_guess,descending=True)              # worst rows first
# probes + observ
W2gp=torch.linalg.pinv(W2guess); gg=torch.Generator(device=dev).manual_seed(0); Hs=[]
for j in range(60):
    tt=torch.randn(80,generator=gg,device=dev)*2.0; tt[j%80]=0.0; Hs.append(torch.clamp(W2gp@(tt-b2t),1e-3,1-1e-3))
H=torch.stack(Hs)
def fwd(A,eta):
    b2,W3,b3,W4,b4,W5,b5=eta[:80],eta[80:80+3200].reshape(40,80),eta[3280:3320],eta[3320:3320+1280].reshape(32,40),eta[4600:4632],eta[4632:4632+320].reshape(10,32),eta[4952:4962]
    W2=A@B; z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2)
    z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4)
    out=s4@W5.t()+b5; U=(W2@B.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3
    dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    return torch.cat([out.reshape(-1),dout.reshape(-1)])
target=fwd(A_true,torch.cat([b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t])).detach(); vsc=float(target.abs().max())
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
def solveK(K,iters=45):
    frozen=order[:K]; free=order[K:]
    Af=A_guess[free].reshape(-1).clone()
    def build(th):
        A=A_true.clone(); A[free]=th[:free.numel()*80].reshape(-1,80); return A, th[free.numel()*80:]
    def resid(th):
        A,eta=build(th); return (fwd(A,eta)-target)/vsc
    th=torch.cat([Af,eta0.clone()]); n=th.numel()
    r=resid(th); c=float(r@r); damp=1e-2
    for it in range(iters):
        _,vjpf=vjp(resid,th); Jt=lambda u:vjpf(u)[0]; Jv=lambda v:jvp(resid,(th,),(v,))[1]; ok=False
        for _ in range(12):
            d=lsqr(Jv,Jt,-r,n,damp); tn=th+d; rn=resid(tn); cn=float(rn@rn)
            if cn<c: th=tn;r=rn;c=cn;damp=max(damp*0.3,1e-9);ok=True;break
            damp*=4
        if not ok or c<1e-16: break
    A,_=build(th); rel=(A[free]@B-W2t[free]).norm(dim=1)/nt[free]
    return float(rel.mean()),float(rel.max()),int((rel<0.01).sum()),free.numel(),c
print("[partial] freeze worst-K W2 rows at truth; joint-refine the rest + downstream. true B, oracle jets.")
print(f"{'K frozen':>9} {'free rows':>9} {'free-row err mean':>18} {'max':>8} {'<1%':>8} {'loss':>10}")
for K in [0,11,20,40,60,70]:
    m,mx,u,nf,c=solveK(K); print(f"{K:>9} {nf:>9} {m:>18.3e} {mx:>8.2e} {u:>5}/{nf:<3} {c:>10.2e}",flush=True)
