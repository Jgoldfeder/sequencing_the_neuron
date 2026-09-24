"""CORRECTED adversarial experiment. BUG FIXED: obs takes an explicit R; target uses R=0 so W2=W2*
(not W2*+R_m). Compensation decomposition and step residual now use the true teacher observation.
Probe sets: ordinary, sampled-adversarial (max D_m0), optimized-adversarial (grad-ascent D_m0).
LABEL: diagnostic, NOT physically sealed (teacher forward in-process; probe opt autograds teacher).
The SELECTION criterion is seal-compatible (forward queries only); optimized version would need FD/SPSA."""
import sys, torch
from torch.func import jvp, vjp, jacrev
torch.set_default_dtype(torch.float32); dev="cuda"
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"; PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
pop=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False); dims=pop["dims"]
bb=MLP(dims,act="sigmoid").to(dev).float(); bb.load_state_dict(pop["teacher_state"]); bb.eval()
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False)
m0net=MLP(dims,act="sigmoid").to(dev).float(); m0net.load_state_dict({k:v.to(dev).float() for k,v in pk["pop_states"][0].items()}); m0net.eval()
mem=[{k:v.to(dev).float() for k,v in sd.items()} for sd in pk["pop_states"]]
W1c=mem[0]["layers.0.weight"]; b1c=mem[0]["layers.0.bias"]; W1cp=W1c.t()@torch.linalg.inv(W1c@W1c.t())
def xofz(Z): return (Z-b1c)@W1cp.t()
g=torch.Generator(device=dev).manual_seed(1); pool=torch.cat([torch.sigmoid(torch.randn(3000,128,generator=g,device=dev)*s).clamp(1e-4,1-1e-4) for s in [0.5,1,2,4,8]])
with torch.no_grad(): D0=(m0net(xofz(torch.log(pool/(1-pool))))-bb(xofz(torch.log(pool/(1-pool))))).norm(dim=1)
NP=60; H_adv=pool[torch.topk(D0,NP).indices]
gm=torch.Generator(device=dev).manual_seed(9); H_ord=torch.sigmoid(torch.randn(NP,128,generator=gm,device=dev)*1.8).clamp(1e-4,1-1e-4)
# optimized probes
Z=torch.log(H_adv/(1-H_adv)).clone().requires_grad_(True); opt=torch.optim.Adam([Z],lr=0.2)
for it in range(150):
    opt.zero_grad(); x=xofz(Z.clamp(-11,11)); (-((m0net(x)-bb(x)).norm(dim=1)**2).sum()).backward(); opt.step()
with torch.no_grad(): H_opt=torch.sigmoid(Z.clamp(-11,11)); Dopt=(m0net(xofz(Z.clamp(-11,11)))-bb(xofz(Z.clamp(-11,11)))).norm(dim=1)
print(f"D_m0 mean: ordinary-seed {float(D0.topk(NP).values.mean()):.3e} -> optimized {float(Dopt.mean()):.3e}")
# truth + member0 aligned
W2t=bb.layers[1].weight.detach(); b2t=bb.layers[1].bias.detach(); W3t=bb.layers[2].weight.detach(); b3t=bb.layers[2].bias.detach()
W4t=bb.layers[3].weight.detach(); b4t=bb.layers[3].bias.detach(); W5t=bb.layers[4].weight.detach(); b5t=bb.layers[4].bias.detach(); nt=W2t.norm(dim=1)
Uw,Sw,Vh=torch.linalg.svd(W2t,full_matrices=False); Bb=Vh[:80]; eta_true=torch.cat([b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t])
def match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r,device=dev);c=torch.tensor(c,device=dev);return r,c,torch.where(Cm[r.cpu(),c.cpu()]<Cp[r.cpu(),c.cpu()],-1.,1.).to(dev)
sd0=mem[0]; W2m=sd0["layers.1.weight"]; r,c,s=match(W2m,W2t); W2al=torch.zeros_like(W2t); W2al[c]=W2m[r]*s[:,None]
b2a=torch.zeros_like(b2t); b2a[c]=sd0["layers.1.bias"][r]*s; W3a=torch.zeros_like(W3t); W3a[:,c]=sd0["layers.2.weight"][:,r]*s[None,:]; b3a=sd0["layers.2.bias"]+(sd0["layers.2.weight"][:,r][:,s<0]).sum(1)
em0=torch.cat([b2a,W3a.reshape(-1),b3a,sd0["layers.3.weight"].reshape(-1),sd0["layers.3.bias"],sd0["layers.4.weight"].reshape(-1),sd0["layers.4.bias"]])
Rm=W2al-(W2al@Bb.t())@Bb; A0=W2al@Bb.t(); Astar=W2t@Bb.t(); ZeroR=torch.zeros_like(W2t)
def ueta(e): return e[:80],e[80:3280].reshape(40,80),e[3280:3320],e[3320:4600].reshape(32,40),e[4600:4632],e[4632:4952].reshape(10,32),e[4952:4962]
def obs(A,e,H,R):                       # W2 = R + A@Bb  (R explicit -- BUG FIX)
    b2,W3,b3,W4,b4,W5,b5=ueta(e); W2=R+A@Bb
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2); z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4); out=s4@W5.t()+b5; U=(W2@Bb.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3; dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    return torch.cat([out.reshape(-1),dout.reshape(-1)])
def comp(H):
    oWt=obs(A0,eta_true,H,Rm); oWm=obs(A0,em0,H,Rm); oT=obs(Astar,eta_true,H,ZeroR)   # W2* target R=0
    rW=oWt-oT; re=oWm-oWt; S=float(rW.norm()); N=float(re.norm()); rt=float((oWm-oT).norm())
    return N/S, rt/S, float((re@rW)/(N*S+1e-30))
def lsqr(Aop,Atop,b,n,damp,iters=70):
    beta=b.norm()
    if float(beta)==0: return torch.zeros(n,device=b.device)
    u=b/beta;v=Atop(u);alfa=v.norm();v=v/alfa.clamp_min(1e-30);w=v.clone();x=torch.zeros(n,device=b.device);phibar=beta;rhobar=alfa
    for _ in range(iters):
        u=Aop(v)-alfa*u;beta=u.norm();u=u/beta.clamp_min(1e-30); v=Atop(u)-beta*v;alfa=v.norm();v=v/alfa.clamp_min(1e-30)
        rb1=(rhobar**2+damp**2).sqrt();c1=rhobar/rb1;phibar=c1*phibar; rho=(rb1**2+beta**2).sqrt();cc=rb1/rho;sg=beta/rho;th=sg*alfa;rhobar=-cc*alfa;phi=cc*phibar;phibar=sg*phibar
        x=x+(phi/rho)*w;w=v-(th/rho)*w
    return x
def werr(A):
    W2=Rm+A@Bb;Cp=torch.cdist(W2,W2t);Cm=torch.cdist(-W2,W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);return float((e/nt[ci].cpu()).mean())
def step(H):
    target=obs(Astar,eta_true,H,ZeroR).detach(); sc=float(target.abs().max()); r=(obs(A0,em0,H,Rm)-target)/sc
    Jn=jacrev(lambda ee:(obs(A0,ee,H,Rm)-target)/sc,chunk_size=256)(em0); Q,_=torch.linalg.qr(Jn)
    fA=lambda a:(obs(a.reshape(80,80),em0,H,Rm)-target)/sc
    Jv=lambda v:jvp(fA,(A0.reshape(-1),),(v,))[1]; Jt=lambda u:vjp(fA,A0.reshape(-1))[1](u)[0]
    Aop=lambda v:(lambda j:j-Q@(Q.t()@j))(Jv(v)); Atop=lambda u:Jt(u-Q@(Q.t()@u)); rp=r-Q@(Q.t()@r); d=lsqr(Aop,Atop,-rp,6400,1e-3).reshape(80,80)
    tv=(Astar-A0).reshape(-1); cos=float((d.reshape(-1)@tv)/(d.norm()*tv.norm()+1e-30))
    return cos,[werr(A0+l*d)*100 for l in (0.03,0.1,0.3,1.0)]
print(f"start W2 {werr(A0)*100:.1f}%. CORRECTED target (R=0=W2*).")
for tag,H in [("ordinary",H_ord),("sampled-adv",H_adv),("optimized-adv",H_opt)]:
    ns,ts,cs=comp(H); co,ws=step(H)
    print(f"  {tag:14s} comp: N/S {ns:.2f} T/S {ts:.3f} cos {cs:+.3f} | step cos {co:+.3f} W2@l: " + " ".join(f"{w:.1f}" for w in ws),flush=True)
