"""PAIRED-PATH compensation test (reviewer). For truth-aligned member m: dW=W2^m-W2*, deta=eta^m-eta*.
Walk W2(a)=W2*+a dW, eta(a)=eta*+a deta at controlled W2 errors. Measure:
  L_W  = ||Phi(W2(a),eta*)-Y*||   (only W2 wrong)
  L_eta= ||Phi(W2*,eta(a))-Y*||   (only eta wrong)
  L_pair=||Phi(W2(a),eta(a))-Y*|| (both, correlated)
Compensation <=> L_pair << L_W,L_eta. Then projected-A step along the paired path; report
cos(theta) between the step dA and (A*-A(a)) (>0 truthward). Also compare committee-eta vs RANDOM eta
matched by OBSERVABLE displacement ||Phi(A,eta*+deta)-Phi(A,eta*)||. Rowspace-only; true B; true-fn target."""
import sys, torch
from torch.func import jvp, vjp, jacrev
torch.set_default_dtype(torch.float32); dev="cuda"
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"; PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
pop=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False)
t=MLP(pop["dims"],act="sigmoid").to(dev).float(); t.load_state_dict(pop["teacher_state"]); t.eval()
W2t=t.layers[1].weight.detach(); b2t=t.layers[1].bias.detach(); nt=W2t.norm(dim=1)
W3t=t.layers[2].weight.detach(); b3t=t.layers[2].bias.detach(); W4t=t.layers[3].weight.detach(); b4t=t.layers[3].bias.detach()
W5t=t.layers[4].weight.detach(); b5t=t.layers[4].bias.detach()
Uw,Sw,Vh=torch.linalg.svd(W2t,full_matrices=False); B=Vh[:80]; Astar=W2t@B.t()
eta_true=torch.cat([b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t])
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False); mem=[{k:v.to(dev).float() for k,v in sd.items()} for sd in pk["pop_states"]]
def match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r,device=dev);c=torch.tensor(c,device=dev);return r,c,torch.where(Cm[r.cpu(),c.cpu()]<Cp[r.cpu(),c.cpu()],-1.,1.).to(dev)
sd=mem[0]; W2m=sd["layers.1.weight"]; r,c,s=match(W2m,W2t)
W2m_al=torch.zeros_like(W2t); W2m_al[c]=W2m[r]*s[:,None]
b2a=torch.zeros_like(b2t); b2a[c]=sd["layers.1.bias"][r]*s
W3a=torch.zeros_like(W3t); W3a[:,c]=sd["layers.2.weight"][:,r]*s[None,:]; b3a=sd["layers.2.bias"]+(sd["layers.2.weight"][:,r][:,s<0]).sum(1)
eta_m=torch.cat([b2a,W3a.reshape(-1),b3a,sd["layers.3.weight"].reshape(-1),sd["layers.3.bias"],sd["layers.4.weight"].reshape(-1),sd["layers.4.bias"]])
Am=W2m_al@B.t(); dA=Am-Astar; deta=eta_m-eta_true
gg=torch.Generator(device=dev).manual_seed(0); H=torch.sigmoid(torch.randn(40,128,generator=gg,device=dev)*1.8).clamp(1e-3,1-1e-3)
def ueta(e): return e[:80],e[80:3280].reshape(40,80),e[3280:3320],e[3320:4600].reshape(32,40),e[4600:4632],e[4632:4952].reshape(10,32),e[4952:4962]
def obs(A,e):
    b2,W3,b3,W4,b4,W5,b5=ueta(e); W2=A@B
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2)
    z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4)
    out=s4@W5.t()+b5; U=(W2@B.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3
    dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    return torch.cat([out.reshape(-1),dout.reshape(-1)])
target=obs(Astar,eta_true).detach(); tn=float(target.norm())
def rms(v): return float(v.norm())/tn
def werr(A):
    W2=A@B;Cp=torch.cdist(W2,W2t);Cm=torch.cdist(-W2,W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);return float((e/nt[ci].cpu()).mean())
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
def stepdir(A,e):
    sc=float(target.abs().max()); r=(obs(A,e)-target)/sc
    Jn=jacrev(lambda ee:(obs(A,ee)-target)/sc,chunk_size=256)(e); Q,_=torch.linalg.qr(Jn)
    fA=lambda a:(obs(a.reshape(80,80),e)-target)/sc
    Jv=lambda v:jvp(fA,(A.reshape(-1),),(v,))[1]; Jt=lambda u:vjp(fA,A.reshape(-1))[1](u)[0]
    Aop=lambda v:(lambda j:j-Q@(Q.t()@j))(Jv(v)); Atop=lambda u:Jt(u-Q@(Q.t()@u))
    rp=r-Q@(Q.t()@r); damp=1e-3; d=lsqr(Aop,Atop,-rp,6400,damp); return d
werr_m=werr(Am)
print(f"member-0 rowspace W2 err (alpha=1) = {werr_m*100:.1f}%")
print(f"{'W2err':>6} {'L_W':>9} {'L_eta':>9} {'L_pair':>9} {'cos(step,truth)':>16}")
for we in [0.02,0.04,0.08,0.12,0.16]:
    al=we/werr_m; Aa=Astar+al*dA; ea=eta_true+al*deta
    LW=rms(obs(Aa,eta_true)-target); Le=rms(obs(Astar,ea)-target); Lp=rms(obs(Aa,ea)-target)
    d=stepdir(Aa,ea); tv=(Astar-Aa).reshape(-1); cos=float((d@tv)/(d.norm()*tv.norm()+1e-30))
    print(f"{we*100:>5.0f}% {LW:>9.2e} {Le:>9.2e} {Lp:>9.2e} {cos:>16.3f}",flush=True)
# matched-random comparison at 8% W2 error
al=0.08/werr_m; Aa=Astar+al*dA
disp_c=float((obs(Aa,eta_true+al*deta)-obs(Aa,eta_true)).norm())
gr=torch.Generator(device=dev).manual_seed(3); cosr=[]
for k in range(4):
    dr=torch.randn(deta.numel(),generator=gr,device=dev)
    dd=float((obs(Aa,eta_true+dr)-obs(Aa,eta_true)).norm()); dr=dr*(disp_c/dd)
    d=stepdir(Aa,eta_true+dr); tv=(Astar-Aa).reshape(-1); cosr.append(float((d@tv)/(d.norm()*tv.norm()+1e-30)))
d=stepdir(Aa,eta_true+al*deta); tv=(Astar-Aa).reshape(-1); cosc=float((d@tv)/(d.norm()*tv.norm()+1e-30))
print(f"\n@8% W2: cos(step,truth)  committee-eta {cosc:.3f}  vs  random-eta[matched obs-disp] {sum(cosr)/len(cosr):.3f} (min {min(cosr):.3f} max {max(cosr):.3f})")
