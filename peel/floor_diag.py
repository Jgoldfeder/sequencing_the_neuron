"""FLOOR causal isolation (reviewer). At the 22% sealed endpoint A_floor, ORACLE diagnostic: compare
projected-A directions using member-0 eta_m vs true eta* vs a non-compensating control eta. Report
cos(d, A*-A_floor) and small-step truth-error curve. Also compensation geometry at A_floor
(N/S, T/S, cos(r_eta,r_W)) vs the original member point (cos~-1). Isolates whether frozen eta causes the floor."""
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
Uw,Sw,Vh=torch.linalg.svd(W2t,full_matrices=False); B=Vh[:80]; eta_true=torch.cat([b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t])
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False); mem=[{k:v.to(dev).float() for k,v in sd.items()} for sd in pk["pop_states"]]
def match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r,device=dev);c=torch.tensor(c,device=dev);return r,c,torch.where(Cm[r.cpu(),c.cpu()]<Cp[r.cpu(),c.cpu()],-1.,1.).to(dev)
sd0=mem[0]; W2m=sd0["layers.1.weight"]; r,c,s=match(W2m,W2t)
b2a=torch.zeros_like(b2t); b2a[c]=sd0["layers.1.bias"][r]*s; W3a=torch.zeros_like(W3t); W3a[:,c]=sd0["layers.2.weight"][:,r]*s[None,:]; b3a=sd0["layers.2.bias"]+(sd0["layers.2.weight"][:,r][:,s<0]).sum(1)
eta_m=torch.cat([b2a,W3a.reshape(-1),b3a,sd0["layers.3.weight"].reshape(-1),sd0["layers.3.bias"],sd0["layers.4.weight"].reshape(-1),sd0["layers.4.bias"]])
# floor W2 from the sealed run
def align_W2(W):   # member-0 gauge -> teacher gauge, SAME (r,c,s) as eta_m
    Wa=torch.zeros_like(W); Wa[c]=W[r]*s[:,None]; return Wa
tr=torch.load(PEEL+"iter_out.pt",map_location=dev,weights_only=False)["traj"]
W2_floor=align_W2(tr[-1].to(dev).float()); W2_start=align_W2(tr[0].to(dev).float())
def AR(W2): A=W2@B.t(); return A, W2-(A@B)   # A, R (out-of-rowspace)
Afl,Rfl=AR(W2_floor); Ast_start,Rst=AR(W2_start); Astar=W2t@B.t()
gg=torch.Generator(device=dev).manual_seed(0); H=torch.sigmoid(torch.randn(60,128,generator=gg,device=dev)*1.8).clamp(1e-3,1-1e-3)
def ueta(e): return e[:80],e[80:3280].reshape(40,80),e[3280:3320],e[3320:4600].reshape(32,40),e[4600:4632],e[4632:4952].reshape(10,32),e[4952:4962]
def obs(A,e,R):
    b2,W3,b3,W4,b4,W5,b5=ueta(e); W2=R+A@B
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2); z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4); out=s4@W5.t()+b5; U=(W2@B.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3; dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    return torch.cat([out.reshape(-1),dout.reshape(-1)])
Y=obs(Astar,eta_true,torch.zeros_like(W2t)).detach()
def lsqr(Aop,Atop,b,n,damp,iters=70):
    beta=b.norm()
    if float(beta)==0: return torch.zeros(n,device=b.device)
    u=b/beta;v=Atop(u);alfa=v.norm();v=v/alfa.clamp_min(1e-30);w=v.clone();x=torch.zeros(n,device=b.device);phibar=beta;rhobar=alfa
    for _ in range(iters):
        u=Aop(v)-alfa*u;beta=u.norm();u=u/beta.clamp_min(1e-30); v=Atop(u)-beta*v;alfa=v.norm();v=v/alfa.clamp_min(1e-30)
        rb1=(rhobar**2+damp**2).sqrt();c1=rhobar/rb1;phibar=c1*phibar; rho=(rb1**2+beta**2).sqrt();cc=rb1/rho;sg=beta/rho;th=sg*alfa;rhobar=-cc*alfa;phi=cc*phibar;phibar=sg*phibar
        x=x+(phi/rho)*w;w=v-(th/rho)*w
    return x
def werr(A,R):
    W2=R+A@B;Cp=torch.cdist(W2,W2t);Cm=torch.cdist(-W2,W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);return float((e/nt[ci].cpu()).mean())
def direction(A,R,e):
    sc=float(Y.abs().max()); r=(obs(A,e,R)-Y)/sc
    Jn=jacrev(lambda ee:(obs(A,ee,R)-Y)/sc,chunk_size=256)(e); Q,_=torch.linalg.qr(Jn)
    fA=lambda a:(obs(a.reshape(80,80),e,R)-Y)/sc
    Jv=lambda v:jvp(fA,(A.reshape(-1),),(v,))[1]; Jt=lambda u:vjp(fA,A.reshape(-1))[1](u)[0]
    Aop=lambda v:(lambda j:j-Q@(Q.t()@j))(Jv(v)); Atop=lambda u:Jt(u-Q@(Q.t()@u)); rp=r-Q@(Q.t()@r)
    return lsqr(Aop,Atop,-rp,6400,1e-3).reshape(80,80)
# compensation geometry at floor
def comp(A,R,e):
    rW=obs(A,eta_true,R)-Y; re=obs(A,e,R)-obs(A,eta_true,R); S=float(rW.norm()); N=float(re.norm())
    return N/S, float((obs(A,e,R)-Y).norm())/S, float((re@rW)/(N*S+1e-30))
gr=torch.Generator(device=dev).manual_seed(3); dr=torch.randn(eta_true.numel(),generator=gr,device=dev); dr=dr/dr.norm()
Dm=float((obs(Afl,eta_m,Rfl)-obs(Afl,eta_true,Rfl)).norm())
def dispr(gv): return float((obs(Afl,eta_true+gv*dr,Rfl)-obs(Afl,eta_true,Rfl)).norm())
hi=1.0
while dispr(hi)<Dm and hi<1e8: hi*=2
lo=0.0
for _ in range(34):
    mid=0.5*(lo+hi); lo,hi=(mid,hi) if dispr(mid)<Dm else (lo,mid)
eta_rand=eta_true+0.5*(lo+hi)*dr
print(f"floor W2 {werr(Afl,Rfl)*100:.2f}% (start {werr(Ast_start,Rst)*100:.2f}%)")
print("compensation at floor:  N/S T/S cos(reta,rW):")
print(f"  member-0 eta : {comp(Afl,Rfl,eta_m)}")
tv=(Astar-Afl).reshape(-1)
for tag,e,R in [("member-0 eta (frozen)",eta_m,Rfl),("TRUE eta*",eta_true,Rfl),("random non-comp eta",eta_rand,Rfl)]:
    d=direction(Afl,Rfl,e); cos=float((d.reshape(-1)@tv)/(d.norm()*tv.norm()+1e-30))
    ws=[werr(Afl+l*d,Rfl)*100 for l in (0.03,0.1,0.3,1.0)]
    print(f"  {tag:24s} cos(d,truth) {cos:+.3f} | W2@lam: " + " ".join(f"{w:.2f}" for w in ws),flush=True)
