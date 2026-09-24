import sys, torch
from scipy.optimize import linear_sum_assignment
from torch.func import jvp, vjp
torch.set_default_dtype(torch.float64); dev="cuda"
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
pk=torch.load("peel_committee.pt",map_location=dev,weights_only=False); dims=pk["dims"]
tt=MLP(dims,act="sigmoid").to(dev).double(); tt.load_state_dict({k:v.double() for k,v in pk["teacher_state"].items()}); tt.eval()
W1t=tt.layers[0].weight.detach(); b1t=tt.layers[0].bias.detach()
W2s=tt.layers[1].weight.detach(); b2s=tt.layers[1].bias.detach()   # TRUE W2/b2 -> SCORING ONLY
def BB(x):   # black box: full 784->10 teacher
    with torch.no_grad(): return tt(x)
# ---- SEALED L1 (committee's shared recovered W1/b1; report distance from true) ----
mem=[{k:v.to(dev).double() for k,v in sd.items()} for sd in pk["pop_states"]]
W1r=mem[0]["layers.0.weight"]; b1r=mem[0]["layers.0.bias"]
print(f"[L1] ||W1_committee - W1_true|| rel = {float((W1r-W1t).norm()/W1t.norm()):.2e}  ||b1 rel|| {float((b1r-b1t).norm()/b1t.norm()):.2e}")
W1rp=W1r.t()@torch.linalg.inv(W1r@W1r.t())
def logit(h): return torch.log(h/(1-h))
def x_of_h(h): return (logit(h)-b1r)@W1rp.t()          # h -> x via RECOVERED L1
def BBh(h): return BB(x_of_h(h))                       # sealed subnetwork query
# verify round-trip h->x->h_real
gh=torch.Generator(device=dev).manual_seed(0); Hchk=torch.sigmoid(torch.randn(50,128,generator=gh,device=dev)*1.2).clamp(2e-2,1-2e-2)
h_real=torch.sigmoid(x_of_h(Hchk)@W1t.t()+b1t)
print(f"[L1] round-trip max|h_real-h| = {float((h_real-Hchk).abs().max()):.2e}")
def Jseal(H,fd=1e-4):                                   # sealed finite-diff Jacobian dBBh/dh -> (N,10,128)
    N=H.shape[0]; E=torch.eye(128,device=dev)*fd
    Hp=(H[:,None,:]+E[None]).reshape(-1,128).clamp(1e-4,1-1e-4); Hm=(H[:,None,:]-E[None]).reshape(-1,128).clamp(1e-4,1-1e-4)
    op=BBh(Hp).reshape(N,128,10); om=BBh(Hm).reshape(N,128,10)
    return ((op-om)/(2*fd)).permute(0,2,1)
# ---- RECOVER B (sealed) ----
gB=torch.Generator(device=dev).manual_seed(3); HB=torch.sigmoid(torch.randn(80,128,generator=gB,device=dev)*1.2).clamp(2e-2,1-2e-2)
JB=Jseal(HB)                                            # (80,10,128)
_,_,Vh=torch.linalg.svd(JB.reshape(-1,128),full_matrices=False); Brec=Vh[:80]
Btrue=torch.linalg.svd(W2s,full_matrices=False)[2][:80]
# subspace error between Brec and Btrue
P=Btrue@Brec.t(); ss=torch.linalg.svdvals(P); print(f"[B] recovered rowspace: min cos principal angle {float(ss.min()):.4f} (1=perfect); mean {float(ss.mean()):.4f}")
# ---- SEALED SOLVER from consensus (its own gauge, own b2) ----
W2c=pk["consensus"]["layers.1.weight"].double() if "consensus" in pk else torch.load("consensus_full.pt",map_location=dev,weights_only=False)["consensus"]["layers.1.weight"].double()
cf=torch.load("consensus_full.pt",map_location=dev,weights_only=False); W2c=cf["consensus"]["layers.1.weight"].double(); b2c=cf["consensus"]["layers.1.bias"].double()
def sig1(z): s=torch.sigmoid(z); return s*(1-s)
def werr(W2):   # SCORING ONLY (Hungarian, gauge-invariant)
    Cp=torch.cdist(W2,W2s);Cm=torch.cdist(-W2,W2s);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);nt=W2s.norm(dim=1)
    return float((torch.tensor([C[ri[i],ci[i]] for i in range(80)])/nt[ci].cpu()).mean())
NB=300; gh2=torch.Generator(device=dev).manual_seed(7); H=torch.sigmoid(torch.randn(NB,128,generator=gh2,device=dev)*1.0).clamp(2e-2,1-2e-2)
U=H@Brec.t(); Q=(Jseal(H)@Brec.t())                    # (NB,10,80) sealed
A=(W2c@Brec.t()).clone()                                # consensus in RECOVERED rowspace, OWN gauge (no truth align)
def Kof(A): Dp=sig1(U@A.t()+b2c); Ainv=torch.linalg.inv(A); return torch.einsum('nok,kj->noj',Q,Ainv)/Dp[:,None,:]
def resid(A,Nm): return torch.einsum('noj,jm->nom',Kof(A),Nm).reshape(-1)
def botr(A):
    sv=torch.linalg.svdvals(Kof(A).reshape(NB*10,80)); return float((sv[40:]**2).sum()/(sv**2).sum())
def lsqr(Aop,Atop,b,n,damp,iters=60):
    beta=b.norm()
    if float(beta)==0: return torch.zeros(n,device=b.device)
    u=b/beta;v=Atop(u);al=v.norm();v=v/al.clamp_min(1e-30);w=v.clone();x=torch.zeros(n,device=b.device);pb=beta;rb=al
    for _ in range(iters):
        u=Aop(v)-al*u;beta=u.norm();u=u/beta.clamp_min(1e-30);v=Atop(u)-beta*v;al=v.norm();v=v/al.clamp_min(1e-30)
        r1=(rb**2+damp**2).sqrt();c1=rb/r1;pb=c1*pb;rho=(r1**2+beta**2).sqrt();cc=r1/rho;sg=beta/rho;th=sg*al;rb=-cc*al;phi=cc*pb;pb=sg*pb
        x=x+(phi/rho)*w;w=v-(th/rho)*w
    return x
print(f"[SEALED SOLVER] start werr {werr(A@Brec)*100:.4f}%  bot40 {botr(A):.3e}",flush=True)
lam=1e-6
for it in range(2001):
    if it in (0,50,100,200,400,700,1000,1400,2000): print(f"  it{it:4d}: werr {werr(A@Brec)*100:.5f}%  bot40 {botr(A):.3e}",flush=True); torch.save({"A":A.cpu(),"Brec":Brec.cpu()},"rank40_sealed_traj.pt")
    with torch.no_grad():
        _,_,Vh=torch.linalg.svd(Kof(A).reshape(NB*10,80),full_matrices=True); Nm=Vh[40:].t().contiguous()
    r0=resid(A,Nm); base=float(r0@r0)
    f=lambda a: resid(a.reshape(80,80),Nm); Jv=lambda v: jvp(f,(A.reshape(-1),),(v,))[1]; Jt=lambda u: vjp(f,A.reshape(-1))[1](u)[0]
    ok=False
    for _ in range(6):
        d=lsqr(Jv,Jt,-r0,6400,lam); An=A+d.reshape(80,80)
        if float(resid(An,Nm)@resid(An,Nm))<base: A=An;lam=max(lam/3,1e-15);ok=True;break
        lam*=5
    if not ok: print(f"  no-progress it{it}"); break
print("DONE",flush=True)
