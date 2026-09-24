import sys, torch, numpy as np
from torch.func import jacrev, vmap, jvp, vjp
from scipy.optimize import linear_sum_assignment
torch.set_default_dtype(torch.float64); dev="cuda"
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
pk=torch.load("peel_committee.pt",map_location=dev,weights_only=False); dims=pk["dims"]
tt=MLP(dims,act="sigmoid").to(dev).double(); tt.load_state_dict({k:v.double() for k,v in pk["teacher_state"].items()}); tt.eval()
W2s=tt.layers[1].weight.detach(); b2=tt.layers[1].bias.detach()
W3=tt.layers[2].weight.detach(); b3=tt.layers[2].bias.detach(); W4=tt.layers[3].weight.detach(); b4=tt.layers[3].bias.detach()
W5=tt.layers[4].weight.detach(); b5=tt.layers[4].bias.detach()
B=torch.linalg.svd(W2s,full_matrices=False)[2][:80]; Astar=W2s@B.t()
def outh(h):
    s2=torch.sigmoid(h@W2s.t()+b2); s3=torch.sigmoid(s2@W3.t()+b3); s4=torch.sigmoid(s3@W4.t()+b4); return s4@W5.t()+b5
Jb=vmap(jacrev(outh))
def sig1(z): s=torch.sigmoid(z); return s*(1-s)
def werr(A):
    W2=A@B;Cp=torch.cdist(W2,W2s);Cm=torch.cdist(-W2,W2s);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);return float((torch.tensor([C[ri[i],ci[i]] for i in range(80)])/W2s.norm(dim=1)[ci].cpu()).mean())
NB=400; gh=torch.Generator(device=dev).manual_seed(7); H=torch.sigmoid(torch.randn(NB,128,generator=gh,device=dev)*1.0).clamp(2e-2,1-2e-2)
U=H@B.t(); Q=(Jb(H)@B.t())
def Kof(A): Dp=sig1(U@A.t()+b2); Ainv=torch.linalg.inv(A); return torch.einsum('nok,kj->noj',Q,Ainv)/Dp[:,None,:]
def resid(A,Nm): return torch.einsum('noj,jm->nom',Kof(A),Nm).reshape(-1)
def botr(A):
    sv=torch.linalg.svdvals(Kof(A).reshape(NB*10,80)); return float((sv[40:]**2).sum()/(sv**2).sum())
def lsqr(Aop,Atop,b,n,damp,iters=80):
    beta=b.norm()
    if float(beta)==0: return torch.zeros(n,device=b.device)
    u=b/beta;v=Atop(u);al=v.norm();v=v/al.clamp_min(1e-30);w=v.clone();x=torch.zeros(n,device=b.device);pb=beta;rb=al
    for _ in range(iters):
        u=Aop(v)-al*u;beta=u.norm();u=u/beta.clamp_min(1e-30);v=Atop(u)-beta*v;al=v.norm();v=v/al.clamp_min(1e-30)
        r1=(rb**2+damp**2).sqrt();c1=rb/r1;pb=c1*pb;rho=(r1**2+beta**2).sqrt();cc=r1/rho;sg=beta/rho;th=sg*al;rb=-cc*al;phi=cc*pb;pb=sg*pb
        x=x+(phi/rho)*w;w=v-(th/rho)*w
    return x
gg=torch.Generator(device=dev).manual_seed(1); A=((W2s+0.01*W2s.norm(dim=1,keepdim=True)*torch.randn(80,128,generator=gg,device=dev)/(128**0.5))@B.t()).clone()
lam=1e-6; traj=[]
print(f"start werr {werr(A)*100:.5f}%",flush=True)
for it in range(801):
    e=werr(A)*100; traj.append(e)
    if it in (0,25,50,100,150,200,300,400,500,600,700,800):
        print(f"  it{it:4d}: werr {e:.5f}%  bot40 {botr(A):.3e}",flush=True)
        torch.save({"traj":traj,"A":A.cpu()},"rank40_long_traj.pt")
    with torch.no_grad():
        _,_,Vh=torch.linalg.svd(Kof(A).reshape(NB*10,80),full_matrices=True); Nm=Vh[40:].t().contiguous()
    r0=resid(A,Nm); base=float(r0@r0)
    f=lambda a: resid(a.reshape(80,80),Nm); Jv=lambda v: jvp(f,(A.reshape(-1),),(v,))[1]; Jt=lambda u: vjp(f,A.reshape(-1))[1](u)[0]
    ok=False
    for _ in range(6):
        d=lsqr(Jv,Jt,-r0,6400,lam); An=A+d.reshape(80,80)
        if float(resid(An,Nm)@resid(An,Nm))<base: A=An;lam=max(lam/3,1e-15);ok=True;break
        lam*=5
    if not ok: print(f"  no-progress at it{it}",flush=True); break
# window-averaged rate diagnostic
t=traj; import numpy as np
def rate(a,b): return (t[a]-t[b])/(b-a)
print("=== ΔE/Δt over growing windows ===")
for (a,b) in [(0,100),(100,200),(200,400),(400,800)]:
    if b<len(t): print(f"  [{a},{b}]: E {t[a]:.4f}->{t[b]:.4f}  rate {rate(a,b)*1000:.4f} e-3 %/iter",flush=True)
print("DONE",flush=True)
