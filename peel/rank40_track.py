import sys, torch, numpy as np
from scipy.optimize import linear_sum_assignment
from torch.func import jvp, vjp
torch.set_default_dtype(torch.float64); dev="cuda"
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
pk=torch.load("peel_committee.pt",map_location=dev,weights_only=False); dims=pk["dims"]
tt=MLP(dims,act="sigmoid").to(dev).double(); tt.load_state_dict({k:v.double() for k,v in pk["teacher_state"].items()}); tt.eval()
W2s=tt.layers[1].weight.detach(); nt=W2s.norm(dim=1)
def out(x):
    with torch.no_grad(): return tt(x)
mem=[{k:v.to(dev).double() for k,v in sd.items()} for sd in pk["pop_states"]]
W1r=mem[0]["layers.0.weight"]; W1rp=W1r.t()@torch.linalg.inv(W1r@W1r.t())
ck=torch.load("rank40_fullsealed_traj.pt",map_location=dev,weights_only=False)
p=ck["p"].to(dev); Brec=ck["Brec"].to(dev); b1r=ck["b1r"].to(dev)
def sig1(z): s=torch.sigmoid(z); return s*(1-s)
def logit(h): return torch.log(h/(1-h))
def x_of_h(h): return (logit(h)-b1r)@W1rp.t()
def BBh(h): return out(x_of_h(h))
def Jseal(H,fd=1e-4,chunk=150):
    N=H.shape[0]; E=torch.eye(128,device=dev)*fd; o=[]
    for s in range(0,N,chunk):
        Hc=H[s:s+chunk]; m=Hc.shape[0]
        Hp=(Hc[:,None,:]+E[None]).reshape(-1,128).clamp(1e-4,1-1e-4); Hm=(Hc[:,None,:]-E[None]).reshape(-1,128).clamp(1e-4,1-1e-4)
        o.append(((BBh(Hp).reshape(m,128,10)-BBh(Hm).reshape(m,128,10))/(2*fd)).permute(0,2,1))
    return torch.cat(o,0)
NB=400; gh2=torch.Generator(device=dev).manual_seed(7); H=torch.sigmoid(torch.randn(NB,128,generator=gh2,device=dev)*1.0).clamp(2e-2,1-2e-2)
U=H@Brec.t(); Q=(Jseal(H)@Brec.t())
# --- per-neuron error breakdown via Hungarian (true-row indexed) ---
def breakdown(A):
    W2h=A@Brec; Dp=torch.cdist(W2s,W2h);Dm=torch.cdist(W2s,-W2h);D=torch.minimum(Dp,Dm).cpu().numpy();r,c=linear_sum_assignment(D)
    e=np.zeros(80)  # indexed by TRUE neuron
    for i in range(80): e[r[i]]=float(torch.minimum((W2h[c[i]]-W2s[r[i]]).norm(),(-W2h[c[i]]-W2s[r[i]]).norm())/nt[r[i]])
    return e
def Kof(A,b2): Dp=sig1(U@A.t()+b2); return torch.einsum('nok,kj->noj',Q,torch.linalg.inv(A))/Dp[:,None,:]
def resid(pp,Nm): A=pp[:6400].reshape(80,80); b2=pp[6400:]; return torch.einsum('noj,jm->nom',Kof(A,b2),Nm).reshape(-1)
def lsqr(Aop,Atop,b,n,damp,iters=70):
    beta=b.norm()
    if float(beta)==0: return torch.zeros(n,device=b.device)
    u=b/beta;v=Atop(u);al=v.norm();v=v/al.clamp_min(1e-30);w=v.clone();x=torch.zeros(n,device=b.device);pb=beta;rb=al
    for _ in range(iters):
        u=Aop(v)-al*u;beta=u.norm();u=u/beta.clamp_min(1e-30);v=Atop(u)-beta*v;al=v.norm();v=v/al.clamp_min(1e-30)
        r1=(rb**2+damp**2).sqrt();c1=rb/r1;pb=c1*pb;rho=(r1**2+beta**2).sqrt();cc=r1/rho;sg=beta/rho;th=sg*al;rb=-cc*al;phi=cc*pb;pb=sg*pb
        x=x+(phi/rho)*w;w=v-(th/rho)*w
    return x
# identify the OUTLIER set (true-neuron indices >30% err) at the START, then TRACK these fixed indices
e0=breakdown(p[:6400].reshape(80,80))
OUT=sorted([i for i in range(80) if e0[i]>0.30])
MOD=sorted([i for i in range(80) if 0.10<e0[i]<=0.30])
print(f"[start] {len(OUT)} outliers(>30%): {OUT}",flush=True)
print(f"        {len(MOD)} moderate(10-30%): {MOD}",flush=True)
def rep(it,e):
    mean=e.mean()*100; med=np.median(e)*100; om=np.mean([e[i] for i in OUT])*100 if OUT else 0
    mm=np.mean([e[i] for i in MOD])*100 if MOD else 0
    print(f"  it{it:5d}: mean {mean:6.3f}%  median {med:5.3f}%  OUTmean {om:6.2f}%  MODmean {mm:5.2f}%  #>30 {int((e>0.3).sum())}  #>10 {int((e>0.1).sum())}  max {e.max()*100:.1f}%",flush=True)
rep(0,e0)
lam=1e-8
for it in range(1,6001):
    A=p[:6400].reshape(80,80); b2=p[6400:]
    with torch.no_grad():
        _,_,Vh=torch.linalg.svd(Kof(A,b2).reshape(NB*10,80),full_matrices=True); Nm=Vh[40:].t().contiguous()
    r0=resid(p,Nm); bb=float(r0@r0)
    f=lambda q: resid(q,Nm); Jv=lambda v: jvp(f,(p,),(v,))[1]; Jt=lambda u: vjp(f,p)[1](u)[0]
    ok=False
    for _ in range(6):
        d=lsqr(Jv,Jt,-r0,6480,lam); pn=p+d
        if float(resid(pn,Nm)@resid(pn,Nm))<bb: p=pn;lam=max(lam/3,1e-16);ok=True;break
        lam*=5
    if not ok: print(f"  no-progress it{it}",flush=True); break
    if it%250==0:
        rep(it,breakdown(p[:6400].reshape(80,80)))
        torch.save({"p":p.cpu(),"Brec":Brec.cpu(),"b1r":b1r.cpu()},"rank40_track_traj.pt")
print("DONE",flush=True)
