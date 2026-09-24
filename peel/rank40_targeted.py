import sys, torch, numpy as np
from torch.func import jvp, vjp, jacfwd
from scipy.optimize import linear_sum_assignment
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
def perneuron_err(A):
    W2h=A@Brec; Dp=torch.cdist(W2s,W2h);Dm=torch.cdist(W2s,-W2h);D=torch.minimum(Dp,Dm).cpu().numpy();r,c=linear_sum_assignment(D)
    e=np.zeros(80)
    for i in range(80): e[c[i]]=float(torch.minimum((W2h[c[i]]-W2s[r[i]]).norm(),(-W2h[c[i]]-W2s[r[i]]).norm())/nt[r[i]])
    return e  # per CANDIDATE row
def werr(A):
    W2=A@Brec;Cp=torch.cdist(W2,W2s);Cm=torch.cdist(-W2,W2s);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C); return float((torch.tensor([C[ri[i],ci[i]] for i in range(80)])/nt[ci].cpu()).mean())
def maxdw(A):
    W2h=A@Brec;Dp=torch.cdist(W2s,W2h);Dm=torch.cdist(W2s,-W2h);D=torch.minimum(Dp,Dm).cpu().numpy();r,c=linear_sum_assignment(D)
    sgn=torch.where(Dm[r,c]<Dp[r,c],-1.,1.);Wal=torch.zeros_like(W2s);Wal[r]=W2h[c]*sgn[:,None]; return float((Wal-W2s).abs().max())
# base probes
NB=400; gh2=torch.Generator(device=dev).manual_seed(7); Hbase=torch.sigmoid(torch.randn(NB,128,generator=gh2,device=dev)*1.0).clamp(2e-2,1-2e-2)
Qb=Jseal(Hbase)@Brec.t()
# identify WEAK candidate rows via Hinv-variance of the current residual Jacobian
A0=p[:6400].reshape(80,80); b2=p[6400:]; U=Hbase@Brec.t()
def Kof(a,Q,U): A=a.reshape(80,80); Dp=sig1(U@A.t()+b2); return torch.einsum('nok,kj->noj',Q,torch.linalg.inv(A))/Dp[:,None,:]
Qd=Qb[:100]; Ud=U[:100]
with torch.no_grad():
    _,_,Vh=torch.linalg.svd(Kof(A0.reshape(-1),Qd,Ud).reshape(100*10,80),full_matrices=True); Nm=Vh[40:].t().contiguous()
J=jacfwd(lambda a: torch.einsum('noj,jm->nom',Kof(a,Qd,Ud),Nm).reshape(-1))(A0.reshape(-1))
Hh=J.t()@J; L,V=torch.linalg.eigh(Hh); Hinvd=(V**2/(L+1e-10*float(L.max()))).sum(1)
vj=torch.tensor([float(Hinvd[j*80:(j+1)*80].sum()) for j in range(80)])
weak=torch.argsort(vj,descending=True)[:12].tolist()
print(f"weak candidate rows (top Hinv-var): {weak}",flush=True)
e0=perneuron_err(A0); print(f"[start] werr {werr(A0)*100:.3f}%  max|dw| {maxdw(A0):.4f}  weak-rows mean err {np.mean([e0[j] for j in weak])*100:.2f}%",flush=True)
# TARGETED probes: z_j ~ 0 for each weak neuron (its sigmoid knee)
W2h=A0@Brec
gt=torch.Generator(device=dev).manual_seed(11); Ht=[]
for j in weak:
    wj=W2h[j]; base=torch.sigmoid(torch.randn(40,128,generator=gt,device=dev)*1.2)
    z=base@wj+b2[j]; proj=(base-(z/(wj@wj))[:,None]*wj[None,:]).clamp(2e-2,1-2e-2); Ht.append(proj)
Ht=torch.cat(Ht,0)
Hall=torch.cat([Hbase,Ht],0); NBa=Hall.shape[0]; Ua=Hall@Brec.t(); Qa=Jseal(Hall)@Brec.t()
print(f"added {Ht.shape[0]} targeted probes -> {NBa} total",flush=True)
def resid(pp,Nm): A=pp[:6400].reshape(80,80); b2=pp[6400:]; return torch.einsum('noj,jm->nom',Kof(pp[:6400],Qa,Ua),Nm).reshape(-1) if False else torch.einsum('noj,jm->nom',(lambda A: torch.einsum('nok,kj->noj',Qa,torch.linalg.inv(A))/sig1(Ua@A.t()+b2)[:,None,:])(A),Nm).reshape(-1)
def lsqr(Aop,Atop,b,n,damp,iters=70):
    beta=b.norm()
    if float(beta)==0: return torch.zeros(n,device=b.device)
    u=b/beta;v=Atop(u);al=v.norm();v=v/al.clamp_min(1e-30);w=v.clone();x=torch.zeros(n,device=b.device);pb=beta;rb=al
    for _ in range(iters):
        u=Aop(v)-al*u;beta=u.norm();u=u/beta.clamp_min(1e-30);v=Atop(u)-beta*v;al=v.norm();v=v/al.clamp_min(1e-30)
        r1=(rb**2+damp**2).sqrt();c1=rb/r1;pb=c1*pb;rho=(r1**2+beta**2).sqrt();cc=r1/rho;sg=beta/rho;th=sg*al;rb=-cc*al;phi=cc*pb;pb=sg*pb
        x=x+(phi/rho)*w;w=v-(th/rho)*w
    return x
lam=1e-8
for it in range(4001):
    A=p[:6400].reshape(80,80); b2=p[6400:]
    if it%400==0:
        ec=perneuron_err(A); print(f"  it{it:4d}: werr {werr(A)*100:.4f}%  max|dw| {maxdw(A):.4f}  weak-rows mean {np.mean([ec[j] for j in weak])*100:.2f}%",flush=True)
        torch.save({"p":p.cpu(),"Brec":Brec.cpu(),"b1r":b1r.cpu()},"rank40_targeted_traj.pt")
    with torch.no_grad():
        _,_,Vh=torch.linalg.svd((lambda A: torch.einsum('nok,kj->noj',Qa,torch.linalg.inv(A))/sig1(Ua@A.t()+b2)[:,None,:])(A).reshape(NBa*10,80),full_matrices=True); Nm=Vh[40:].t().contiguous()
    r0=resid(p,Nm); bb=float(r0@r0)
    f=lambda q: resid(q,Nm); Jv=lambda v: jvp(f,(p,),(v,))[1]; Jt=lambda u: vjp(f,p)[1](u)[0]
    ok=False
    for _ in range(6):
        d=lsqr(Jv,Jt,-r0,6480,lam); pn=p+d
        if float(resid(pn,Nm)@resid(pn,Nm))<bb: p=pn;lam=max(lam/3,1e-16);ok=True;break
        lam*=5
    if not ok: print(f"  no-progress it{it}"); break
print("DONE",flush=True)
