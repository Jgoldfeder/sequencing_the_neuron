import sys, torch, numpy as np
from scipy.optimize import linear_sum_assignment
from torch.func import jvp, vjp
torch.set_default_dtype(torch.float64); dev="cuda"
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
pk=torch.load("peel_committee.pt",map_location=dev,weights_only=False); dims=pk["dims"]
tt=MLP(dims,act="sigmoid").to(dev).double(); tt.load_state_dict({k:v.double() for k,v in pk["teacher_state"].items()}); tt.eval()
W2s=tt.layers[1].weight.detach(); b2s=tt.layers[1].bias.detach(); nt=W2s.norm(dim=1)
def out(x):
    with torch.no_grad(): return tt(x)
mem=[{k:v.to(dev).double() for k,v in sd.items()} for sd in pk["pop_states"]]
W1r=mem[0]["layers.0.weight"]; W1rp=W1r.t()@torch.linalg.inv(W1r@W1r.t())
ck=torch.load("rank40_trackgood2_traj.pt",map_location=dev,weights_only=False)
p0=ck["p"].to(dev); Brec=ck["Brec"].to(dev); b1r=ck["b1r"].to(dev)
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
# ---- current Hungarian match -> classify GOOD/BAD candidate rows, build oracle for bad rows ----
A0=p0[:6400].reshape(80,80); W2h0=A0@Brec
Dp=torch.cdist(W2s,W2h0);Dm=torch.cdist(W2s,-W2h0);D=torch.minimum(Dp,Dm).cpu().numpy();ri,ci=linear_sum_assignment(D)
e=np.zeros(80); sign_c=np.ones(80); true_of=np.zeros(80,dtype=int)
for k in range(80):
    tr,ca=ri[k],ci[k]; s=-1.0 if Dm[tr,ca]<Dp[tr,ca] else 1.0
    e[ca]=float(torch.minimum((W2h0[ca]-W2s[tr]).norm(),(-W2h0[ca]-W2s[tr]).norm())/nt[tr]); sign_c[ca]=s; true_of[ca]=tr
BAD=[c for c in range(80) if e[c]>0.10]           # candidate rows (params) that are bad
GOODc=[c for c in range(80) if e[c]<0.08]          # good candidate rows
print(f"BAD candidate rows: {len(BAD)}   GOOD candidate rows: {len(GOODc)}",flush=True)
# oracle A-row and b2 for the bad rows (correct gauge & sign)
Aor=A0.clone(); b2or=p0[6400:].clone()
for c in BAD:
    Aor[c]=(sign_c[c]*W2s[true_of[c]])@Brec.t(); b2or[c]=sign_c[c]*b2s[true_of[c]]
# freeze mask over the 6480-vector: True where FROZEN (bad rows of A + their b2)
frozen=torch.zeros(6480,dtype=torch.bool,device=dev)
for c in BAD:
    frozen[c*80:(c+1)*80]=True; frozen[6400+c]=True
def goodmean(p):
    A=p[:6400].reshape(80,80); W=A@Brec
    Dp=torch.cdist(W2s,W);Dm=torch.cdist(W2s,-W);DD=torch.minimum(Dp,Dm).cpu().numpy();r,c=linear_sum_assignment(DD)
    ee=np.zeros(80)
    for i in range(80): ee[c[i]]=float(torch.minimum((W[c[i]]-W2s[r[i]]).norm(),(-W[c[i]]-W2s[r[i]]).norm())/nt[r[i]])
    g=[ee[c] for c in GOODc]; return np.mean(g)*100, np.max(g)*100
freem=~frozen; nf=int(freem.sum())
def embed(vr): v=torch.zeros(6480,device=dev); v[freem]=vr; return v
def run(tag, badval_A, badval_b2, iters=1600):
    p=p0.clone()
    for c in BAD:  # set the frozen bad rows to the chosen value
        p[c*80:(c+1)*80]=badval_A[c]; p[6400+c]=badval_b2[c]
    lam=1e-8; gm0,gx0=goodmean(p); print(f"[{tag}] start good-mean {gm0:.4f}% good-max {gx0:.3f}%",flush=True)
    for it in range(1,iters+1):
        A=p[:6400].reshape(80,80); b2=p[6400:]
        with torch.no_grad():
            _,_,Vh=torch.linalg.svd(Kof(A,b2).reshape(NB*10,80),full_matrices=True); Nm=Vh[40:].t().contiguous()
        r0=resid(p,Nm); bb=float(r0@r0)
        f=lambda q: resid(q,Nm)
        Jvr=lambda vr: jvp(f,(p,),(embed(vr),))[1]           # projected: free params only
        Jtr=lambda u: vjp(f,p)[1](u)[0][freem]
        ok=False
        for _ in range(6):
            d=embed(lsqr(Jvr,Jtr,-r0,nf,lam)); pn=p+d          # step lives in free subspace
            if float(resid(pn,Nm)@resid(pn,Nm))<bb: p=pn;lam=max(lam/3,1e-16);ok=True;break
            lam*=5
        if not ok: print(f"  [{tag}] no-progress it{it}",flush=True); break
        if it%200==0:
            gm,gx=goodmean(p); print(f"  [{tag}] it{it:4d}: good-mean {gm:.4f}%  good-max {gx:.3f}%",flush=True)
    return p
# CONTROL: bad rows frozen at CURRENT (wrong) values
run("WRONG-frozen", A0.clone(), p0[6400:].clone())
# TREATMENT: bad rows frozen at ORACLE values
run("ORACLE-frozen", Aor, b2or)
print("DONE",flush=True)
