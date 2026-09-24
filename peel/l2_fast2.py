"""FAST + PRECISE L2 diagnostic: float32 but GN step via damped LSQR (matrix-free, condition kappa
not kappa^2), so it reaches the 1e-4 target without fp64. Validates on the cheat control, then honest.
"""
import sys, time, torch
from torch.func import jvp, vjp
torch.set_default_dtype(torch.float32)
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
dev="cuda"
P="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
pk=torch.load(P+"peel_committee.pt",map_location=dev,weights_only=False); dims=[784,128,80,40,32,10]
teacher=MLP(dims,act="sigmoid").to(dev); teacher.load_state_dict(pk["teacher_state"]); teacher.eval()
for p in teacher.parameters(): p.requires_grad_(False)
Wt=[teacher.layers[i].weight.detach().float() for i in range(5)]; bt=[teacher.layers[i].bias.detach().float() for i in range(5)]
W2t=Wt[1]; nt=W2t.norm(dim=1)
def oracle_fwd(h):
    q2=torch.sigmoid(h@Wt[1].t()+bt[1]);q3=torch.sigmoid(q2@Wt[2].t()+bt[2])
    q4=torch.sigmoid(q3@Wt[3].t()+bt[3]);return q4@Wt[4].t()+bt[4]
def match(A,B):
    Cp=torch.cdist(A,B);Cm=torch.cdist(-A,B);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r);c=torch.tensor(c);s=torch.where(Cm[r,c]<Cp[r,c],-1.,1.);return r,c,s
mem=[{k:v.to(dev).float() for k,v in sd.items()} for sd in pk["pop_states"]]
def consensus(ref):
    A=[]
    for sd in mem:
        Wm=sd["layers.1.weight"]; r,c,s=match(Wm,ref); Wa=torch.zeros_like(Wm); Wa[c]=Wm[r]*s[:,None]; A.append(Wa)
    A=torch.stack(A); med=A.median(0).values.clone()
    for j in range(80):
        m=A[:,j].median(0).values; cl=((A[:,j]-m).norm(dim=1)/m.norm().clamp_min(1e-9))<0.2
        if int(cl.sum())>=5: med[j]=A[cl,j].mean(0)
    return med
seed_true=consensus(W2t); seed_mem0=consensus(mem[0]["layers.1.weight"])
gb2=mem[0]["layers.1.bias"];gW3=mem[0]["layers.2.weight"];gb3=mem[0]["layers.2.bias"]
gW4=mem[0]["layers.3.weight"];gb4=mem[0]["layers.3.bias"];gW5=mem[0]["layers.4.weight"];gb5=mem[0]["layers.4.bias"]
def wscore(W2):
    Cp=torch.cdist(W2,W2t);Cm=torch.cdist(-W2,W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[t],ci[t]] for t in range(len(ri))]);rel=e/nt[ci].cpu()
    return float(rel.mean()),float(rel.max()),int((rel<0.01).sum())
N=512; M=96
gz=torch.Generator(device="cpu").manual_seed(0);scales=torch.tensor([1.5,4.0,10.0])
sidx=torch.randint(0,3,(N,),generator=gz);z=torch.randn(N,128,generator=gz)*scales[sidx][:,None]
h=torch.sigmoid(z).clamp(1e-5,1-1e-5).to(dev)
Qg=torch.Generator(device="cpu").manual_seed(3);Q,_=torch.linalg.qr(torch.randn(128,128,generator=Qg));Q=Q[:M].to(dev)
valt=oracle_fwd(h).detach();scv=valt.abs().max()
ddt=torch.func.vmap(lambda u:jvp(oracle_fwd,(h,),(u.expand(N,128),))[1])(Q).detach();scd=ddt.abs().max()

def lsqr(Aop, Atop, b, n, damp, iters=80):
    # min ||A x - b||^2 + damp^2||x||^2, matrix-free (Paige-Saunders).
    beta=b.norm();
    if float(beta)==0: return torch.zeros(n,device=b.device)
    u=b/beta; v=Atop(u); alfa=v.norm(); v=v/alfa.clamp_min(1e-30)
    w=v.clone(); x=torch.zeros(n,device=b.device); phibar=beta; rhobar=alfa
    for _ in range(iters):
        u=Aop(v)-alfa*u; beta=u.norm(); u=u/beta.clamp_min(1e-30)
        v=Atop(u)-beta*v; alfa=v.norm(); v=v/alfa.clamp_min(1e-30)
        rhobar1=(rhobar**2+damp**2).sqrt(); c1=rhobar/rhobar1; s1=damp/rhobar1
        phibar=c1*phibar
        rho=(rhobar1**2+beta**2).sqrt(); c=rhobar1/rho; s=beta/rho
        theta=s*alfa; rhobar=-c*alfa; phi=c*phibar; phibar=s*phibar
        x=x+(phi/rho)*w; w=v-(theta/rho)*w
    return x
def refine(w0,resid,iters=40):
    wf=w0.clone(); r=resid(wf); c=float(r@r); n=wf.numel(); damp=1e-2
    for it in range(iters):
        _,vjpf=vjp(resid,wf); Jt=lambda u:vjpf(u)[0]; Jv=lambda v:jvp(resid,(wf,),(v,))[1]
        ok=False
        for _ in range(10):
            dwf=lsqr(Jv,Jt,-r,n,damp); wn=wf+dwf; rn=resid(wn); cn=float(rn@rn)
            if cn<c: wf=wn;r=rn;c=cn;damp=max(damp*0.3,1e-8);ok=True;break
            damp*=4
        if not ok or c<1e-18: break
    return wf,c
def make_model(down):
    b2,W3,b3,W4,b4,W5,b5=down
    def m(h,W2):
        q2=torch.sigmoid(h@W2.t()+b2);q3=torch.sigmoid(q2@W3.t()+b3);q4=torch.sigmoid(q3@W4.t()+b4);return q4@W5.t()+b5
    return m
def resid_W2(model):
    def r(wf):
        W2=wf.reshape(80,128);fh=lambda hb: model(hb,W2)
        val=model(h,W2);dd=torch.func.vmap(lambda u:jvp(fh,(h,),(u.expand(N,128),))[1])(Q)
        return torch.cat([((val-valt)/scv).reshape(-1),((dd-ddt)/scd).reshape(-1)])
    return r
TRUE=(bt[1],Wt[2],bt[2],Wt[3],bt[3],Wt[4],bt[4]); GUESS=(gb2,gW3,gb3,gW4,gb4,gW5,gb5)
def timed(tag,seed,resid):
    t=time.time();m0=wscore(seed);wf,c=refine(seed.reshape(-1).clone(),resid);m1=wscore(wf.reshape(80,128))
    print(f"  {tag:38s} {m0[0]:.3e}/{m0[1]:.2e} -> {m1[0]:.3e}/{m1[1]:.2e}  <1%:{m1[2]}/80  loss {c:.2e}  [{time.time()-t:.1f}s]",flush=True)
print(f"float32 + LSQR-GN, N={N}, m={M}.  W rel-row mean/max.")
timed("(1) CHEAT true-corr+true-down", seed_true, resid_W2(make_model(TRUE)))
timed("(2) HONEST guessed-down, W2-only", seed_mem0, resid_W2(make_model(GUESS)))
