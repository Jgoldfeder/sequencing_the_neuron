"""STAGE 1 of the reviewer's staged basin test: TRUE B + EXACT ORACLE jets, from the committee init.
Question: does constraining W2=A B (reduce 80x128 -> 80x80) enlarge the optimization basin enough to
converge from the ~16% committee guess? If NO here (perfect B, perfect jets), the row-space reduction
did NOT fix the real bottleneck. DIAGNOSTIC stage (oracle target); stages 2/3 add B-error and sealed FD.

Model: theta=(A,b2,W3,b3,W4,b4,W5,b5), W2=A@B. Observables = value + jets along the 80 B directions.
Target = same observables at the TRUE params (exact). Solve with float32 + matrix-free damped LSQR-GN.
"""
import sys, torch
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
W3t=t.layers[2].weight.detach(); b3t=t.layers[2].bias.detach()
W4t=t.layers[3].weight.detach(); b4t=t.layers[3].bias.detach()
W5t=t.layers[4].weight.detach(); b5t=t.layers[4].bias.detach()
# TRUE row-space basis B (orthonormal); A_true=W2t B^T
Uw,Sw,Vh=torch.linalg.svd(W2t,full_matrices=False); B=Vh[:80]
# ---- committee init (truth-free consensus W2 in member-0 frame + member-0 downstream) ----
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False)
mem=[{kk:vv.to(dev).float() for kk,vv in sd.items()} for sd in pk["pop_states"]]
def match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r);c=torch.tensor(c);s=torch.where(Cm[r,c]<Cp[r,c],-1.,1.);return r,c,s
ref=mem[0]["layers.1.weight"]; AW=[ref.clone()]
for sd in mem[1:]:
    Wm=sd["layers.1.weight"];r,c,s=match(Wm,ref);Wa=torch.zeros_like(Wm);Wa[c]=Wm[r]*s[:,None];AW.append(Wa)
W2g=torch.stack(AW).median(0).values.clone()                     # consensus, member-0 frame (~16%)
b2g=mem[0]["layers.1.bias"].clone()
W3g=mem[0]["layers.2.weight"].clone(); b3g=mem[0]["layers.2.bias"].clone()
W4g=mem[0]["layers.3.weight"].clone(); b4g=mem[0]["layers.3.bias"].clone()
W5g=mem[0]["layers.4.weight"].clone(); b5g=mem[0]["layers.4.bias"].clone()
A_init=W2g@B.t()                                                  # 80x80
# ---- probes (guess-targeted, excite each neuron) ----
W2gp=torch.linalg.pinv(W2g); gg=torch.Generator(device=dev).manual_seed(0); Hs=[]
for rep in range(1):
    for j in range(80):
        tt=torch.randn(80,generator=gg,device=dev)*2.0; tt[j]=0.0
        Hs.append(torch.clamp(W2gp@(tt-b2g),1e-3,1-1e-3))
H=torch.stack(Hs); Pn=H.shape[0]
# ---- pack/unpack ----
sizes=[6400,80,3200,40,1280,32,320,10]; shp=[(80,80),(80,),(40,80),(40,),(32,40),(32,),(10,32),(10,)]
def unpack(th):
    o=0;out=[]
    for n,sh in zip(sizes,shp): out.append(th[o:o+n].reshape(sh)); o+=n
    return out
def obs(th):
    A,b2,W3,b3,W4,b4,W5,b5=unpack(th); W2=A@B
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2)
    z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4)
    out=s4@W5.t()+b5
    U=(W2@B.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3
    dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    return torch.cat([out.reshape(-1),dout.reshape(-1)])
th_true=torch.cat([(W2t@B.t()).reshape(-1),b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t])
th0=torch.cat([A_init.reshape(-1),b2g,W3g.reshape(-1),b3g,W4g.reshape(-1),b4g,W5g.reshape(-1),b5g])
target=obs(th_true).detach()
vsc=target.abs().max()
def resid(th): return (obs(th)-target)/vsc
def wscore(th):
    A=unpack(th)[0]; W2=A@B
    Cp=torch.cdist(W2,W2t);Cm=torch.cdist(-W2,W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[t2],ci[t2]] for t2 in range(len(ri))]);rel=e/nt[ci].cpu()
    return float(rel.mean()),float(rel.max()),int((rel<0.01).sum())
def lsqr(Aop,Atop,b,n,damp,iters=80):
    beta=b.norm()
    if float(beta)==0: return torch.zeros(n,device=b.device)
    u=b/beta; v=Atop(u); alfa=v.norm(); v=v/alfa.clamp_min(1e-30); w=v.clone(); x=torch.zeros(n,device=b.device); phibar=beta; rhobar=alfa
    for _ in range(iters):
        u=Aop(v)-alfa*u; beta=u.norm(); u=u/beta.clamp_min(1e-30)
        v=Atop(u)-beta*v; alfa=v.norm(); v=v/alfa.clamp_min(1e-30)
        rb1=(rhobar**2+damp**2).sqrt(); c1=rhobar/rb1; phibar=c1*phibar
        rho=(rb1**2+beta**2).sqrt(); c=rb1/rho; s=beta/rho
        theta=s*alfa; rhobar=-c*alfa; phi=c*phibar; phibar=s*phibar
        x=x+(phi/rho)*w; w=v-(theta/rho)*w
    return x
def refine(w0,iters=60):
    wf=w0.clone(); r=resid(wf); c=float(r@r); n=wf.numel(); damp=1e-2
    for it in range(iters):
        _,vjpf=vjp(resid,wf); Jt=lambda u:vjpf(u)[0]; Jv=lambda v:jvp(resid,(wf,),(v,))[1]; ok=False
        for _ in range(12):
            dwf=lsqr(Jv,Jt,-r,n,damp); wn=wf+dwf; rn=resid(wn); cn=float(rn@rn)
            if cn<c: wf=wn;r=rn;c=cn;damp=max(damp*0.3,1e-9);ok=True;break
            damp*=4
        if not ok or c<1e-16: break
    return wf,c
print(f"[stage1] TRUE B, exact oracle jets. probes={Pn}, obs={target.numel()}, params={th0.numel()}")
# member-0 WHOLESALE init (consistent functional pair): W2 = member-0's own
A_mem0=mem[0]["layers.1.weight"]@B.t()
th_mem0=torch.cat([A_mem0.reshape(-1),b2g,W3g.reshape(-1),b3g,W4g.reshape(-1),b4g,W5g.reshape(-1),b5g])
# near-truth control
gc=torch.Generator(device=dev).manual_seed(7); th_ctrl=th_true+0.03*th_true.abs().mean()*torch.randn(th_true.numel(),generator=gc,device=dev)
def run(tag,th):
    m0=wscore(th); wf,c=refine(th); m1=wscore(wf)
    print(f"  {tag:32s} init {m0[0]:.3e}/{m0[1]:.2e} -> final {m1[0]:.3e}/{m1[1]:.2e}  <1%:{m1[2]}/80  loss {c:.2e}",flush=True)
run("(a) member-0 wholesale (23%)", th_mem0)
run("(b) consensus W2 + mem0 down (16%)", th0)
print("basin radius sweep (isotropic perturbation of TRUTH):")
for eps in [0.03,0.05,0.08,0.11,0.15]:
    gcs=torch.Generator(device=dev).manual_seed(100+int(eps*1000))
    thp=th_true+eps*th_true.abs().mean()*torch.randn(th_true.numel(),generator=gcs,device=dev)
    run(f"  truth+{eps:.2f}", thp)
