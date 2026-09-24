"""DOWNSTREAM-ANCHOR experiment (reviewer's roadmap). The population gives 8 factorizations
(W2^m,b2^m,D^m) of the same teacher; each downstream ADAPTED to its own W2 error. Test whether any
frozen committee downstream anchors a SHARED W2 toward truth, and whether jointly fitting one W2
across ALL 8 frozen downstreams cancels the nuisance.

Alignment: put all members in member-0's LAYER-2 gauge, propagating perm+complement into layer 3
(complement j: s2->1-s2  => W3[:,j]->-W3[:,j], b3+=W3[:,j]). W2=A B constrained (true B here to
isolate the anchor question). Oracle jets (diagnostic); sealed FD is the follow-up if this works.
Ranking uses HELD-OUT black-box loss (no truth).
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
W3t=t.layers[2].weight.detach(); b3t=t.layers[2].bias.detach(); W4t=t.layers[3].weight.detach(); b4t=t.layers[3].bias.detach()
W5t=t.layers[4].weight.detach(); b5t=t.layers[4].bias.detach()
Uw,Sw,Vh=torch.linalg.svd(W2t,full_matrices=False); B=Vh[:80]
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False)
mem=[{kk:vv.to(dev).float() for kk,vv in sd.items()} for sd in pk["pop_states"]]
def match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    return torch.tensor(r),torch.tensor(c),torch.where(Cm[torch.tensor(r),torch.tensor(c)]<Cp[torch.tensor(r),torch.tensor(c)],-1.,1.)
# ---- align all members to member-0 layer-2 gauge, propagate to layer 3 ----
ref=mem[0]["layers.1.weight"]; aligned=[]
for sd in mem:
    W2m=sd["layers.1.weight"]; b2m=sd["layers.1.bias"]; W3m=sd["layers.2.weight"]; b3m=sd["layers.2.bias"]
    r,c,s=match(W2m,ref)
    W2a=torch.zeros_like(W2m); b2a=torch.zeros_like(b2m); W3a=torch.zeros_like(W3m)
    W2a[c]=W2m[r]*s[:,None]; b2a[c]=b2m[r]*s
    W3a[:,c]=W3m[:,r]*s[None,:]                                   # column sign
    b3a=b3m+ (W3m[:,r][:, s<0]).sum(1)                            # complement constant into b3
    aligned.append(dict(W2=W2a,b2=b2a,W3=W3a,b3=b3a,W4=sd["layers.3.weight"],b4=sd["layers.3.bias"],
                        W5=sd["layers.4.weight"],b5=sd["layers.4.bias"]))
W2cons=torch.stack([a["W2"] for a in aligned]).median(0).values          # consensus W2 (common gauge)
A_init=(W2cons@B.t())
def w2err(A):
    W2=A@B; Cp=torch.cdist(W2,W2t);Cm=torch.cdist(-W2,W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);rel=e/nt[ci].cpu()
    return float(rel.mean()),float(rel.max()),int((rel<0.01).sum())
# ---- probes (train + held-out), guess-targeted ----
W2gp=torch.linalg.pinv(W2cons); b2cons=torch.stack([a["b2"] for a in aligned]).median(0).values
def make_probes(seed,per):
    gg=torch.Generator(device=dev).manual_seed(seed); Hs=[]
    for j in range(per):
        tt=torch.randn(80,generator=gg,device=dev)*2.0; tt[j%80]=0.0
        Hs.append(torch.clamp(W2gp@(tt-b2cons),1e-3,1-1e-3))
    return torch.stack(Hs)
Htr=make_probes(0,80); Hho=make_probes(999,40)
def observ(A,D,H):
    W2=A@B; b2,W3,b3,W4,b4,W5,b5=D
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2)
    z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4)
    out=s4@W5.t()+b5; U=(W2@B.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3
    dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    return torch.cat([out.reshape(-1),dout.reshape(-1)])
Dtrue=(b2t,W3t,b3t,W4t,b4t,W5t,b5t)
tgt_tr=observ(W2t@B.t(),Dtrue,Htr).detach(); vsc=tgt_tr.abs().max()
tgt_ho=observ(W2t@B.t(),Dtrue,Hho).detach()
def lsqr(Aop,Atop,b,n,damp,iters=60):
    beta=b.norm()
    if float(beta)==0: return torch.zeros(n,device=b.device)
    u=b/beta;v=Atop(u);alfa=v.norm();v=v/alfa.clamp_min(1e-30);w=v.clone();x=torch.zeros(n,device=b.device);phibar=beta;rhobar=alfa
    for _ in range(iters):
        u=Aop(v)-alfa*u;beta=u.norm();u=u/beta.clamp_min(1e-30)
        v=Atop(u)-beta*v;alfa=v.norm();v=v/alfa.clamp_min(1e-30)
        rb1=(rhobar**2+damp**2).sqrt();c1=rhobar/rb1;phibar=c1*phibar
        rho=(rb1**2+beta**2).sqrt();c=rb1/rho;s=beta/rho
        theta=s*alfa;rhobar=-c*alfa;phi=c*phibar;phibar=s*phibar
        x=x+(phi/rho)*w;w=v-(theta/rho)*w
    return x
def refineA(resid,a0,iters=50):
    wf=a0.clone();r=resid(wf);c=float(r@r);n=wf.numel();damp=1e-2
    for it in range(iters):
        _,vjpf=vjp(resid,wf);Jt=lambda u:vjpf(u)[0];Jv=lambda v:jvp(resid,(wf,),(v,))[1];ok=False
        for _ in range(12):
            dwf=lsqr(Jv,Jt,-r,n,damp);wn=wf+dwf;rn=resid(wn);cn=float(rn@rn)
            if cn<c: wf=wn;r=rn;c=cn;damp=max(damp*0.3,1e-9);ok=True;break
            damp*=4
        if not ok or c<1e-16:break
    return wf,c
mi=w2err(A_init); print(f"[anchor] TRUE B, oracle jets. init consensus W2: mean {mi[0]:.3e} max {mi[1]:.2e} <1%:{mi[2]}/80\n")
print("per-member: freeze member m's downstream, refine shared W2=AB only")
print(f"{'m':>2} {'W2 init->final mean':>22} {'max':>8} {'<1%':>5} {'heldout-loss':>13}")
for m,a in enumerate(aligned):
    D=(a["b2"],a["W3"],a["b3"],a["W4"],a["b4"],a["W5"],a["b5"])
    resid=lambda af,D=D:(observ(af.reshape(80,80),D,Htr)-tgt_tr)/vsc
    wf,c=refineA(resid,A_init.reshape(-1).clone()); mf=w2err(wf.reshape(80,80))
    ho=float(((observ(wf.reshape(80,80),D,Hho)-tgt_ho)/vsc).pow(2).mean())
    print(f"{m:>2} {mi[0]:.3e}->{mf[0]:.3e} {mf[1]:>8.2e} {mf[2]:>4}/80 {ho:>13.2e}",flush=True)
# ---- shared W2 across ALL 8 frozen downstreams simultaneously ----
Ds=[(a["b2"],a["W3"],a["b3"],a["W4"],a["b4"],a["W5"],a["b5"]) for a in aligned]
def resid_all(af):
    A=af.reshape(80,80); return torch.cat([(observ(A,D,Htr)-tgt_tr)/vsc for D in Ds])
wf,c=refineA(resid_all,A_init.reshape(-1).clone()); mf=w2err(wf.reshape(80,80))
print(f"\nSHARED W2 across all 8 frozen downstreams: init {mi[0]:.3e} -> final {mf[0]:.3e} max {mf[1]:.2e} <1%:{mf[2]}/80  loss {c:.2e}")
