"""CONTROLS to localize why honest L2 refine fails. Same seed + optimizer as l2_honest_refine.py;
the ONLY change per row is what downstream/b2 the model uses. If TRUE downstream solves W2 from the
same honest seed and GUESSED downstream doesn't, the downstream is the load-bearing cheat.
"""
import sys, torch
from torch.func import jvp, vjp
torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
dev = "cuda" if torch.cuda.is_available() else "cpu"
P = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
pk = torch.load(P + "peel_committee.pt", map_location=dev, weights_only=False)
dims=[784,128,80,40,32,10]
teacher=MLP(dims,act="sigmoid").to(dev); teacher.load_state_dict(pk["teacher_state"]); teacher.eval()
for p in teacher.parameters(): p.requires_grad_(False)
Wt=[teacher.layers[i].weight.detach() for i in range(5)]; bt=[teacher.layers[i].bias.detach() for i in range(5)]
W2t=Wt[1]; nt=W2t.norm(dim=1)
def oracle_fwd(h):
    q2=torch.sigmoid(h@Wt[1].t()+bt[1]);q3=torch.sigmoid(q2@Wt[2].t()+bt[2])
    q4=torch.sigmoid(q3@Wt[3].t()+bt[3]);return q4@Wt[4].t()+bt[4]

# honest guesses
def match(A,B):
    Cp=torch.cdist(A,B);Cm=torch.cdist(-A,B);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r);c=torch.tensor(c);s=torch.where(Cm[r,c]<Cp[r,c],-1.,1.);return r,c,s
mem=[(sd["layers.1.weight"].to(dev).double(),sd["layers.1.bias"].to(dev).double()) for sd in pk["pop_states"]]
W0,_=mem[0]; AW=[W0.clone()]
for Wm,_ in mem[1:]:
    r,c,s=match(Wm,W0); Wa=torch.zeros_like(Wm); Wa[c]=Wm[r]*s[:,None]; AW.append(Wa)
W2_median=torch.stack(AW).median(0).values.clone()     # truth-free consensus (16% mean)
sd0=pk["pop_states"][0]
W2_mem0=sd0["layers.1.weight"].to(dev).double().clone()  # single member (23% mean)
gd=[sd0[f"layers.{i}.weight"].to(dev).double() for i in (2,3,4)]; gdb=[sd0[f"layers.{i}.bias"].to(dev).double() for i in (2,3,4)]
gb2=sd0["layers.1.bias"].to(dev).double()

def model(h,W2,b2,W3,b3,W4,b4,W5,b5):
    q2=torch.sigmoid(h@W2.t()+b2);q3=torch.sigmoid(q2@W3.t()+b3);q4=torch.sigmoid(q3@W4.t()+b4);return q4@W5.t()+b5
def wscore(W2):
    Cp=torch.cdist(W2,W2t);Cm=torch.cdist(-W2,W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[t],ci[t]] for t in range(len(ri))]);rel=e/nt[ci].cpu()
    return float(rel.mean()),float(rel.max()),int((rel<0.01).sum())

N=400
gz=torch.Generator(device="cpu").manual_seed(0);scales=torch.tensor([1.5,4.0,10.0])
sidx=torch.randint(0,3,(N,),generator=gz);z=torch.randn(N,128,generator=gz)*scales[sidx][:,None]
h=torch.sigmoid(z).clamp(1e-5,1-1e-5).to(dev)
Qg=torch.Generator(device="cpu").manual_seed(3);Q,_=torch.linalg.qr(torch.randn(128,128,generator=Qg));Q=Q[:64].to(dev)
valt=oracle_fwd(h).detach();scv=valt.abs().max()
ddt=torch.func.vmap(lambda u:jvp(oracle_fwd,(h,),(u.expand(N,128),))[1])(Q).detach();scd=ddt.abs().max()
def cg(A,b,it=50,tol=1e-14):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rs=r@r
    for _ in range(it):
        Ap=A(p);a=rs/(p@Ap+1e-30);x=x+a*p;r=r-a*Ap;rs2=r@r
        if rs2.sqrt()<tol:break
        p=r+(rs2/rs)*p;rs=rs2
    return x
def refine(w0,resid,iters=60):
    wf=w0.clone();lam=1e-3;r=resid(wf);c=float(r@r)
    for it in range(iters):
        _,vjpf=vjp(resid,wf);Jt=lambda u:vjpf(u)[0];Jv=lambda v:jvp(resid,(wf,),(v,))[1]
        gvec=Jt(r);A=lambda v:Jt(Jv(v))+lam*v;ok=False
        for _ in range(12):
            dwf=cg(A,-gvec);wn=wf+dwf;rn=resid(wn);cn=float(rn@rn)
            if cn<c:wf=wn;r=rn;c=cn;lam=max(lam*0.3,1e-15);ok=True;break
            lam*=5
        if not ok or c<1e-28:break
    return wf,c
def run(seedW2, down, tag):
    b2,W3,b3,W4,b4,W5,b5 = down
    def resid(wf):
        W2=wf.reshape(80,128); fh=lambda hb: model(hb,W2,b2,W3,b3,W4,b4,W5,b5)
        val=model(h,W2,b2,W3,b3,W4,b4,W5,b5)
        dd=torch.func.vmap(lambda u:jvp(fh,(h,),(u.expand(N,128),))[1])(Q)
        return torch.cat([((val-valt)/scv).reshape(-1),((dd-ddt)/scd).reshape(-1)])
    m0=wscore(seedW2); wf,c=refine(seedW2.reshape(-1).clone(),resid); m1=wscore(wf.reshape(80,128))
    print(f"  {tag:44s} init {m0[0]:.3e}/{m0[1]:.3e} -> after {m1[0]:.3e}/{m1[1]:.3e}  <1%:{m1[2]}/80  loss {c:.2e}")

TRUE_down=(bt[1],Wt[2],bt[2],Wt[3],bt[3],Wt[4],bt[4])
GUESS_down=(gb2,gd[0],gdb[0],gd[1],gdb[1],gd[2],gdb[2])
print("W rel-row-err mean/max.  downstream = TRUE is the CHEAT; GUESS is honest.  (same optimizer/seed)")
run(W2_mem0,   TRUE_down,  "CHEAT  down=true , seed=member0 (23%)")
run(W2_median, TRUE_down,  "CHEAT  down=true , seed=median  (16%)")
run(W2_mem0,   GUESS_down, "HONEST down=guess, seed=member0 (23%)")
run(W2_median, GUESS_down, "HONEST down=guess, seed=median  (16%)")
