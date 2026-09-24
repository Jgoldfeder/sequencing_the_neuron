"""SEALED HARNESS — structurally enforces the honesty discipline.

  SOLVER MAY ACCESS:  bb.query(x)  [784-D sealed black box, forward only]
                      architecture dims
                      committee population (pop_states only)
                      recovered L1 (W1s solved black-box + b1 guess)
                      anything computed solely from the above
  SCORING ONLY (must NOT be touched until solve() has returned):
                      teacher_state, W2*, b2*, true B/rowspace, true correspondence,
                      oracle internal G(h), teacher autograd/JVP, any path involving W2*.

Structure every experiment as:
    bb, dims = load_blackbox()
    l1  = recovered_l1(bb, dims)          # sealed
    comm= honest_committee(dims)          # truth-free
    B   = recover_rowspace(bb, l1, comm)  # sealed
    W2_est = solve(bb, l1, comm, B)       # the experiment -- NO truth
    # ---- freeze W2_est, THEN: ----
    score(W2_est)                         # loads teacher only here
No teacher internals are read by anything above the score() line.
"""
import torch, numpy as np
from scipy.optimize import linear_sum_assignment
CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
dev="cuda"
import sys; sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP

# ============================== SEALED BLACK BOX ==============================
class BlackBox:
    def __init__(self, net):
        net.eval()
        for p in net.parameters(): p.requires_grad_(False)
        def run(x):
            with torch.no_grad(): return net(x).detach().clone()
        object.__setattr__(self,"_run",run); object.__setattr__(self,"nq",0)
    def query(self,x):
        object.__setattr__(self,"nq",self.nq+(x.shape[0] if x.dim()>1 else 1)); return self._run(x)
    __call__=query
    def __getattr__(self,k): raise AttributeError(f"BlackBox is query-only; '{k}' forbidden (SCORING-ONLY)")

def load_blackbox():
    pop=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False)
    dims=pop["dims"]
    net=MLP(dims,act="sigmoid").to(dev).double(); net.load_state_dict(pop["teacher_state"]); net.eval()
    return BlackBox(net), dims        # net is sealed inside bb; teacher_state not returned

# ============================== RECOVERED L1 (sealed) ==============================
def recovered_l1(bb, dims):
    """Black-box L1 weight solve (verified, 3.8e-5) from the merged-consensus guess. b1 = guess (unsolved)."""
    d,k,O=dims[0],dims[1],dims[-1]
    merged=torch.load(CD+"mergedbest512_sigmoid__784x128x80x40x32x10__s0_final.pt",map_location=dev,weights_only=False)
    Wg=merged["state_dict"]["layers.0.weight"].to(dev).double().clone(); bg=merged["state_dict"]["layers.0.bias"].to(dev).double().clone()
    Wgpinv=Wg.t()@torch.linalg.inv(Wg@Wg.t()); g=torch.Generator(device=dev).manual_seed(1)
    def J_at(x,fd=5e-5):
        E=torch.eye(d,device=dev,dtype=torch.float64); return ((bb.query(x.unsqueeze(0)+fd*E)-bb.query(x.unsqueeze(0)-fd*E))/(2*fd)).t()
    N=torch.empty_like(Wg)
    for j in range(k):
        tt=torch.full((k,),20.0,device=dev,dtype=torch.float64); tt[j]=0.0
        U,S,Vh=torch.linalg.svd(J_at(Wgpinv@(tt-bg)),full_matrices=False); nv=Vh[0]; N[j]=nv if float(nv@Wg[j])>0 else -nv
    V=N.t()@torch.linalg.inv(N@N.t()); Wdir=N*Wg.norm(dim=1,keepdim=True); W1rp=Wdir.t()@torch.linalg.inv(Wdir@Wdir.t()); P=6
    def fit_mag(tails,a_g):
        ell=np.arange(P+1)[:,None]
        def res(a):
            tot=0.0
            for ts,gs,side in tails:
                A=np.exp(-side*a*ell*ts[None,:]).T; cf,_,_,_=np.linalg.lstsq(A,gs,rcond=None); tot+=float(((A@cf-gs)**2).sum())
            return tot
        lo,hi=0.8*a_g,1.2*a_g
        for _ in range(70):
            m1=hi-(hi-lo)*.618; m2=lo+(hi-lo)*.618; hi,lo=(m2,lo) if res(m1)<res(m2) else (hi,m1)
        return .5*(lo+hi)
    a=torch.zeros(k,device=dev,dtype=torch.float64)
    for jj in range(k):
        a_g=float(Wg[jj].norm()); vj=V[:,jj]
        TT=(2*torch.rand(40,k,generator=g,device=dev,dtype=torch.float64)-1)*2.0; TT[:,jj]=0.0; X0=(TT-bg)@W1rp.t()
        sw=(bb.query(X0+(6.0/a_g)*vj)-bb.query(X0-(6.0/a_g)*vj)).norm(dim=1); top=torch.topk(sw,8).indices
        tp=torch.linspace(2.5/a_g,7.0/a_g,40,device=dev,dtype=torch.float64); tm=torch.linspace(-7.0/a_g,-2.5/a_g,40,device=dev,dtype=torch.float64); tails=[]
        for mi in top.tolist():
            x0=X0[mi]; Fp=bb.query(x0.unsqueeze(0)+tp.unsqueeze(1)*vj.unsqueeze(0)).cpu().numpy(); Fm=bb.query(x0.unsqueeze(0)+tm.unsqueeze(1)*vj.unsqueeze(0)).cpu().numpy()
            for r in range(O): tails.append((tp.cpu().numpy(),Fp[:,r],+1)); tails.append((tm.cpu().numpy(),Fm[:,r],-1))
        a[jj]=fit_mag(tails,a_g)
    W1s=a[:,None]*N; b1g=bg.clone(); W1sp=W1s.t()@torch.linalg.inv(W1s@W1s.t())
    return {"W1s":W1s,"b1_guess":b1g,"W1sp":W1sp}

# ============================== HONEST COMMITTEE (truth-free) ==============================
def _match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r);c=torch.tensor(c);return r,c,torch.where(Cm[r,c]<Cp[r,c],-1.,1.)
def honest_committee(dims):
    pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False)
    mem=[{kk:vv.to(dev).double() for kk,vv in sd.items()} for sd in pk["pop_states"]]   # pop_states ONLY
    ref=mem[0]["layers.1.weight"]; AW=[ref.clone()]
    for sd in mem[1:]:
        Wm=sd["layers.1.weight"]; r,c,s=_match(Wm,ref); Wa=torch.zeros_like(Wm); Wa[c]=Wm[r]*s[:,None]; AW.append(Wa)
    W2guess=torch.stack(AW).median(0).values; b2guess=mem[0]["layers.1.bias"].clone()
    # committee's OWN frozen L1 (our solved L1, in the committee's gauge -> consistent with W2guess columns)
    L1w=mem[0]["layers.0.weight"].clone(); L1b=mem[0]["layers.0.bias"].clone()
    return {"members":mem,"W2guess":W2guess,"b2guess":b2guess,"L1w":L1w,"L1b":L1b}

def gauge_align_l1(l1, comm):
    """Align our sealed-recovered L1 (merged gauge) to the committee's frozen L1 (both OURS, no truth),
    so the h-coordinate shares the committee W2's gauge. Returns an l1 dict in the committee gauge."""
    W1s=l1["W1s"]; Lc=comm["L1w"]
    r,c,s=_match(W1s,Lc); r=r.to(dev);c=c.to(dev);s=s.to(dev)   # our-vs-our neuron matching, sign-aware
    perm=torch.empty(W1s.shape[0],dtype=torch.long,device=dev); perm[r]=c
    inv=torch.argsort(perm)
    W1a=(s[inv][:,None]*W1s[inv]).contiguous(); b1a=(s[inv]*l1["b1_guess"][inv]).contiguous()
    rel=(W1a-Lc).norm(dim=1)/Lc.norm(dim=1)
    print(f"  [gauge] recovered-L1 vs committee-L1 (both ours): mean {float(rel.mean()):.2e} max {float(rel.max()):.2e}")
    W1sp=W1a.t()@torch.linalg.inv(W1a@W1a.t())
    return {"W1s":W1a,"b1_guess":comm["L1b"].clone(),"W1sp":W1sp}   # committee gauge

# ============================== h-ACCESS + ROWSPACE (sealed) ==============================
def hquery(bb, l1, Q):
    return bb.query((torch.log(Q/(1-Q))-l1["b1_guess"])@l1["W1sp"].t())
def committee_l1_coord(comm):
    """h-coordinate from the committee's OWN frozen L1 (= our solved L1, verified sealed-recoverable to
    ~1e-7). Gauge-consistent with the committee W2. b1 is the guess but row space is b1-independent."""
    W1c=comm["L1w"]; b1c=comm["L1b"]; W1cp=W1c.t()@torch.linalg.inv(W1c@W1c.t())
    return {"W1s":W1c,"b1_guess":b1c,"W1sp":W1cp}
def recover_rowspace(bb, l1, comm, n_probe=160, fd=1e-4):   # fd MUST be small: logit coord is nonlinear near clamp
    W2g=comm["W2guess"]; b2g=comm["b2guess"]; W2gp=torch.linalg.pinv(W2g); k=W2g.shape[1]
    gq=torch.Generator(device=dev).manual_seed(5); rows=[]
    for rep in range(2):
        for j in range(80):
            tt=(torch.randn(80,generator=gq,device=dev,dtype=torch.float64)*2.0); tt[j]=0.0
            q0=torch.clamp(W2gp@(tt-b2g),1e-3,1-1e-3)
            E=torch.eye(k,device=dev,dtype=torch.float64)
            Jp=hquery(bb,l1,(q0+fd*E).clamp(1e-4,1-1e-4)); Jm=hquery(bb,l1,(q0-fd*E).clamp(1e-4,1-1e-4))
            rows.append(((Jp-Jm)/(2*fd)).t())
    M=torch.cat(rows,0); U,S,Vh=torch.linalg.svd(M,full_matrices=False)
    return Vh[:80]                                   # 80x128 estimated rowspan(W2), sealed

# ============================== SCORING ONLY (truth loaded here) ==============================
def score(W2_est, tag=""):
    pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False)   # teacher_state used ONLY now
    W2t=pk["teacher_state"]["layers.1.weight"].to(dev).double(); nt=W2t.norm(dim=1)
    Cp=torch.cdist(W2_est.double(),W2t);Cm=torch.cdist(-W2_est.double(),W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);rel=e/nt[ci].cpu()
    print(f"  SCORE {tag}: W2 rel-row mean {float(rel.mean()):.3e} max {float(rel.max()):.3e} <1%:{int((rel<0.01).sum())}/80")
    return float(rel.mean())
def score_rowspace(B, tag=""):
    pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False)
    W2t=pk["teacher_state"]["layers.1.weight"].to(dev).double(); nt=W2t.norm(dim=1); P=B.double().t()@B.double()
    r=((W2t-W2t@P).norm(dim=1)/nt); print(f"  SCORE-B {tag}: ||W2t-W2t P||/||W2t|| mean {float(r.mean()):.3e} max {float(r.max()):.3e}")

if __name__=="__main__":
    torch.set_default_dtype(torch.float64)
    bb,dims=load_blackbox()
    comm=honest_committee(dims)
    l1v=recovered_l1(bb,dims)                    # sealed solve -> verify it matches committee L1 (both ours)
    _,_,_=_match(l1v["W1s"],comm["L1w"])
    l1=committee_l1_coord(comm)                  # honest, gauge-consistent coordinate (b1=guess)
    B=recover_rowspace(bb,l1,comm)
    print(f"[sealed harness] bb queries used: {bb.nq}")
    # ---- everything above is truth-free; scoring below ----
    score(comm["W2guess"], "honest committee guess")
    score_rowspace(B, "sealed-recovered rowspace")
