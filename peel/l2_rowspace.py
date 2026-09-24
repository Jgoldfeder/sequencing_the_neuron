"""L2 experiment 1 (v2): downstream-independent ROW-SPACE recovery, GUESS-TARGETED probes.
rowspan(J_Gtilde(q)) subset rowspan(W2) (dh/dq ~ I). Each probe's Jacobian is rank<=10 and only
carries rows of neurons that are EXCITED (sigma'(z2)!~0) there. In the bounded cube random q leave
many neurons saturated, so we place probes with the guess to excite each neuron (z2g_j~0), union the
Jacobian rowspaces -> rowspan(W2). Then project the committee guess in. Diagnostic: 16% -> ?

SEAL: teacher only via bb.query(x) on 784-D inputs during the solve. Truth loaded ONLY for scoring.
"""
import sys, torch, numpy as np
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
torch.set_default_dtype(torch.float64)
dev="cuda"; CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
class BlackBox:
    def __init__(self,net):
        net.eval()
        for p in net.parameters(): p.requires_grad_(False)
        def run(x):
            with torch.no_grad(): return net(x).detach().clone()
        object.__setattr__(self,"_run",run); object.__setattr__(self,"n",0)
    def query(self,x):
        object.__setattr__(self,"n",self.n+(x.shape[0] if x.dim()>1 else 1)); return self._run(x)
    __call__=query
    def __getattr__(self,k): raise AttributeError(f"query-only; '{k}' forbidden")
pop=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False)
merged=torch.load(CD+"mergedbest512_sigmoid__784x128x80x40x32x10__s0_final.pt",map_location=dev,weights_only=False)
dims=pop["dims"]; d,k,O=dims[0],dims[1],dims[-1]
_t=MLP(dims,act="sigmoid").to(dev).double(); _t.load_state_dict(pop["teacher_state"]); _t.eval()
bb=BlackBox(_t); del _t
# ---- coordinate basis = committee's OWN frozen layer-0 (same gauge as its W2 guess) ----
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False)
mem=[{kk:vv.to(dev).double() for kk,vv in sd.items()} for sd in pk["pop_states"]]
W1s=mem[0]["layers.0.weight"].clone(); b1_guess=mem[0]["layers.0.bias"].clone()
W1sp=W1s.t()@torch.linalg.inv(W1s@W1s.t())
# quick quality check of the committee's frozen L1 (scoring)
_W1t=pop["teacher_state"]["layers.0.weight"].to(dev).double(); _nt=_W1t.norm(dim=1)
_Cp=torch.cdist(W1s,_W1t);_Cm=torch.cdist(-W1s,_W1t);_C=torch.minimum(_Cp,_Cm).cpu().numpy()
_ri,_ci=linear_sum_assignment(_C); _e=torch.tensor([_C[_ri[t],_ci[t]] for t in range(len(_ri))])/_nt[_ci].cpu()
print(f"[setup] committee frozen L1 quality vs truth: mean {float(_e.mean()):.3e} max {float(_e.max()):.3e}")
def match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r);c=torch.tensor(c);s=torch.where(Cm[r,c]<Cp[r,c],-1.,1.);return r,c,s
ref=mem[0]["layers.1.weight"]; AA=[ref.clone()]
for sd in mem[1:]:
    Wm=sd["layers.1.weight"];r,c,s=match(Wm,ref);Wa=torch.zeros_like(Wm);Wa[c]=Wm[r]*s[:,None];AA.append(Wa)
W2g=torch.stack(AA).median(0).values.clone(); b2g=mem[0]["layers.1.bias"].clone(); W2gp=torch.linalg.pinv(W2g)
# ---- probes: guess-targeted (excite each neuron) + random ----
def xofq(Q): return (torch.log(Q/(1-Q))-b1_guess)@W1sp.t()
def Jq(q0,eps=5e-4):
    E=torch.eye(k,device=dev,dtype=torch.float64)
    Qp=(q0+eps*E).clamp(1e-4,1-1e-4); Qm=(q0-eps*E).clamp(1e-4,1-1e-4)
    return ((bb.query(xofq(Qp))-bb.query(xofq(Qm)))/(2*eps)).t()
gq=torch.Generator(device=dev).manual_seed(5); probes=[]
for rep in range(2):
    for j in range(k if False else 80):
        t=(torch.randn(80,generator=gq,device=dev)*2.0); t[j]=0.0
        probes.append(torch.clamp(W2gp@(t-b2g),1e-3,1-1e-3))
for _ in range(40):
    scl=float(torch.tensor([1.0,2.5],device=dev)[torch.randint(0,2,(1,),generator=gq,device=dev)])
    probes.append(torch.sigmoid(torch.randn(k,generator=gq,device=dev)*scl).clamp(1e-3,1-1e-3))
rows=[Jq(q0) for q0 in probes]
M=torch.cat(rows,0)
Um,Sm,Vm=torch.linalg.svd(M,full_matrices=False)
B=Vm[:80]; P=B.t()@B
sv=Sm.cpu()
print(f"[rowspace] {len(probes)} probes, M={tuple(M.shape)}; S[60,70,79,80,90]={float(sv[60]):.2e},{float(sv[70]):.2e},{float(sv[79]):.2e},{float(sv[80]):.2e},{float(sv[90]):.2e}")
# ================= SCORING (truth unlocked) =================
W2t=pop["teacher_state"]["layers.1.weight"].to(dev).double(); b2t=pop["teacher_state"]["layers.1.bias"].to(dev).double(); nt=W2t.norm(dim=1)
# coverage diagnostic: per-neuron max sigma'(z2) across probes (truth)
Qs=torch.stack(probes); Z2=Qs@W2t.t()+b2t; sp=torch.sigmoid(Z2)*(1-torch.sigmoid(Z2)); maxsp=sp.max(0).values
print(f"[coverage] neurons with max sigma'(z2)<0.05 across probes: {int((maxsp<0.05).sum())}/80  (these rows are unrecoverable from these probes)")
res_true=((W2t-W2t@P).norm(dim=1)/nt)
print(f"[rowspace quality] ||W2t - W2t P||/||W2t|| per-row: mean {float(res_true.mean()):.3e} max {float(res_true.max()):.3e}  rows>10%: {int((res_true>0.1).sum())}")
def rowerr(W):
    Cp=torch.cdist(W,W2t);Cm=torch.cdist(-W,W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[t],ci[t]] for t in range(len(ri))]);rel=e/nt[ci].cpu(); return float(rel.mean()),float(rel.max())
mg,xg=rowerr(W2g); mp,xp=rowerr(W2g@P)
print(f"\nGUESS row error (truth-free committee): mean {mg:.3e}  max {xg:.3e}")
print(f"PROJECTED into recovered rowspace     : mean {mp:.3e}  max {xp:.3e}")
print(f"[queries total: {bb.n}]")
