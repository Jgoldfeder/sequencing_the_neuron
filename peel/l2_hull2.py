"""HULL diagnostic, HARDENED (reviewer's two caveats):
 (1) TRUTH-FREE correspondence: align members to MEMBER-0 (the gauge we can actually build), score vs
     truth only afterward. If the 5% bridge survives -> real; if it jumps to 10-15% -> correspondence
     is still unsolved.
 (2) coefficient stability: affine (sum a=1) allows wild extrapolation. Print ||a||_1,||a||_2,max|a|,
     and compare affine vs CONVEX hull (a>=0) vs REGULARIZED affine. If the bridge survives with modest
     coeffs it's much stronger.
Oracle target used only to score/measure best-achievable (this is a diagnostic).
"""
import sys, torch, numpy as np
torch.set_default_dtype(torch.float64)
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
dev="cuda"; CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
pop=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False)
dims=pop["dims"]
t=MLP(dims,act="sigmoid").to(dev).double(); t.load_state_dict(pop["teacher_state"]); t.eval()
W2t=t.layers[1].weight.detach(); nt=W2t.norm(dim=1); Uw,Sw,Vh=torch.linalg.svd(W2t,full_matrices=False); Pb=Vh[:80].t()@Vh[:80]
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False)
mem=[{kk:vv.to(dev).double() for kk,vv in sd.items()} for sd in pk["pop_states"]]
M=len(mem)
def match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r);c=torch.tensor(c);return r,c,torch.where(Cm[r,c]<Cp[r,c],-1.,1.)
def align_to(refW):
    out=[]
    for sd in mem:
        W2m=sd["layers.1.weight"]; r,c,s=match(W2m,refW); Wa=torch.zeros_like(W2m); Wa[c]=W2m[r]*s[:,None]; out.append(Wa)
    return torch.stack(out)                      # (8,80,128) in refW's gauge
def score(rec):                                   # Hungarian sign-aware to truth
    Cp=torch.cdist(rec,W2t);Cm=torch.cdist(-rec,W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);rel=e/nt[ci].cpu()
    return float(rel.mean()),float(rel.max()),int((rel<0.01).sum()),int((rel<0.03).sum()),int((rel<0.05).sum())
def pr(tag,rec):
    m=score(rec); print(f"{tag:36s} mean {m[0]:.3e} max {m[1]:.3e} | <1%:{m[2]:>2} <3%:{m[3]:>2} <5%:{m[4]:>2}/80")
ones=torch.ones(M,device=dev); a0=ones/M
_,_,V8=torch.linalg.svd(ones.reshape(1,M)); Z=V8[1:].t()      # (8,7)
def affine_fit(cand,tgt,lam=0.0):                 # cand (80,8,128), tgt (80,128); returns rec, alpha
    A=cand.transpose(1,2)                          # (80,128,8)
    AZ=A@Z; rhs=tgt-A@a0
    # (Z^T A^T A Z + lam I) beta = (AZ)^T rhs
    G=AZ.transpose(1,2)@AZ + lam*torch.eye(7,device=dev)[None]
    beta=torch.linalg.solve(G, (AZ.transpose(1,2)@rhs.unsqueeze(-1))).squeeze(-1)
    alpha=a0+beta@Z.t()
    rec=torch.einsum('jdc,jc->jd',A,alpha)
    return rec,alpha
def simplex_proj(v):                              # batched projection onto {a>=0, sum=1}
    u,_=torch.sort(v,dim=1,descending=True); css=u.cumsum(1)-1
    idx=torch.arange(1,M+1,device=dev)[None].double()
    cond=u-css/idx>0; rho=cond.double().sum(1,keepdim=True)
    theta=torch.gather(css,1,(rho.long()-1)).squeeze(1)/rho.squeeze(1)
    return torch.clamp(v-theta[:,None],min=0)
def convex_fit(cand,tgt,iters=400,lr=0.5):
    A=cand.transpose(1,2); alpha=a0.repeat(80,1)
    AtA=A.transpose(1,2)@A; Atb=(A.transpose(1,2)@tgt.unsqueeze(-1)).squeeze(-1)
    L=torch.linalg.matrix_norm(AtA,ord=2).max()
    for _ in range(iters):
        grad=(AtA@alpha.unsqueeze(-1)).squeeze(-1)-Atb
        alpha=simplex_proj(alpha-(lr/L)*grad)
    rec=torch.einsum('jdc,jc->jd',A,alpha); return rec,alpha
def cstat(alpha,tag):
    print(f"   coeff {tag}: |a|_1 mean {float(alpha.abs().sum(1).mean()):.2f} max {float(alpha.abs().sum(1).max()):.2f}"
          f" | max|a_m| mean {float(alpha.abs().max(1).values.mean()):.2f} max {float(alpha.abs().max(1).values.max()):.2f}")

print("=== ORACLE correspondence (each member aligned to TRUTH) -- the earlier optimistic number ===")
Wor=align_to(W2t); cand_or=Wor.permute(1,0,2)
rec,al=affine_fit(cand_or,W2t); pr("oracle per-row affine",rec)

print("\n=== TRUTH-FREE correspondence (members aligned to MEMBER-0) ===")
Wtf=align_to(mem[0]["layers.1.weight"]); cand_tf=Wtf.permute(1,0,2)
pr("median", Wtf.median(0).values)
# oracle TARGET per member-0 neuron = the true neuron member-0 represents
r0,c0,s0=match(mem[0]["layers.1.weight"],W2t); tgt=torch.zeros_like(W2t); tgt[r0]=s0[:,None]*W2t[c0]  # member-0 order
rec_aff,al_aff=affine_fit(cand_tf,tgt); pr("per-row affine (truth-free)",rec_aff); cstat(al_aff,"affine")
rec_cvx,al_cvx=convex_fit(cand_tf,tgt); pr("per-row CONVEX hull (a>=0)",rec_cvx); cstat(al_cvx,"convex")
for lam in [0.1,1.0]:
    rec_r,al_r=affine_fit(cand_tf,tgt,lam=lam); pr(f"per-row affine +reg lam={lam}",rec_r); cstat(al_r,f"reg{lam}")
pr("per-row affine (truth-free)+Bproj", rec_aff@Pb)
print("\ninterpretation: compare truth-free affine vs oracle affine (correspondence cost);")
print("compare affine vs convex + coeff norms (extrapolation cost).")
