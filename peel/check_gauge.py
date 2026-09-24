"""GAUGE SANITY (reviewer): before trusting the 11.8% sealed row space, prove the L1 coordinate map is
internally exact in the committee gauge (weights AND bias), and the truth-free identity
h_committee(x(q)) = q holds. THEN a labeled DIAGNOSTIC swaps in true-b1 to test whether the unknown
b1 warp is what degrades sealed row-space recovery."""
import torch
import sys; sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
torch.set_default_dtype(torch.float64)
from peel.sealed_harness import load_blackbox, recovered_l1, honest_committee, recover_rowspace, score_rowspace, _match, dev, PEEL

bb,dims=load_blackbox()
l1r=recovered_l1(bb,dims)                 # sealed, merged gauge
comm=honest_committee(dims)
W1s=l1r["W1s"]; Lc=comm["L1w"]; Lb=comm["L1b"]
# ---- align recovered L1 -> committee gauge (both OURS); check WEIGHTS and BIAS ----
r,c,s=_match(W1s,Lc); r,c,s=r.to(dev),c.to(dev),s.to(dev)
perm=torch.empty(W1s.shape[0],dtype=torch.long,device=dev); perm[r]=c; inv=torch.argsort(perm)
W1a=(s[inv][:,None]*W1s[inv]).contiguous(); b1a=(s[inv]*l1r["b1_guess"][inv]).contiguous()
print(f"[gauge] ||W1a - L1_committee|| = {float((W1a-Lc).norm()):.2e}   ||b1a - b1_committee|| = {float((b1a-Lb).norm()):.2e}")
print(f"         (b1a relative to committee b1: {float((b1a-Lb).norm()/Lb.norm()):.2e})")
# ---- truth-free coordinate identity: h_committee(x(q)) == q ? using committee L1 as coordinate ----
W1c=Lc; b1c=Lb; W1cp=W1c.t()@torch.linalg.inv(W1c@W1c.t())
g=torch.Generator(device=dev).manual_seed(0)
q=torch.sigmoid(torch.randn(500,128,generator=g,device=dev)*2.0).clamp(1e-4,1-1e-4)
x=(torch.log(q/(1-q))-b1c)@W1cp.t(); hc=torch.sigmoid(x@W1c.t()+b1c)
print(f"[identity] ||h_committee(x(q)) - q||  mean {float((hc-q).abs().mean()):.2e}  max {float((hc-q).abs().max()):.2e}  (truth-free; should be ~0)")
# ---- sealed row space with committee-L1 coordinate (HONEST) ----
l1_comm={"W1s":W1c,"b1_guess":b1c,"W1sp":W1cp}
B=recover_rowspace(bb,l1_comm,comm); score_rowspace(B,"HONEST committee-L1 coord (b1=guess)")
# ================= DIAGNOSTIC (uses true b1 -- label clearly) =================
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False)
b1true=pk["teacher_state"]["layers.0.bias"].to(dev).double()
# map true b1 into committee L1 gauge: committee L1 == our solved L1 (values), so true b1 aligns by the
# committee-vs-true correspondence. committee L1 is already true-ordered (values match W1_true), so:
W1true=pk["teacher_state"]["layers.0.weight"].to(dev).double()
rr,cc,ss=_match(Lc,W1true); rr,cc,ss=rr.to(dev),cc.to(dev),ss.to(dev)
permc=torch.empty(128,dtype=torch.long,device=dev); permc[rr]=cc
b1true_cg=ss*b1true[permc]                # true b1 expressed in committee gauge
print(f"[diag] committee b1(guess) vs true b1 (committee gauge): ||.|| {float((b1c-b1true_cg).norm()):.2e} rel {float((b1c-b1true_cg).norm()/b1true_cg.norm()):.2e}")
l1_true={"W1s":W1c,"b1_guess":b1true_cg,"W1sp":W1cp}
Bt=recover_rowspace(bb,l1_true,comm); score_rowspace(Bt,"DIAGNOSTIC true-b1 coord")
