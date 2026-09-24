"""Is the honest 11.8% row space intrinsic or a FD/probe-count artifact? Sweep fd and #probes, using
the verified-exact committee-L1 coordinate (truth-free). Score against true rowspace (scoring only)."""
import torch
import sys; sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
torch.set_default_dtype(torch.float64)
from peel.sealed_harness import load_blackbox, honest_committee, dev, PEEL
bb,dims=load_blackbox(); comm=honest_committee(dims)
W1c=comm["L1w"]; b1c=comm["L1b"]; W1cp=W1c.t()@torch.linalg.inv(W1c@W1c.t())
W2g=comm["W2guess"]; b2g=comm["b2guess"]; W2gp=torch.linalg.pinv(W2g)
def hq(Q): return bb.query((torch.log(Q/(1-Q))-b1c)@W1cp.t())
def rowspace(nrep,fd):
    gq=torch.Generator(device=dev).manual_seed(5); rows=[]
    for rep in range(nrep):
        for j in range(80):
            tt=(torch.randn(80,generator=gq,device=dev)*2.0); tt[j]=0.0
            q0=torch.clamp(W2gp@(tt-b2g),1e-3,1-1e-3); E=torch.eye(128,device=dev)
            Jp=hq((q0+fd*E).clamp(1e-4,1-1e-4)); Jm=hq((q0-fd*E).clamp(1e-4,1-1e-4))
            rows.append(((Jp-Jm)/(2*fd)).t())
    M=torch.cat(rows,0); U,S,Vh=torch.linalg.svd(M,full_matrices=False); return Vh[:80], float(S[79]/S[80])
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False)
W2t=pk["teacher_state"]["layers.1.weight"].to(dev).double(); nt=W2t.norm(dim=1)
def sc(B): P=B.t()@B; r=((W2t-W2t@P).norm(dim=1)/nt); return float(r.mean()),float(r.max())
print(f"{'nrep':>5} {'fd':>8} {'gap S79/S80':>12} {'rowspace mean':>14} {'max':>8}")
for nrep,fd in [(2,1e-3),(2,3e-4),(2,1e-4),(2,3e-5),(4,1e-4),(8,1e-4)]:
    B,gap=rowspace(nrep,fd); m,mx=sc(B); print(f"{nrep:>5} {fd:>8.0e} {gap:>12.3e} {m:>14.3e} {mx:>8.2e}",flush=True)
