"""Honest before/after layer-1 max_eps for sigmoid refinement, with base-point
averaging (the single-base estimate is noisy; averaging M bases cuts the random
part ~1/sqrt(M)). Good guess -> refine each neuron -> layer max_eps before/after."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP

dev = "cuda"
pop = torch.load("recon/_pop__v18_lbfgs_sigmoid__3072x256x100__s0.pt", map_location=dev, weights_only=False)
dims = pop["dims"]; d = dims[0]
teacher = MLP(dims, act="sigmoid").to(dev).double(); teacher.load_state_dict(pop["teacher_state"])
fin = torch.load("recon/v18_lbfgs_sigmoid__3072x256x100__s0_final.pt", map_location=dev, weights_only=False)
guess = MLP(dims, act="sigmoid").to(dev).double(); guess.load_state_dict(fin["state_dict"])
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
W1g=guess.layers[0].weight.detach(); b1g=guess.layers[0].bias.detach(); W2g=guess.layers[1].weight.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True); gn=W1g/W1g.norm(dim=1,keepdim=True)
def sig_p(z): s=torch.sigmoid(z); return s*(1-s)
@torch.no_grad()
def J_at(x, fd=5e-4):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()
def maxeps(u,tk):
    s=1.0 if float(u@tn[tk])>0 else -1.0; return float((s*u/u.norm()-tn[tk]).abs().max())

@torch.no_grad()
def refine(tk, w_guess, M, gen):
    wg=w_guess*W1t[tk].norm(); gk=int((gn@tn[tk]).abs().argmax())
    acc=torch.zeros(d,device=dev,dtype=torch.float64); q=0; used=0; tries=0
    while used<M and tries<4*M:
        tries+=1
        x0=torch.randn(d,generator=gen,device=dev,dtype=torch.float64)
        x0=x0-(wg@x0+b1t[tk])/(wg@wg)*wg
        # skip base points too close to another hyperplane (bad isolation)
        zn=((W1t@x0+b1t)/W1t.norm(dim=1)).abs(); zn[tk]=9
        if float(zn.min())<0.01: continue
        used+=1
        J=J_at(x0); q+=2*d
        sp=sig_p(W1g@x0+b1g); contrib=(W2g*sp.unsqueeze(0))@W1g
        ck=sp[gk]*torch.outer(W2g[:,gk],W1g[gk])
        U,S,Vh=torch.linalg.svd(J-(contrib-ck),full_matrices=False)
        w=Vh[0]; acc=acc+(w if float(w@w_guess)>0 else -w)
    return acc/acc.norm(), q

N, M = 24, 24
gen=torch.Generator(device=dev).manual_seed(7)
for start in (5.0, 2.0):
    th=math.radians(start); before,after,qs=[],[],[]
    for tk in range(N):
        v=torch.randn(d,generator=gen,device=dev,dtype=torch.float64); v=v-(v@tn[tk])*tn[tk]; v=v/v.norm()
        w_guess=math.cos(th)*tn[tk]+math.sin(th)*v
        before.append(maxeps(w_guess,tk))
        wr,q=refine(tk,w_guess,M,gen); after.append(maxeps(wr,tk)); qs.append(q)
    print(f"good guess perturbed {start}deg, refine over ~{M} bases ({N} neurons):")
    print(f"   BEFORE layer-1 max_eps = {max(before):.3e}  (median {sorted(before)[N//2]:.3e})")
    print(f"   AFTER  layer-1 max_eps = {max(after):.3e}  (median {sorted(after)[N//2]:.3e})")
    print(f"   -> {max(before)/max(after):.0f}x smaller | ~{sum(qs)//N} queries/neuron\n")
print(f"[context] the run's staged solve got layer-1 max_eps ~3.25e-4; the model's")
print(f"          overall 2.9e-2 is all in the layer-2 bias (refining L1 won't touch it).")
