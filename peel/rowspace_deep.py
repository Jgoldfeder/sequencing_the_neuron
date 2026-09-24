"""Is first-layer row-space recovery true at ANY depth? Test on a 5-hidden-layer
sigmoid net. f = G(W1 x + b1) for arbitrary G, so grad f in row(W1) regardless."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
dev="cuda"; torch.manual_seed(0)

for dims in ([128,24,16,8],                          # 2 hidden
             [128,24,40,40,24,16,8],                 # 5 hidden layers
             [128,24,64,64,64,64,64,8]):             # 6 hidden, wide
    d=dims[0]; k=dims[1]
    teacher=MLP(dims,act="sigmoid").to(dev).double()
    with torch.no_grad():
        for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
    W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
    tn=W1t/W1t.norm(dim=1,keepdim=True)
    @torch.no_grad()
    def J_at(x, fd=1e-3):
        E=torch.eye(d,device=dev,dtype=torch.float64)
        return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()
    g=torch.Generator(device=dev).manual_seed(2)
    rows=[]; npts=max(8, k//dims[-1]+4)
    for _ in range(npts):
        x=torch.randn(d,generator=g,device=dev,dtype=torch.float64)*2
        rows.append(J_at(x))
    M=torch.cat(rows,0)
    U,S,Vh=torch.linalg.svd(M,full_matrices=False); basis=Vh[:k]
    capt=(tn@basis.t()).norm(dim=1)                  # 1 => row lies in measured subspace
    # row-projection refinement of a 5deg whole-layer guess
    th=math.radians(5.0); before=[]; after=[]
    for kk in range(k):
        u=tn[kk]; v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
        wg=math.cos(th)*u+math.sin(th)*v; wp=(wg@basis.t())@basis
        s=1.0 if float(wg@u)>0 else -1.0
        before.append(float((s*wg/wg.norm()-u).abs().max()))
        s2=1.0 if float(wp@u)>0 else -1.0
        after.append(float((s2*wp/wp.norm()-u).abs().max()))
    print(f"depth={len(dims)-1} hidden  arch={dims}  ({npts*2*d} queries):")
    print(f"   row space captures true first-layer rows: min {capt.min():.5f} med {capt.median():.5f}")
    print(f"   5deg guess max_eps {max(before):.2e} -> row-proj {max(after):.2e}  ({max(before)/max(after):.1f}x)\n")
