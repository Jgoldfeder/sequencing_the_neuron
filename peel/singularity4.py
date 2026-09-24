"""Push the singularity method: density-cluster poles across many contexts (the
first-layer pole recurs at ONE location; downstream poles wander), then iterate
(rebuild W1 with refined a,b -> better isolation -> re-probe)."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch, numpy as np
from scipy.interpolate import AAA
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True); g=torch.Generator(device=dev).manual_seed(1)
@torch.no_grad()
def J_at(x,fd=5e-5):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()
Wg=torch.empty(k,d,device=dev,dtype=torch.float64); bg0=torch.empty(k,device=dev,dtype=torch.float64)
for j in range(k):
    u=tn[j]; v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
    Wg[j]=(math.cos(math.radians(5))*u+math.sin(math.radians(5))*v)*W1t[j].norm()*(1+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64)-1))
    bg0[j]=b1t[j]+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64).item()-1)
Wg_pinv=Wg.t()@torch.linalg.inv(Wg@Wg.t())
# directions (SVD) -- exact, once
N=torch.empty_like(Wg)
for j in range(k):
    t=torch.full((k,),20.0,device=dev,dtype=torch.float64); t[j]=0.0
    x0=Wg_pinv@(t-bg0); U,Sv,Vh=torch.linalg.svd(J_at(x0),full_matrices=False)
    nv=Vh[0]; N[j]=nv if float(nv@Wg[j])>0 else -nv

def support_center(P, ctx, r):
    """center of the pole cluster with the most DISTINCT-context support
    (the first-layer pole recurs across contexts; downstream poles don't)."""
    if len(P)==0: return None
    P=np.array(P); ctx=np.array(ctx)
    sup=np.array([len(np.unique(ctx[np.abs(P-p)<r])) for p in P])
    best=P[sup.argmax()]; m=np.abs(P-best)<r
    return complex(np.median(P[m].real), np.median(P[m].imag))

S=20.0; M=28; L=3.2; NP=180
tanp=(np.cos(np.pi*(np.arange(NP)+0.5)/NP)*L)
ta=torch.tensor(tanp,device=dev,dtype=torch.float64)

# a_cur/b_cur: current best estimates of magnitude & bias (init: guess)
a_cur=Wg.norm(dim=1).clone(); b_cur=bg0.clone()
for it in range(2):
    W1cur=N*a_cur[:,None]; W1cur_pinv=W1cur.t()@torch.linalg.inv(W1cur@W1cur.t())
    a_new=a_cur.clone(); b_new=b_cur.clone()
    for j in range(k):
        im_exp=math.pi/float(a_cur[j]); nx=-float(b_cur[j])/float(a_cur[j])
        allp=[]; allc=[]
        for m in range(M):
            sg=(torch.randint(0,2,(k,),generator=g,device=dev)*2-1).double()*S; sg[j]=0.0
            xm=W1cur_pinv@(sg-b_cur)
            pts=xm.unsqueeze(0)+ta.unsqueeze(1)*N[j].unsqueeze(0)
            with torch.no_grad(): F=teacher(pts)
            for _ in range(4):
                q=torch.randn(dims[-1],generator=g,device=dev,dtype=torch.float64)
                try: poles=AAA(tanp,(F@q).cpu().numpy()).poles()
                except Exception: continue
                sel=poles[(poles.imag>0.5*im_exp)&(poles.imag<1.8*im_exp)&(np.abs(poles.real)<0.6)]
                allp.extend(sel.tolist()); allc.extend([m]*len(sel))
        c=support_center(allp, allc, 0.05*im_exp) if allp else None
        if c is None: continue
        a_j=math.pi/c.imag; a_new[j]=a_j; b_new[j]=-a_j*(c.real+nx)
    # report this iteration
    me=(a_new-W1t.norm(dim=1)).abs(); be=(b_new-b1t).abs()
    de=torch.tensor([math.degrees(math.acos(min(1.0,abs(float(N[j]@tn[j]))))) for j in range(k)])
    print(f"iter {it}: dir worst {de.max():.2e}deg | mag worst {me.max():.3e} med {me.median():.3e} "
          f"| bias worst {be.max():.3e} med {be.median():.3e}")
    a_cur=a_new; b_cur=b_new
