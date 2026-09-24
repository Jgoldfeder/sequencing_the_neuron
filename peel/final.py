"""FINAL: full first-layer refinement, counting every black-box query.
Reports before/after MAX weight+bias error and total samples."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch, numpy as np
from scipy.interpolate import AAA
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]; O=dims[-1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True); g=torch.Generator(device=dev).manual_seed(1)

QCOUNT=[0]
@torch.no_grad()
def bb(x):                                # THE black box; counts every input row
    QCOUNT[0]+=int(x.shape[0]); return teacher(x)
@torch.no_grad()
def J_at(x,fd=5e-5):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((bb(x.unsqueeze(0)+fd*E)-bb(x.unsqueeze(0)-fd*E))/(2*fd)).t()

Wg=torch.empty(k,d,device=dev,dtype=torch.float64); bg0=torch.empty(k,device=dev,dtype=torch.float64)
for j in range(k):
    u=tn[j]; v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
    Wg[j]=(math.cos(math.radians(5))*u+math.sin(math.radians(5))*v)*W1t[j].norm()*(1+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64)-1))
    bg0[j]=b1t[j]+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64).item()-1)
Wg_pinv=Wg.t()@torch.linalg.inv(Wg@Wg.t())

def per_neuron_err(W,b):                  # per-neuron max(|w err|, |b err|), sign-aligned
    out=[]
    for j in range(k):
        s=1.0 if float((W[j]/W[j].norm())@tn[j])>0 else -1.0
        out.append(max(float((s*W[j]-W1t[j]).abs().max()), abs(s*float(b[j])-float(b1t[j]))))
    return torch.tensor(out)
def full_max_err(W,b): return float(per_neuron_err(W,b).max())
def rpt(tag,W,b):
    e=per_neuron_err(W,b); print(f"{tag}: max {float(e.max()):.3e}  mean {float(e.mean()):.3e}  median {float(e.median()):.3e}")
rpt("BEFORE ",Wg,bg0)

# direction (SVD)
N=torch.empty_like(Wg)
for j in range(k):
    t=torch.full((k,),20.0,device=dev,dtype=torch.float64); t[j]=0.0
    x0=Wg_pinv@(t-bg0); U,Sv,Vh=torch.linalg.svd(J_at(x0),full_matrices=False)
    nv=Vh[0]; N[j]=nv if float(nv@Wg[j])>0 else -nv

S=20.0; M=24; L=3.1; NP=170; delta=1e-3
tanp=np.cos(np.pi*(np.arange(NP)+0.5)/NP)*L; ta=torch.tensor(tanp,device=dev,dtype=torch.float64)
a_cur=Wg.norm(dim=1).clone(); b_cur=bg0.clone()
for it in range(3):
    W1c=N*a_cur[:,None]; W1c_pinv=W1c.t()@torch.linalg.inv(W1c@W1c.t())
    a_new=a_cur.clone(); b_new=b_cur.clone()
    for j in range(k):
        im_exp=math.pi/float(a_cur[j]); nx=-float(b_cur[j])/float(a_cur[j])
        SG=(torch.randint(0,2,(400,k),generator=g,device=dev)*2-1).double()*S; SG[:,j]=0.0
        Xm=(SG-b_cur)@W1c_pinv.t()
        vis=((bb(Xm+delta*N[j])-bb(Xm-delta*N[j]))/(2*delta)).norm(dim=1)
        top=torch.topk(vis,M).indices
        cand=[]; ctx=[]
        for ci,mi in enumerate(top.tolist()):
            F=bb(Xm[mi].unsqueeze(0)+ta.unsqueeze(1)*N[j].unsqueeze(0)).cpu().numpy()
            for r in range(O):
                try: poles=AAA(tanp,F[:,r]).poles()
                except Exception: continue
                sel=poles[(poles.imag>0.5*im_exp)&(poles.imag<1.8*im_exp)&(np.abs(poles.real)<0.6)]
                cand.extend(sel.tolist()); ctx.extend([ci]*len(sel))
        if not cand: continue
        cand=np.array(cand); ctx=np.array(ctx); rr=0.05*im_exp
        cnt=np.array([np.sum(np.abs(cand-p)<rr) for p in cand]); db=cand[cnt.argmax()]
        im_star=np.median(cand[np.abs(cand-db)<rr].imag)
        sup=np.array([len(np.unique(ctx[np.abs(cand-p)<rr])) for p in cand]); sb=cand[sup.argmax()]
        re_star=np.median(cand[np.abs(cand-sb)<rr].real)
        a_new[j]=math.pi/im_star; b_new[j]=-(math.pi/im_star)*(re_star+nx)
    a_cur=a_new; b_cur=b_new
    e=per_neuron_err(N*a_cur[:,None],b_cur)
    print(f"  iter {it}: max {float(e.max()):.3e}  median {float(e.median()):.3e}   (queries {QCOUNT[0]:,})")

Wr=N*a_cur[:,None]
rpt("\nAFTER  ",Wr,b_cur)
print(f"TOTAL black-box queries = {QCOUNT[0]:,}  (~{QCOUNT[0]//k:,} per neuron, {k} neurons)")
