"""Joint common-denominator singularity fit, done right: CHEBYSHEV basis (not
Vandermonde), variable-projection over the shared denominator Q across ALL
contexts x ALL outputs, stability across orders/windows. Stays within layer-1
guess only (no downstream fitting). Measures pole-localization error directly."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch, numpy as np
from numpy.polynomial.chebyshev import chebvander, chebroots
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]; O=dims[-1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True); g=torch.Generator(device=dev).manual_seed(1)
@torch.no_grad()
def J_at(x,fd=5e-5):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()
Wg=torch.empty(k,d,device=dev,dtype=torch.float64); bg=torch.empty(k,device=dev,dtype=torch.float64)
for j in range(k):
    u=tn[j]; v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
    Wg[j]=(math.cos(math.radians(5))*u+math.sin(math.radians(5))*v)*W1t[j].norm()*(1+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64)-1))
    bg[j]=b1t[j]+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64).item()-1)
Wgpinv=Wg.t()@torch.linalg.inv(Wg@Wg.t())
N=torch.empty_like(Wg)
for j in range(k):
    t=torch.full((k,),20.0,device=dev,dtype=torch.float64); t[j]=0.0
    x0=Wgpinv@(t-bg); U,Sv,Vh=torch.linalg.svd(J_at(x0),full_matrices=False)
    nv=Vh[0]; N[j]=nv if float(nv@Wg[j])>0 else -nv
W1rec=N*Wg.norm(dim=1,keepdim=True); W1rp=W1rec.t()@torch.linalg.inv(W1rec@W1rec.t())

def joint_poles(tnp, G, R, P, L):
    """shared-denominator Chebyshev fit: g_i*Q ~= P_i for all i, one Q. -> roots(Q)."""
    s=tnp/L; V=chebvander(s,P); Qv,_=np.linalg.qr(V)
    Wc=chebvander(s,R)[:,1:]                                   # T_1..T_R
    def prj(M): return M-Qv@(Qv.T@M)
    A=[]; c=[]
    for gi in G:
        A.append(prj(gi[:,None]*Wc)); c.append(prj(gi))
    A=np.vstack(A); c=np.concatenate(c)
    q,_,_,_=np.linalg.lstsq(A,-c,rcond=None)
    return chebroots(np.concatenate([[1.0],q]))*L             # Chebyshev roots -> t

S=20.0; M=28; NP=120; delta=1e-3
merr=[]; berr=[]; pole_err=[]
for j in range(k):
    a_g=float(Wg[j].norm()); im_exp=math.pi/a_g; nx=-float(bg[j])/a_g
    a_true=float(W1t[j].norm())
    # visibility-selected contexts
    SG=(torch.randint(0,2,(400,k),generator=g,device=dev)*2-1).double()*S; SG[:,j]=0.0
    Xm=(SG-bg)@W1rp.t()
    with torch.no_grad(): vis=((teacher(Xm+delta*N[j])-teacher(Xm-delta*N[j]))/(2*delta)).norm(dim=1)
    top=torch.topk(vis,M).indices
    cand=[]
    for L in (2.6,3.0,3.4):
        tnp=np.cos(np.pi*(np.arange(NP)+0.5)/NP)*L; ta=torch.tensor(tnp,device=dev,dtype=torch.float64)
        G=[]
        for mi in top.tolist():
            F=teacher(Xm[mi].unsqueeze(0)+ta.unsqueeze(1)*N[j].unsqueeze(0)).detach().cpu().numpy()
            for r in range(O): G.append(F[:,r])
        for R in (14,20):
          for Pn in (5,7,9,12):
            try: roots=joint_poles(tnp,G,R,Pn,L)
            except Exception: continue
            sel=roots[(roots.imag>0.6*im_exp)&(roots.imag<1.6*im_exp)&(np.abs(roots.real)<0.5)]
            cand.extend([p if p.imag>0 else np.conj(p) for p in sel])
    if not cand: merr.append(9.9); berr.append(9.9); continue
    cand=np.array(cand)
    cnt=np.array([np.sum(np.abs(cand-p)<0.03*im_exp) for p in cand]); c=cand[cnt.argmax()]
    cl=cand[np.abs(cand-c)<0.03*im_exp]; cc=complex(np.median(cl.real),np.median(cl.imag))
    a_j=math.pi/cc.imag; b_j=-a_j*(cc.real+nx)
    # true pole (last context used): tau*=-(n.x_m + b/a) + i pi/a
    xm=Xm[top[0]]; cj=float(W1t[j]@xm)+float(b1t[j]); tp=complex(-cj/a_true, math.pi/a_true)
    pole_err.append(abs(cc-tp)); merr.append(abs(a_j-a_true)); berr.append(abs(b_j-float(b1t[j])))
merr=torch.tensor(merr); berr=torch.tensor(berr); pe=torch.tensor(pole_err)
print("JOINT common-denominator (Chebyshev) singularity fit:")
print(f"  pole localization err: max {pe.max():.2e} mean {pe.mean():.2e} median {pe.median():.2e}  (need ~1e-4)")
print(f"  magnitude err: max {merr.max():.3e} mean {merr.mean():.3e} median {merr.median():.3e}")
print(f"  bias err:      max {berr.max():.3e} mean {berr.mean():.3e} median {berr.median():.3e}")
