"""Recover magnitude AND bias from the sigmoid's complex singularity (chat's idea).
Along x_m + tau*n_j the black-box 1-D curve g(tau) has a singularity at
tau* = -db_j/a_j + i*pi/a_j (from the inner sigma), CONTEXT-INVARIANT; downstream
can't move it. Fit rational approximants to real samples, find the pole near
+-i*pi/||w_hat||, extract a_j=pi/|Im tau*|, b_j = b_hat_j - a_j*Re(tau*)."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch, numpy as np
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True); g=torch.Generator(device=dev).manual_seed(1)
@torch.no_grad()
def J_at(x,fd=1e-4):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()

Wg=torch.empty(k,d,device=dev,dtype=torch.float64); bg=torch.empty(k,device=dev,dtype=torch.float64)
for j in range(k):
    u=tn[j]; v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
    Wg[j]=(math.cos(math.radians(5))*u+math.sin(math.radians(5))*v)*W1t[j].norm()*(1+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64)-1))
    bg[j]=b1t[j]+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64).item()-1)
Wg_pinv=Wg.t()@torch.linalg.inv(Wg@Wg.t())

def rational_poles(tau, gv, n, L):
    """linearized Pade: fit g*Q = P, deg n each, in scaled s=tau/L; return poles (tau)."""
    s=tau/L; V=np.vander(s, n+1, increasing=True)          # (N, n+1)
    A=np.concatenate([gv[:,None]*V, -V], axis=1)            # (N, 2n+2)
    _,_,Vh=np.linalg.svd(A); coef=Vh[-1]
    bq=coef[:n+1]                                           # Q(s)=sum bq_l s^l
    r=np.roots(bq[::-1])                                    # numpy wants highest-first
    return r*L

S=20.0; M=14; L=3.4; N=90; n=9
merr=[]; berr=[]
for j in range(k):
    # direction (SVD from one context)
    t=torch.full((k,),S,device=dev,dtype=torch.float64); t[j]=0.0
    x0=Wg_pinv@(t-bg); U,Sv,Vh=torch.linalg.svd(J_at(x0),full_matrices=False)
    nvec=Vh[0]; nvec=nvec if float(nvec@Wg[j])>0 else -nvec
    im_exp=math.pi/float(Wg[j].norm())                     # expected |Im tau*|
    # Chebyshev nodes on [-L,L]
    cheb=torch.tensor(np.cos(np.pi*(np.arange(N)+0.5)/N)*L, device=dev, dtype=torch.float64)
    cands=[]
    for m in range(M):
        sg=(torch.randint(0,2,(k,),generator=g,device=dev)*2-1).double()*S; sg[j]=0.0
        xm=Wg_pinv@(sg-bg)
        pts=xm.unsqueeze(0)+cheb.unsqueeze(1)*nvec.unsqueeze(0)
        with torch.no_grad(): F=teacher(pts)               # (N,O)
        for _ in range(3):                                  # a few output projections
            q=torch.randn(dims[-1],generator=g,device=dev,dtype=torch.float64)
            gv=(F@q).cpu().numpy(); ta=cheb.cpu().numpy()
            for nn in (7,9,11):
                try: poles=rational_poles(ta,gv,nn,L)
                except Exception: continue
                for p in poles:
                    if abs(p.imag)>0.3*im_exp and abs(p.imag)<2.5*im_exp and abs(p.real)<1.0:
                        cands.append(p if p.imag>0 else np.conj(p))
    if not cands:
        merr.append(9.9); berr.append(9.9); continue
    cands=np.array(cands)
    # cluster near expected: median of poles closest to (0, im_exp)
    dist=np.abs(cands.real)+np.abs(np.abs(cands.imag)-im_exp)
    sel=cands[np.argsort(dist)[:max(3,len(cands)//4)]]
    tau_star=complex(np.median(sel.real), np.median(np.abs(sel.imag)))
    a_j=math.pi/abs(tau_star.imag); b_j=float(bg[j])-a_j*tau_star.real
    merr.append(abs(a_j-float(W1t[j].norm()))); berr.append(abs(b_j-float(b1t[j])))

merr=torch.tensor(merr); berr=torch.tensor(berr)
print(f"SINGULARITY method ({M} contexts, Pade):")
print(f"  magnitude err: worst {merr.max():.3e}  median {merr.median():.3e}   (guess ~8e-2; my fit 6.4e-3)")
print(f"  bias err:      worst {berr.max():.3e}  median {berr.median():.3e}   (guess ~8e-2; stuck at ~6e-2 before)")
