"""Singularity method, done properly: recover all directions (SVD), build the
probe lines with the RECOVERED directions (so Re(tau*) is context-invariant too),
sample g(tau) on Chebyshev nodes, use scipy AAA to find poles. Diagnostic first:
does AAA find the TRUE pole? Then extract a_j, b_j for all neurons."""
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
def J_at(x,fd=1e-4):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()
Wg=torch.empty(k,d,device=dev,dtype=torch.float64); bg=torch.empty(k,device=dev,dtype=torch.float64)
for j in range(k):
    u=tn[j]; v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
    Wg[j]=(math.cos(math.radians(5))*u+math.sin(math.radians(5))*v)*W1t[j].norm()*(1+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64)-1))
    bg[j]=b1t[j]+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64).item()-1)
Wg_pinv=Wg.t()@torch.linalg.inv(Wg@Wg.t())

# --- recover all directions (SVD) ---
N=torch.empty_like(Wg)
for j in range(k):
    t=torch.full((k,),20.0,device=dev,dtype=torch.float64); t[j]=0.0
    x0=Wg_pinv@(t-bg); U,Sv,Vh=torch.linalg.svd(J_at(x0),full_matrices=False)
    nv=Vh[0]; N[j]=nv if float(nv@Wg[j])>0 else -nv
W1rec=N*Wg.norm(dim=1,keepdim=True)                    # recovered dirs, guess magnitudes
W1rec_pinv=W1rec.t()@torch.linalg.inv(W1rec@W1rec.t())

S=20.0; M=12; L=3.2; NP=120
def cheb(): return torch.tensor(np.cos(np.pi*(np.arange(NP)+0.5)/NP)*L,device=dev,dtype=torch.float64)

# ---- DIAGNOSTIC on neuron 0: does AAA find the true pole? ----
j=0; a_true=float(W1t[j].norm())
sg=(torch.randint(0,2,(k,),generator=g,device=dev)*2-1).double()*S; sg[j]=0.0
xm=W1rec_pinv@(sg-bg)
cj_true=float(W1t[j]@xm)+float(b1t[j])                 # true c_j = w.x_m + b
tau_true=complex(-cj_true/a_true, math.pi/a_true)
ta=cheb(); pts=xm.unsqueeze(0)+ta.unsqueeze(1)*N[j].unsqueeze(0)
with torch.no_grad(): F=teacher(pts)
q=torch.randn(dims[-1],generator=g,device=dev,dtype=torch.float64); gv=(F@q).cpu().numpy()
r=AAA(ta.cpu().numpy(), gv)
poles=r.poles(); poles=poles[np.argsort(np.abs(poles-tau_true))]
print(f"[diagnostic n0] true pole tau* = {tau_true.real:+.4f}{tau_true.imag:+.4f}i")
print(f"   AAA nearest poles: "+", ".join(f"{p.real:+.3f}{p.imag:+.3f}i" for p in poles[:3]))
print(f"   -> nearest-pole error {abs(poles[0]-tau_true):.2e}\n")

# ---- full extraction ----
merr=[]; berr=[]
for j in range(k):
    a_g=float(Wg[j].norm()); im_exp=math.pi/a_g
    ta=cheb()
    A_est=[]; B_est=[]
    for m in range(M):
        sg=(torch.randint(0,2,(k,),generator=g,device=dev)*2-1).double()*S; sg[j]=0.0
        xm=W1rec_pinv@(sg-bg); nx=float(N[j]@xm)
        pts=xm.unsqueeze(0)+ta.unsqueeze(1)*N[j].unsqueeze(0)
        with torch.no_grad(): F=teacher(pts)
        for _ in range(3):
            q=torch.randn(dims[-1],generator=g,device=dev,dtype=torch.float64)
            try: poles=AAA(ta.cpu().numpy(),(F@q).cpu().numpy()).poles()
            except Exception: continue
            poles=poles[(np.abs(poles.imag)>0.4*im_exp)&(np.abs(poles.imag)<2.2*im_exp)&(np.abs(poles.real)<1.5)]
            if len(poles)==0: continue
            p=poles[np.argmin(np.abs(np.abs(poles.imag)-im_exp))]
            a=math.pi/abs(p.imag); A_est.append(a); B_est.append(-a*(p.real+nx))
    if not A_est: merr.append(9.9); berr.append(9.9); continue
    a_j=float(np.median(A_est)); b_j=float(np.median(B_est))
    merr.append(abs(a_j-float(W1t[j].norm()))); berr.append(abs(b_j-float(b1t[j])))
merr=torch.tensor(merr); berr=torch.tensor(berr)
print(f"SINGULARITY (AAA, recovered-dir contexts):")
print(f"  magnitude err: worst {merr.max():.3e}  median {merr.median():.3e}")
print(f"  bias err:      worst {berr.max():.3e}  median {berr.median():.3e}")
