"""Bias via downstream-linearization + symmetry test (chat's real-domain attack).
Dual-isolation gives g(t)=H(sigma(a(t-tau))). At the true center, for affine H,
g(tau+u)+g(tau-u)=const (since sigma(z)+sigma(-z)=1). Search other-coordinate
contexts for near-affine H (low symmetry residual); there tau -> bias is exact.
Diagnostic: does low residual R_m => low bias error, and can we reach 1e-4?"""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch, numpy as np
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
V=N.t()@torch.linalg.inv(N@N.t())
W1rec=N*Wg.norm(dim=1,keepdim=True); W1rp=W1rec.t()@torch.linalg.inv(W1rec@W1rec.t())

ncand=200
best_err=[]; corr_pts=[]
for j in range(k):
    a=float(W1t[j].norm())                                # recovered to ~5e-6; use true for bias test
    vj=V[:,j]; nx=-float(bg[j])/float(Wg[j].norm())
    us=(torch.arange(1,6,device=dev,dtype=torch.float64))/a          # offsets z=1..5
    taus=torch.linspace(-0.14/a,0.14/a,29,device=dev,dtype=torch.float64)
    TT=(2*torch.rand(ncand,k,generator=g,device=dev,dtype=torch.float64)-1)*2.5; TT[:,j]=0.0
    X0=(TT-bg)@W1rp.t()
    res=[]
    for m in range(ncand):
        x0=X0[m]
        # amplitude (reject silent)
        with torch.no_grad(): amp=float((teacher(x0+us[-1]*vj)-teacher(x0-us[-1]*vj)).norm())
        if amp<3e-3: continue
        # E(tau) = sum_O var over u of [g(tau+u)+g(tau-u)]
        P=torch.cat([x0.unsqueeze(0)+(taus[:,None,None]+us[None,:,None])*vj[None,None,:],
                     x0.unsqueeze(0)+(taus[:,None,None]-us[None,:,None])*vj[None,None,:]],0).reshape(-1,d)
        with torch.no_grad(): F=teacher(P).reshape(2,len(taus),len(us),O)
        s=F[0]+F[1]                                        # (ntau,nu,O) symmetric sums
        E=((s-s.mean(1,keepdim=True))**2).sum((1,2))       # (ntau,)  variance over u
        i=int(E.argmin())
        if 0<i<len(E)-1:                                   # sub-grid parabolic refine
            y0,y1,y2=float(E[i-1]),float(E[i]),float(E[i+1]); den=y0-2*y1+y2
            dd=max(-1.0,min(1.0,0.5*(y0-y2)/den)) if abs(den)>1e-300 else 0.0
            tau_hat=float(taus[i])+dd*float(taus[1]-taus[0])
        else: tau_hat=float(taus[i])
        Emin=float(E[i])/(amp**2)
        b_hat=-a*(tau_hat+nx)
        res.append((Emin, abs(b_hat-float(b1t[j]))))
    if not res: best_err.append(9.9); continue
    res.sort()
    # best-residual context's bias error; also weighted median of best 8
    best_err.append(res[0][1])
    for e,be in res[:15]: corr_pts.append((e,be))
be=torch.tensor(best_err)
print("Bias via symmetry-selected near-affine contexts (200 candidates/neuron):")
print(f"  best-context bias err: max {be.max():.2e} mean {be.mean():.2e} median {be.median():.2e}  (old bias ~6e-3)")
cp=np.array(corr_pts)
lo=cp[cp[:,0]<np.quantile(cp[:,0],0.1)]; hi=cp[cp[:,0]>np.quantile(cp[:,0],0.9)]
print(f"  DIAGNOSTIC: bias err for lowest-10%% residual: median {np.median(lo[:,1]):.2e}")
print(f"              bias err for highest-10%% residual: median {np.median(hi[:,1]):.2e}")
print(f"  (if low-residual << high-residual, the mechanism works)")
