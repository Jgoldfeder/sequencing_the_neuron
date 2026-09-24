"""Singularity method, tuned: cluster the AAA pole-cloud (median Re & Im) across
many contexts/projections. Report bias (from Re tau*, well-determined) and
magnitude (from Im tau*). Also report bias using the GOOD magnitude fit + Re tau*."""
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
N=torch.empty_like(Wg)
for j in range(k):
    t=torch.full((k,),20.0,device=dev,dtype=torch.float64); t[j]=0.0
    x0=Wg_pinv@(t-bg); U,Sv,Vh=torch.linalg.svd(J_at(x0),full_matrices=False)
    nv=Vh[0]; N[j]=nv if float(nv@Wg[j])>0 else -nv
W1rec=N*Wg.norm(dim=1,keepdim=True); W1rec_pinv=W1rec.t()@torch.linalg.inv(W1rec@W1rec.t())

# my envelope magnitude fit (from earlier), for the combined bias estimate
@torch.no_grad()
def mag_fit(j):
    t=torch.full((k,),20.0,device=dev,dtype=torch.float64); t[j]=0.0; x0=W1rec_pinv@(t-bg)
    span=6.0/float(Wg[j].norm()); ts=torch.linspace(-span,span,161,device=dev,dtype=torch.float64); h=1e-4
    pts=torch.cat([x0.unsqueeze(0)+(ts+h).unsqueeze(1)*N[j].unsqueeze(0),x0.unsqueeze(0)+(ts-h).unsqueeze(1)*N[j].unsqueeze(0)],0)
    Y=teacher(pts); mag=((Y[:161]-Y[161:])/(2*h)).norm(dim=1); ip=int(mag.argmax())
    keep=mag>0.1*mag[ip]; tw=ts[keep]; logm=torch.log(mag[keep].clamp_min(1e-300))
    A=torch.stack([torch.ones_like(tw),tw,tw*tw],1); P=torch.linalg.inv(A.t()@A)@A.t()
    W=torch.linspace(0.6*float(Wg[j].norm()),1.5*float(Wg[j].norm()),400,device=dev,dtype=torch.float64)
    T=torch.linspace(float(ts[ip])-0.2,float(ts[ip])+0.2,60,device=dev,dtype=torch.float64)
    z=W[:,None,None]*(tw[None,None,:]-T[None,:,None]); sig=torch.sigmoid(z)
    r=logm[None,None,:]-torch.log((sig*(1-sig)).clamp_min(1e-300)); coef=torch.einsum('cn,wtn->wtc',P,r)
    res=((r-torch.einsum('nc,wtc->wtn',A,coef))**2).sum(-1); fi=int(res.argmin())
    return float(W[fi//60])

S=20.0; M=16; L=3.2; NP=140
ta=torch.tensor(np.cos(np.pi*(np.arange(NP)+0.5)/NP)*L,device=dev,dtype=torch.float64)
tanp=ta.cpu().numpy()
mA=[]; bA=[]; bC=[]
for j in range(k):
    a_g=float(Wg[j].norm()); im_exp=math.pi/a_g
    res=[float(W1rec[j]@(W1rec_pinv@(torch.zeros(k,device=dev,dtype=torch.float64)-bg)))]  # dummy
    nx=-float(bg[j])/a_g                                    # N[j].x_m (context-invariant)
    Res=[]; Ims=[]
    for m in range(M):
        sg=(torch.randint(0,2,(k,),generator=g,device=dev)*2-1).double()*S; sg[j]=0.0
        xm=W1rec_pinv@(sg-bg)
        pts=xm.unsqueeze(0)+ta.unsqueeze(1)*N[j].unsqueeze(0)
        with torch.no_grad(): F=teacher(pts)
        for _ in range(3):
            q=torch.randn(dims[-1],generator=g,device=dev,dtype=torch.float64)
            try: poles=AAA(tanp,(F@q).cpu().numpy()).poles()
            except Exception: continue
            sel=poles[(np.abs(poles.imag)>0.5*im_exp)&(np.abs(poles.imag)<1.8*im_exp)&(np.abs(poles.real)<0.8)]
            for p in sel: Res.append(p.real); Ims.append(abs(p.imag))
    if not Ims: mA.append(9.9); bA.append(9.9); bC.append(9.9); continue
    Res=np.array(Res); Ims=np.array(Ims)
    # trimmed cluster: keep poles whose Im is closest to the median, then median again
    im0=np.median(Ims); keep=np.abs(Ims-im0)<0.25*im_exp
    re_star=np.median(Res[keep]) if keep.any() else np.median(Res)
    im_star=np.median(Ims[keep]) if keep.any() else im0
    a_sing=math.pi/im_star
    mA.append(abs(a_sing-float(W1t[j].norm())))
    bA.append(abs(-a_sing*(re_star+nx)-float(b1t[j])))     # bias from singularity a
    a_fit=mag_fit(j)                                        # good magnitude
    bC.append(abs(-a_fit*(re_star+nx)-float(b1t[j])))      # bias from fit-magnitude + Re tau*
mA=torch.tensor(mA); bA=torch.tensor(bA); bC=torch.tensor(bC)
print(f"SINGULARITY (AAA, clustered, {M} contexts):")
print(f"  magnitude (from Im): worst {mA.max():.3e} median {mA.median():.3e}")
print(f"  bias (from Im+Re):   worst {bA.max():.3e} median {bA.median():.3e}")
print(f"  bias (fit-mag + Re): worst {bC.max():.3e} median {bC.median():.3e}   <- combined")
