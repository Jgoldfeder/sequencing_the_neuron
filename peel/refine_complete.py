"""COMPLETE refinement with BASE-AVERAGING for the bias/magnitude.
Direction: saturate others -> Jacobian SVD (once). Magnitude+center: scan along
the recovered direction from several DIFFERENT saturation patterns (different
downstream envelopes) and take the median -- the envelope-induced center shift
varies across patterns and averages out. Report full (w,b) max_eps."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True)
g=torch.Generator(device=dev).manual_seed(1)

@torch.no_grad()
def J_at(x,fd=1e-4):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()

@torch.no_grad()
def scan_center(x0, n, wguess, N=161, h=1e-4):
    """||df/dt|| bump along n; fit (||w||, center) with a quadratic log-envelope."""
    span=6.0/wguess; ts=torch.linspace(-span,span,N,device=dev,dtype=torch.float64)
    pts=torch.cat([x0.unsqueeze(0)+(ts+h).unsqueeze(1)*n.unsqueeze(0),
                   x0.unsqueeze(0)+(ts-h).unsqueeze(1)*n.unsqueeze(0)],0)
    Y=teacher(pts); mag=((Y[:N]-Y[N:])/(2*h)).norm(dim=1)          # (N,)
    ip=int(mag.argmax()); t0=float(ts[ip]); gs=float(ts[1]-ts[0])
    keep=mag>0.10*mag[ip]; tw=ts[keep]; logm=torch.log(mag[keep].clamp_min(1e-300))
    A=torch.stack([torch.ones_like(tw),tw,tw*tw],1); P=torch.linalg.inv(A.t()@A)@A.t()
    wlo,whi,tlo,thi=0.5*wguess,1.6*wguess,t0-3*gs,t0+3*gs
    wnorm,tc=wguess,t0
    for _ in range(3):
        W=torch.linspace(wlo,whi,160,device=dev,dtype=torch.float64)
        T=torch.linspace(tlo,thi,160,device=dev,dtype=torch.float64)
        z=W[:,None,None]*(tw[None,None,:]-T[None,:,None]); sig=torch.sigmoid(z)
        r=logm[None,None,:]-torch.log((sig*(1-sig)).clamp_min(1e-300))
        coef=torch.einsum('cn,wtn->wtc',P,r)
        res=((r-torch.einsum('nc,wtc->wtn',A,coef))**2).sum(-1)
        fi=int(res.argmin()); wnorm=float(W[fi//160]); tc=float(T[fi%160])
        wlo,whi,tlo,thi=wnorm-(whi-wlo)/40,wnorm+(whi-wlo)/40,tc-(thi-tlo)/40,tc+(thi-tlo)/40
    return wnorm, tc

# --- full-weight guess ---
Wg=torch.empty(k,d,device=dev,dtype=torch.float64); bg=torch.empty(k,device=dev,dtype=torch.float64)
for j in range(k):
    u=tn[j]; v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
    Wg[j]=(math.cos(math.radians(5))*u+math.sin(math.radians(5))*v)*W1t[j].norm()*(1+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64)-1))
    bg[j]=b1t[j]+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64).item()-1)
Wg_pinv=Wg.t()@torch.linalg.inv(Wg@Wg.t())

def full_err(W,b):
    e=0.0
    for j in range(k):
        s=1.0 if float((W[j]/W[j].norm())@tn[j])>0 else -1.0
        e=max(e, float((s*W[j]-W1t[j]).abs().max()), abs(s*float(b[j])-float(b1t[j])))
    return e
print(f"BEFORE (full-weight guess): max_eps = {full_err(Wg,bg):.3e}")

S=20.0; M=12
Wr=torch.empty_like(Wg); br=torch.empty_like(bg)
for j in range(k):
    t=torch.full((k,),S,device=dev,dtype=torch.float64); t[j]=0.0
    x0=Wg_pinv@(t-bg)
    U,Sv,Vh=torch.linalg.svd(J_at(x0),full_matrices=False)
    n=Vh[0]; n = n if float(n@Wg[j])>0 else -n
    wns=[]; offs=[]
    for m in range(M):
        sg=(torch.randint(0,2,(k,),generator=g,device=dev)*2-1).double()*S; sg[j]=0.0
        x0m=Wg_pinv@(sg-bg)
        wn,tc=scan_center(x0m,n,float(Wg[j].norm()))
        wns.append(wn); offs.append(float(n@x0m)+tc)
    wnorm=torch.tensor(wns).median().item(); off=torch.tensor(offs).median().item()
    Wr[j]=wnorm*n; br[j]=-wnorm*off

print(f"AFTER  (refined, {M}-base median): max_eps = {full_err(Wr,br):.3e}")
de=max(math.degrees(math.acos(min(1.0,abs(float((Wr[j]/Wr[j].norm())@tn[j]))))) for j in range(k))
me=max(abs(float(Wr[j].norm()-W1t[j].norm())) for j in range(k))
be=max(abs(float(br[j])-float(b1t[j])) for j in range(k))
print(f"  worst direction {de:.3e}deg | worst magnitude {me:.3e} | worst bias {be:.3e}")
