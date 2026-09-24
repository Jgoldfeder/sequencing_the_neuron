"""Does the single-neuron bump-finder actually land on neuron k's TRUE hyperplane?
Report distance-to-true-plane of located points vs the guess-plane distance."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,16,8]; d=dims[0]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()

@torch.no_grad()
def slopes(x0,wref,ts,h=1e-3):
    P=torch.cat([x0.unsqueeze(0)+(ts.unsqueeze(1)+h)*wref.unsqueeze(0),
                 x0.unsqueeze(0)+(ts.unsqueeze(1)-h)*wref.unsqueeze(0)],0)
    Y=teacher(P); n=len(ts); return (Y[:n]-Y[n:])/(2*h)

@torch.no_grad()
def bump_point(x0,wref,wn,span_mult=6.0):
    span=span_mult/wn; ts=torch.linspace(-span,span,61,device=dev,dtype=torch.float64)
    s=slopes(x0,wref,ts); A=torch.stack([torch.ones_like(ts),ts],1)
    coef=torch.linalg.lstsq(A,s).solution; resid=s-A@coef; mag=resid.norm(dim=1)
    i=int(mag.argmax()); return x0+float(ts[i])*wref, float(ts[i]), span

k=0; g=torch.Generator(device=dev).manual_seed(3)
wk=W1t[k]; u=wk/wk.norm(); wn=float(wk.norm())
th=math.radians(5.0)
v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
w_guess=math.cos(th)*u+math.sin(th)*v
for span_mult, thr in [(6.0,0.15),(2.0,0.15),(1.0,0.05),(0.5,0.05)]:
    dist_true=[]; dist_guess=[]; nother=[]
    for m in range(300):
        x0=torch.randn(d,generator=g,device=dev,dtype=torch.float64)
        x0=x0-(w_guess@x0 + float(b1t[k])/wn)*w_guess          # onto guess plane
        zn=((W1t@x0+b1t)/W1t.norm(dim=1)).abs(); zn[k]=9
        if float(zn.min())<thr: continue
        xstar,tstar,span=bump_point(x0,w_guess,wn,span_mult)
        dist_true.append(abs(float(W1t[k]@xstar+b1t[k]))/wn)   # dist to TRUE plane
        dist_guess.append(abs(float(w_guess@xstar+float(b1t[k])/wn)))  # dist to guess plane
        # how many OTHER planes fall within the scan window (overlapping bumps)?
        within=((W1t@x0+b1t).abs()/W1t.norm(dim=1) < span); within[k]=False
        nother.append(int(within.sum()))
    if not dist_true: print(f"span {span_mult}: no points"); continue
    dt=torch.tensor(dist_true); dg=torch.tensor(dist_guess)
    print(f"span=+-{span_mult}/|w| ({len(dt)} pts, thr {thr}): "
          f"dist-to-TRUE med {dt.median():.4f}  dist-to-GUESS med {dg.median():.4f}  "
          f"| other planes in window: med {sorted(nother)[len(nother)//2]}")
print(f"\n(guess plane is ~sin(5deg)= {math.sin(math.radians(5)):.3f} from true plane on average;")
print(f" if bump lands closer to TRUE than that, the single-neuron signal works)")
