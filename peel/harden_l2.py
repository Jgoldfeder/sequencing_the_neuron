"""Hardening pass: does ANY reachable cube probe isolate a layer-2 neuron's first-layer direction?
For ALL 80 neurons, minimize L_j(h) = 1 - ||J n_j||^2/||J||^2 (fraction of the black-box Jacobian
OUTSIDE the target row w_j) over h in (0,1)^128, |z_j|<=tau, with many starts + mixed init
distributions (interior + near-corner). Record best leading-vector angle and rho per neuron.
True W2,b2 (perfect guess). If best angle stays >>10 deg for essentially all 80, isolation fails."""
import sys, math, numpy as np, torch
from torch.func import jacrev
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_default_dtype(torch.float64)
teacher = MLP([784,128,80,40,32,10], act="sigmoid").to(dev); teacher.load_state_dict(torch.load(
    "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt", map_location=dev, weights_only=False))
Wl=[teacher.layers[i].weight.detach() for i in range(5)]; bl=[teacher.layers[i].bias.detach() for i in range(5)]
W=Wl[1]; b=bl[1]; H,d=W.shape; wn=W/W.norm(dim=1,keepdim=True); tau=0.1
def Gfrom2(h):
    x=h
    for L in range(1,5): z=x@Wl[L].t()+bl[L]; x=torch.sigmoid(z) if L<4 else z
    return x
NSTART=14; STEPS=200
best_ang=np.full(H,180.0); best_rho=np.full(H,1.0)
for j in range(H):
    nj=wn[j]
    for s in range(NSTART):
        sc = 0.5 if s%3==0 else (2.0 if s%3==1 else 4.0)                 # interior / mid / near-corner
        u=(torch.randn(d,generator=torch.Generator(device=dev).manual_seed(1000*j+s),device=dev)*sc).requires_grad_(True)
        opt=torch.optim.Adam([u],lr=0.06)
        for _ in range(STEPS):
            h=torch.sigmoid(u); J=jacrev(Gfrom2)(h); Jn=J@nj
            L=1-(Jn.pow(2).sum())/(J.pow(2).sum().clamp_min(1e-30))+15.0*torch.relu((W[j]@h+b[j]).abs()-tau)
            opt.zero_grad(); L.backward(); opt.step()
        with torch.no_grad():
            h=torch.sigmoid(u); J=jacrev(Gfrom2)(h); zj=float((W[j]@h+b[j]).abs())
            if zj<=tau+0.05:
                U,Sv,Vh=torch.linalg.svd(J,full_matrices=False)
                ang=math.degrees(math.acos(min(1.0,float((Vh[0]@nj).abs()))))
                if ang<best_ang[j]: best_ang[j]=ang; best_rho[j]=float(Sv[1]/Sv[0])
    if (j+1)%16==0:
        bnp=best_ang[:j+1]; print(f"  ...{j+1}/80 done: running min {bnp.min():.1f} deg, median {np.median(bnp):.1f} deg, #<10deg={int((bnp<10).sum())}", flush=True)
print(f"\nHARDENING (all 80 layer-2 neurons, {NSTART} starts x mixed init, whole-Jacobian objective):")
print(f"  best angle(v1,w_j): MIN {best_ang.min():.1f}  25pct {np.percentile(best_ang,25):.1f}  median {np.median(best_ang):.1f}  max {best_ang.max():.1f} deg")
print(f"  neurons with best angle < 10 deg: {int((best_ang<10).sum())}/80 ;  < 15 deg: {int((best_ang<15).sum())}/80")
print(f"  rho at best point: median {np.median(best_rho):.2e}")
