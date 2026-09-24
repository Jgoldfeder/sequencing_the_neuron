"""(1) Corrected big-M MILP for isolation margin M_j at layer 2 (tight per-branch M1,M2; small gap).
   (2) The RIGHT oracle test: directly optimize target-direction dominance over the cube --
       min_h  1 - (v1(J(h)).w_j_hat)^2  s.t. |z_j|<=0.1, h in (0,1)^d, many starts.
   If even the best cube point gives angle(v1,w_j) >> 10 deg for many neurons, the isolation
   primitive genuinely fails at layer 2. If some start finds a small angle, isolation is possible."""
import sys, math, numpy as np, torch
from torch.func import jacrev
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import milp, LinearConstraint, Bounds
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_default_dtype(torch.float64)
teacher = MLP([784,128,80,40,32,10], act="sigmoid").to(dev); teacher.load_state_dict(torch.load(
    "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt", map_location=dev, weights_only=False))
Wl=[teacher.layers[i].weight.detach() for i in range(5)]; bl=[teacher.layers[i].bias.detach() for i in range(5)]
W=Wl[1]; b=bl[1]; H,d=W.shape; tau=0.1; Wn=W.cpu().numpy(); bn=b.cpu().numpy()
def milp_margin(j, tl=25.0):
    others=[k for k in range(H) if k!=j]; nb=len(others); n=d+1+nb
    c=np.zeros(n); c[d]=-1.0
    zmin=np.array([np.minimum(Wn[k],0).sum()+bn[k] for k in others]); zmax=np.array([np.maximum(Wn[k],0).sum()+bn[k] for k in others])
    gub=float(np.maximum(np.abs(zmin),np.abs(zmax)).max()); M1=gub-zmin; M2=gub+zmax
    A=[]; ub=[]
    A.append(np.r_[Wn[j],0,np.zeros(nb)]); ub.append(tau-bn[j]); A.append(np.r_[-Wn[j],0,np.zeros(nb)]); ub.append(tau+bn[j])
    for m,k in enumerate(others):
        e=np.zeros(nb); e[m]=M1[m]; A.append(np.r_[-Wn[k],1.0,e]); ub.append(bn[k]+M1[m])
        e2=np.zeros(nb); e2[m]=-M2[m]; A.append(np.r_[Wn[k],1.0,e2]); ub.append(-bn[k])
    A=np.array(A); ub=np.array(ub)
    lb=np.r_[np.zeros(d),0.0,np.zeros(nb)]; hb=np.r_[np.ones(d),gub,np.ones(nb)]; intg=np.r_[np.zeros(d+1),np.ones(nb)]
    r=milp(c,constraints=[LinearConstraint(A,-np.inf,ub)],integrality=intg,bounds=Bounds(lb,hb),options={"time_limit":tl,"mip_rel_gap":1e-3})
    return (float(r.x[d]) if r.x is not None else None), (r.status==0)
print("(1) CORRECTED big-M MILP margin M_j, layer 2:")
for j in [79, 0]:
    g,cert=milp_margin(j); print(f"  neuron {j:2d}: M_j={g:.2f}  certified(gap<=1e-3)={cert}", flush=True)
def Gfrom2(h):
    x=h
    for L in range(1,5): z=x@Wl[L].t()+bl[L]; x=torch.sigmoid(z) if L<4 else z
    return x
wn=W/W.norm(dim=1,keepdim=True)
print("\n(2) ORACLE direction-dominance optimization over the cube (best of 6 starts):")
print(f"{'neuron':>7} {'best angle(v1,w_j)':>18} {'rho there':>10} {'|z_j|':>7}")
for j in [0,15,30,45,60,79]:
    best=(1e9,None,None)
    for s in range(6):
        u=(torch.randn(d,generator=torch.Generator(device=dev).manual_seed(s+j),device=dev)*1.5).requires_grad_(True)
        opt=torch.optim.Adam([u],lr=0.06)
        for _ in range(300):
            h=torch.sigmoid(u); J=jacrev(Gfrom2)(h); U,Sv,Vh=torch.linalg.svd(J,full_matrices=False)
            zj=(W[j]@h+b[j]); loss=1-(Vh[0]@wn[j])**2+15.0*torch.relu(zj.abs()-tau)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            h=torch.sigmoid(u); J=jacrev(Gfrom2)(h); U,Sv,Vh=torch.linalg.svd(J,full_matrices=False)
            zj=float((W[j]@h+b[j]).abs()); ang=math.degrees(math.acos(min(1.0,float((Vh[0]@wn[j]).abs()))))
            if zj<=tau+0.05 and ang<best[0]: best=(ang,float(Sv[1]/Sv[0]),zj)
    print(f"{j:>7} {best[0]:>18.2f} {best[1]:>10.2e} {best[2]:>7.3f}", flush=True)
