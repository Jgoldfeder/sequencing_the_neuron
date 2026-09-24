"""PROJECTED LOCAL-IDENTIFIABILITY TEST for the reduced L2 problem W2 = A B (reviewer's step).
DIAGNOSTIC ONLY (uses truth explicitly) -- NOT a sealed solver. Question: do the observable jets
(values + first derivatives in h) distinguish theta_A = vec(A) from everything the unknown downstream
nuisance theta_N = (b2,W3,b3,W4,b4,W5,b5) can imitate?

Build observation Jacobians J_A, J_N at the TRUE params; residualize J_A against col(J_N):
   J_A_perp = (I - P_{col J_N}) J_A ;  inspect rank, sigma_min, kappa.
rank(J_A_perp) ~ 6400 with decent conditioning  => A locally identifiable despite unknown downstream.
rank collapses                                   => moving into the row space did NOT remove the ambiguity.

Need #obs - rank(J_N) >= 6400 or A looks deficient from too few observations. 810 obs/probe, 24 probes.
"""
import sys, torch
from torch.func import jacrev
torch.set_default_dtype(torch.float64)
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
dev="cuda"; CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
pop=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False)
dims=pop["dims"]
t=MLP(dims,act="sigmoid").to(dev).double(); t.load_state_dict(pop["teacher_state"]); t.eval()
W2t=t.layers[1].weight.detach(); b2t=t.layers[1].bias.detach()
W3t=t.layers[2].weight.detach(); b3t=t.layers[2].bias.detach()
W4t=t.layers[3].weight.detach(); b4t=t.layers[3].bias.detach()
W5t=t.layers[4].weight.detach(); b5t=t.layers[4].bias.detach()
# row-space basis B (orthonormal, spans rowspan(W2t)); A_true = W2t B^T so W2t = A_true B
Uw,Sw,Vh=torch.linalg.svd(W2t,full_matrices=False); B=Vh[:80]            # 80x128 orthonormal
A_true=W2t@B.t()                                                          # 80x80
print(f"[check] ||W2t - A_true B|| = {float((W2t-A_true@B).norm()):.2e}  (reconstruction)")
# probes h: excite each neuron (t_j=0), true placement (diagnostic)
W2tp=torch.linalg.pinv(W2t); gg=torch.Generator(device=dev).manual_seed(0); H=[]
for j in range(24):
    tt=torch.randn(80,generator=gg,device=dev)*2.0; tt[j%80]=0.0
    H.append(torch.clamp(W2tp@(tt-b2t),1e-3,1-1e-3))
H=torch.stack(H)                                                          # 24x128
P=H.shape[0]
# pack/unpack theta
shapes=[(80,80),(80,),(40,80),(40,),(32,40),(32,),(10,32),(10,)]
numel=[s if isinstance(s,int) else torch.tensor(s).prod().item() for s in [80*80,80,40*80,40,32*40,32,10*32,10]]
def unpack(th):
    o=0;out=[]
    for sh,n in zip(shapes,numel): out.append(th[o:o+n].reshape(sh)); o+=n
    return out
theta0=torch.cat([A_true.reshape(-1),b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t])
def M(th):
    A,b2,W3,b3,W4,b4,W5,b5=unpack(th)
    W2=A@B
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2)
    z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4)
    out=s4@W5.t()+b5                                                      # (P,10) values
    U=(W2@B.t()).t()                                                      # (80dir,80)  U[j,a]=W2[a].B[j]
    ds2=sp2[:,None,:]*U[None,:,:]                                         # (P,80,80)
    dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3
    dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4
    dout=ds4@W5.t()                                                       # (P,80,10) jets
    return torch.cat([out.reshape(-1),dout.reshape(-1)])
nobs=P*10+P*80*10
print(f"[setup] probes={P}, obs/probe={10+800}, total obs={nobs}, params={theta0.numel()} (A=6400, N={theta0.numel()-6400})")
Jac=jacrev(M,chunk_size=512)(theta0)                                     # (nobs, 11362), chunked to fit
JA=Jac[:,:6400]; JN=Jac[:,6400:]
# rank of nuisance block
sN=torch.linalg.svdvals(JN); rN=int((sN>1e-9*sN[0]).sum())
QN,_=torch.linalg.qr(JN)                                                  # orthonormal basis of col(JN)
JAperp=JA-QN@(QN.t()@JA)
sA=torch.linalg.svdvals(JA)                                              # A sensitivity before nuisance removal
sAp=torch.linalg.svdvals(JAperp)                                        # after removing what downstream imitates
def rk(s,rt=1e-8): return int((s>rt*s[0]).sum())
print(f"[nuisance] rank(J_N)={rN}/{JN.shape[1]}  (obs left after removal ~ {nobs-rN})")
print(f"[A raw ]   rank(J_A)      ={rk(sA):>5}/6400   sigma_max {float(sA[0]):.2e}  sigma_min {float(sA[rk(sA)-1]):.2e}")
print(f"[A perp]   rank(J_A_perp) ={rk(sAp):>5}/6400   sigma_max {float(sAp[0]):.2e}  sigma_min@rank {float(sAp[rk(sAp)-1]):.2e}")
print(f"[A perp]   sigma spectrum: [0]={float(sAp[0]):.2e} [3200]={float(sAp[3200]):.2e} [6000]={float(sAp[6000]):.2e} [6399]={float(sAp[6399]):.2e}")
print(f"[A perp]   full-rank kappa (s0/s6399) = {float(sAp[0]/sAp[6399]):.2e}")
print(f"[verdict]  A locally identifiable vs unknown downstream?  rank(J_A_perp)={rk(sAp)}/6400")
