"""EXACT coherent compensation check (reviewer). W2(A)=R_m+A B with R_m=W2m-W2m P_B, A_m=W2m B^T so
W2(A_m)=W2m EXACTLY. Measure r_W=Phi(A_m,eta*)-Y*, r_eta=Phi(A_m,eta_m)-Phi(A_m,eta*),
r_tot=Phi(A_m,eta_m)-Y*. Print S=||r_W||, N=||r_eta||, T=||r_tot||, and cos(r_eta,r_W) DIRECTLY.
Compensation <=> N/S~=1, T/S<<1, cos~=-1. Otherwise 'equal norms' was not cancellation."""
import sys, torch
torch.set_default_dtype(torch.float32); dev="cuda"
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"; PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
pop=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False)
t=MLP(pop["dims"],act="sigmoid").to(dev).float(); t.load_state_dict(pop["teacher_state"]); t.eval()
W2t=t.layers[1].weight.detach(); b2t=t.layers[1].bias.detach(); nt=W2t.norm(dim=1)
W3t=t.layers[2].weight.detach(); b3t=t.layers[2].bias.detach(); W4t=t.layers[3].weight.detach(); b4t=t.layers[3].bias.detach()
W5t=t.layers[4].weight.detach(); b5t=t.layers[4].bias.detach()
Uw,Sw,Vh=torch.linalg.svd(W2t,full_matrices=False); B=Vh[:80]
eta_true=torch.cat([b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t])
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False); mem=[{k:v.to(dev).float() for k,v in sd.items()} for sd in pk["pop_states"]]
def match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r,device=dev);c=torch.tensor(c,device=dev);return r,c,torch.where(Cm[r.cpu(),c.cpu()]<Cp[r.cpu(),c.cpu()],-1.,1.).to(dev)
def align(sd):
    W2m=sd["layers.1.weight"]; r,c,s=match(W2m,W2t); W2al=torch.zeros_like(W2t); W2al[c]=W2m[r]*s[:,None]
    b2a=torch.zeros_like(b2t); b2a[c]=sd["layers.1.bias"][r]*s
    W3a=torch.zeros_like(W3t); W3a[:,c]=sd["layers.2.weight"][:,r]*s[None,:]; b3a=sd["layers.2.bias"]+(sd["layers.2.weight"][:,r][:,s<0]).sum(1)
    eta=torch.cat([b2a,W3a.reshape(-1),b3a,sd["layers.3.weight"].reshape(-1),sd["layers.3.bias"],sd["layers.4.weight"].reshape(-1),sd["layers.4.bias"]])
    return W2al, eta
gg=torch.Generator(device=dev).manual_seed(0); H=torch.sigmoid(torch.randn(40,128,generator=gg,device=dev)*1.8).clamp(1e-3,1-1e-3)
def ueta(e): return e[:80],e[80:3280].reshape(40,80),e[3280:3320],e[3320:4600].reshape(32,40),e[4600:4632],e[4632:4952].reshape(10,32),e[4952:4962]
def obs(W2,e):
    b2,W3,b3,W4,b4,W5,b5=ueta(e)
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2)
    z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4)
    out=s4@W5.t()+b5; U=(W2@B.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3
    dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    return torch.cat([out.reshape(-1),dout.reshape(-1)])
Y=obs(W2t,eta_true).detach()   # true-function observations
def werr(W2):
    Cp=torch.cdist(W2,W2t);Cm=torch.cdist(-W2,W2t);C=torch.minimum(Cp,Cm).cpu().numpy()
    ri,ci=linear_sum_assignment(C);e=torch.tensor([C[ri[i],ci[i]] for i in range(len(ri))]);return float((e/nt[ci].cpu()).mean())
print(f"{'m':>2} {'W2%':>4} {'S=|rW|':>9} {'N=|reta|':>9} {'T=|rtot|':>9} {'N/S':>5} {'T/S':>6} {'cos(reta,rW)':>13}")
for m in range(8):
    W2al,em=align(mem[m])
    Am=W2al@B.t(); Rm=W2al-(Am@B)         # out-of-rowspace residual; W2(Am)=Rm+Am@B=W2al exactly
    W2_full=Rm+Am@B
    rW=obs(W2_full,eta_true)-Y             # member W2, TRUE eta
    reta=obs(W2_full,em)-obs(W2_full,eta_true)  # member W2: member eta vs true eta
    rtot=obs(W2_full,em)-Y                 # member's actual function - Y
    S=float(rW.norm()); N=float(reta.norm()); T=float(rtot.norm()); cos=float((reta@rW)/(N*S+1e-30))
    print(f"{m:>2} {werr(W2al)*100:>3.0f}% {S:>9.3e} {N:>9.3e} {T:>9.3e} {N/S:>5.2f} {T/S:>6.3f} {cos:>13.3f}",flush=True)
