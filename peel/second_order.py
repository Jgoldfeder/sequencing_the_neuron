"""SECOND-ORDER compensation check (reviewer, stricter). At exact coherent member points, decompose the
residual BY DERIVATIVE ORDER: k=0 value, k=1 first-jets(along B), k=2 MIXED bilinear Hessian v_a^T H v_b
(randomized sketch, i!=j included). For each order report N_k/S_k, T_k/S_k, cos(r_eta^k, r_W^k).
If cos^(2) is NOT ~-1, curvature breaks the learned first-order compensation. Oracle diagnostic."""
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
gg=torch.Generator(device=dev).manual_seed(0); H=torch.sigmoid(torch.randn(30,128,generator=gg,device=dev)*1.8).clamp(1e-3,1-1e-3)
M=200; gp=torch.Generator(device=dev).manual_seed(2)
Va=(torch.randn(M,80,generator=gp,device=dev))@B; Vb=(torch.randn(M,80,generator=gp,device=dev))@B  # mixed pairs in rowspace
Va=Va/Va.norm(dim=1,keepdim=True); Vb=Vb/Vb.norm(dim=1,keepdim=True)
def ueta(e): return e[:80],e[80:3280].reshape(40,80),e[3280:3320],e[3320:4600].reshape(32,40),e[4600:4632],e[4632:4952].reshape(10,32),e[4952:4962]
def orders(W2,e):
    b2,W3,b3,W4,b4,W5,b5=ueta(e)
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2); spp2=sp2*(1-2*s2)
    U1=(W2@B.t()).t(); ds1=sp2[:,None,:]*U1[None,:,:]
    za=Va@W2.t(); zb=Vb@W2.t(); dsa=sp2[:,None,:]*za[None,:,:]; dsb=sp2[:,None,:]*zb[None,:,:]; dds=spp2[:,None,:]*(za*zb)[None,:,:]
    z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3); spp3=sp3*(1-2*s3)
    dz1=ds1@W3.t(); ds1=sp3[:,None,:]*dz1
    dza=dsa@W3.t(); dzb=dsb@W3.t(); ddz=dds@W3.t(); dds=spp3[:,None,:]*dza*dzb+sp3[:,None,:]*ddz; dsa=sp3[:,None,:]*dza; dsb=sp3[:,None,:]*dzb
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4); spp4=sp4*(1-2*s4)
    dz1=ds1@W4.t(); ds1=sp4[:,None,:]*dz1
    dza=dsa@W4.t(); dzb=dsb@W4.t(); ddz=dds@W4.t(); dds=spp4[:,None,:]*dza*dzb+sp4[:,None,:]*ddz; dsa=sp4[:,None,:]*dza; dsb=sp4[:,None,:]*dzb
    out=s4@W5.t()+b5; dout1=ds1@W5.t(); ddout=dds@W5.t()
    return out.reshape(-1), dout1.reshape(-1), ddout.reshape(-1)
Y=[o.detach() for o in orders(W2t,eta_true)]
def stats(rW,re):
    S=float(rW.norm()); N=float(re.norm()); cos=float((re@rW)/(N*S+1e-30)); return S,N,cos
agg={0:[],1:[],2:[]}
for m in range(8):
    W2m,em=align(mem[m])
    oWt=orders(W2m,eta_true); oWm=orders(W2m,em)
    for k in range(3):
        rW=oWt[k]-Y[k]; re=oWm[k]-oWt[k]; rtot=oWm[k]-Y[k]
        S,N,cos=stats(rW,re); agg[k].append((N/S,float(rtot.norm())/S,cos))
names=["value(k=0)","jacobian(k=1)","hessian-mix(k=2)"]
print(f"{'order':>18} {'N/S':>6} {'T/S':>6} {'cos(reta,rW)':>13}  (mean over 8 members)")
for k in range(3):
    a=agg[k]; ns=sum(x[0] for x in a)/8; ts=sum(x[1] for x in a)/8; cs=sum(x[2] for x in a)/8
    print(f"{names[k]:>18} {ns:>6.2f} {ts:>6.3f} {cs:>13.3f}",flush=True)
print("\nper-member cos at k=2:", " ".join(f"{agg[2][m][2]:+.2f}" for m in range(8)))
