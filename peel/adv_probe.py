"""SEALED adversarial probe mining (reviewer). Selection uses ONLY committee-member forwards f_m and the
black-box f_BB (no truth): D_m(q)=||f_m(x(q))-f_BB(x(q))|| and D_ens(q)=Var_m f_m(x(q)), x(q)=W1c^+(logit q - b1c).
Broadly sample q (scales .5,1,2,4,8), retain top-discrepancy / top-disagreement probes. AFTER freezing,
unlock truth diagnostically and recompute compensation geometry (T/S, cos(r_eta,r_W) by order) on those
probes vs ordinary. If adversarial probes give cos off -1 / large T/S, the compensation BREAKS there."""
import sys, torch
torch.set_default_dtype(torch.float32); dev="cuda"
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"; PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
pop=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False); dims=pop["dims"]
# ---- SEALED objects: black box (forward only) + committee member nets ----
bb=MLP(dims,act="sigmoid").to(dev).float(); bb.load_state_dict(pop["teacher_state"]); bb.eval()
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False)
memnet=[MLP(dims,act="sigmoid").to(dev).float() for _ in pk["pop_states"]]
for n,sd in zip(memnet,pk["pop_states"]): n.load_state_dict({k:v.to(dev).float() for k,v in sd.items()}); n.eval()
mem=[{k:v.to(dev).float() for k,v in sd.items()} for sd in pk["pop_states"]]
W1c=mem[0]["layers.0.weight"]; b1c=mem[0]["layers.0.bias"]; W1cp=W1c.t()@torch.linalg.inv(W1c@W1c.t())
def xofq(Q): return (torch.log(Q/(1-Q))-b1c)@W1cp.t()
@torch.no_grad()
def fwd(net,Q): return net(xofq(Q))
# ---- sample broad q pool, compute sealed discrepancies ----
g=torch.Generator(device=dev).manual_seed(1); pool=[]
for s in [0.5,1.0,2.0,4.0,8.0]:
    pool.append(torch.sigmoid(torch.randn(3000,128,generator=g,device=dev)*s).clamp(1e-4,1-1e-4))
Q=torch.cat(pool)
with torch.no_grad():
    fbb=fwd(bb,Q); f0=fwd(memnet[0],Q)
    D0=(f0-fbb).norm(dim=1)
    allm=torch.stack([fwd(memnet[m],Q) for m in range(8)])   # (8,N,10)
    Dens=allm.var(0).sum(1)                                  # committee disagreement
print(f"[sealed] pool {Q.shape[0]}. member0 D range [{float(D0.min()):.2e},{float(D0.max()):.2e}] median {float(D0.median()):.2e}")
top_adv=Q[torch.topk(D0,30).indices]          # max member0-vs-BB residual
top_dis=Q[torch.topk(Dens,30).indices]        # max committee disagreement
gm=torch.Generator(device=dev).manual_seed(9); H_ord=torch.sigmoid(torch.randn(30,128,generator=gm,device=dev)*1.8).clamp(1e-4,1-1e-4)
# ================= AFTER FREEZE: unlock truth, decompose compensation =================
W2t=bb.layers[1].weight.detach(); b2t=bb.layers[1].bias.detach(); W3t=bb.layers[2].weight.detach(); b3t=bb.layers[2].bias.detach()
W4t=bb.layers[3].weight.detach(); b4t=bb.layers[3].bias.detach(); W5t=bb.layers[4].weight.detach(); b5t=bb.layers[4].bias.detach(); nt=W2t.norm(dim=1)
Uw,Sw,Vh=torch.linalg.svd(W2t,full_matrices=False); Bb=Vh[:80]
eta_true=torch.cat([b2t,W3t.reshape(-1),b3t,W4t.reshape(-1),b4t,W5t.reshape(-1),b5t])
def match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,c=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r,device=dev);c=torch.tensor(c,device=dev);return r,c,torch.where(Cm[r.cpu(),c.cpu()]<Cp[r.cpu(),c.cpu()],-1.,1.).to(dev)
def align(sd):
    W2m=sd["layers.1.weight"]; r,c,s=match(W2m,W2t); W2al=torch.zeros_like(W2t); W2al[c]=W2m[r]*s[:,None]
    b2a=torch.zeros_like(b2t); b2a[c]=sd["layers.1.bias"][r]*s
    W3a=torch.zeros_like(W3t); W3a[:,c]=sd["layers.2.weight"][:,r]*s[None,:]; b3a=sd["layers.2.bias"]+(sd["layers.2.weight"][:,r][:,s<0]).sum(1)
    eta=torch.cat([b2a,W3a.reshape(-1),b3a,sd["layers.3.weight"].reshape(-1),sd["layers.3.bias"],sd["layers.4.weight"].reshape(-1),sd["layers.4.bias"]])
    return W2al, eta
W2m0,em0=align(mem[0])
def ueta(e): return e[:80],e[80:3280].reshape(40,80),e[3280:3320],e[3320:4600].reshape(32,40),e[4600:4632],e[4632:4952].reshape(10,32),e[4952:4962]
def orders(W2,e,H):
    b2,W3,b3,W4,b4,W5,b5=ueta(e)
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2)
    z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4)
    out=s4@W5.t()+b5; U=(W2@Bb.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3
    dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    return out.reshape(-1), dout.reshape(-1)
def comp(H,tag):
    oWt=orders(W2m0,eta_true,H); oWm=orders(W2m0,em0,H); oT=orders(W2t,eta_true,H)
    for k,nm in [(0,"val"),(1,"jac")]:
        rW=oWt[k]-oT[k]; re=oWm[k]-oWt[k]; S=float(rW.norm()); N=float(re.norm()); rt=float((oWm[k]-oT[k]).norm()); cos=float((re@rW)/(N*S+1e-30))
        print(f"  {tag:28s} k={k}({nm}): N/S {N/S:.2f}  T/S {rt/S:.3f}  cos {cos:+.3f}",flush=True)
print("\ncompensation geometry (member 0) on ORDINARY vs SEALED-ADVERSARIAL probes:")
comp(H_ord,"ordinary(std1.8)")
comp(top_adv,"adversarial(max D_m0)")
comp(top_dis,"adversarial(max committee-var)")
