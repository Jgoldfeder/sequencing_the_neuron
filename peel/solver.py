"""SOLVER (no truth). Reads solver_inputs.pt + queries the black-box server only. Recovers honest B,
then runs the PROJECTED-A update (reviewer): step A only, quotient the candidate downstream tangent,
downstream NEVER updated. Writes W2_est.pt. No teacher_state / W2* / true B anywhere in this process.
"""
import socket, io, torch, time
from scipy.optimize import linear_sum_assignment
from torch.func import jvp, vjp, jacrev
torch.set_default_dtype(torch.float64); dev="cuda"
SOCK="/tmp/claude-1001/-home-judah/2cb1357d-51f9-4a19-9d02-8f39bb198f3b/scratchpad/bb.sock"; PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
# ---- connect to sealed black box ----
c=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); c.connect(SOCK)
def _rall(n):
    b=b""
    while len(b)<n:
        d=c.recv(n-len(b)); b+=d
    return b
def _send(o):
    buf=io.BytesIO(); torch.save(o,buf); d=buf.getvalue(); c.sendall(len(d).to_bytes(8,"big")+d)
def _recv():
    n=int.from_bytes(_rall(8),"big"); return torch.load(io.BytesIO(_rall(n)),weights_only=False)
NQ=[0]
def query(x):
    xx=x if x.dim()>1 else x.unsqueeze(0); NQ[0]+=xx.shape[0]; _send({"x":xx.cpu()}); return _recv()["out"].to(dev)
# ---- stripped inputs (no truth) ----
inp=torch.load(PEEL+"solver_inputs.pt",map_location="cpu",weights_only=False)
mem=[{k:v.to(dev).double() for k,v in sd.items()} for sd in inp["pop_states"]]
def match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,cc=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r,device=dev);cc=torch.tensor(cc,device=dev);return r,cc,torch.where(Cm[r.cpu(),cc.cpu()]<Cp[r.cpu(),cc.cpu()],-1.,1.).to(dev)
# committee L1 coordinate (our solved L1)
W1c=mem[0]["layers.0.weight"]; b1c=mem[0]["layers.0.bias"]; W1cp=W1c.t()@torch.linalg.inv(W1c@W1c.t())
def hq(Q): return query((torch.log(Q/(1-Q))-b1c)@W1cp.t())
# honest committee W2 guess + downstream (member-0), truth-free
ref=mem[0]["layers.1.weight"]; AW=[ref.clone()]
for sd in mem[1:]:
    Wm=sd["layers.1.weight"]; r,cc,s=match(Wm,ref); Wa=torch.zeros_like(Wm); Wa[cc]=Wm[r]*s[:,None]; AW.append(Wa)
W2guess=torch.stack(AW).median(0).values
b2g=mem[0]["layers.1.bias"]; W3g=mem[0]["layers.2.weight"]; b3g=mem[0]["layers.2.bias"]
W4g=mem[0]["layers.3.weight"]; b4g=mem[0]["layers.3.bias"]; W5g=mem[0]["layers.4.weight"]; b5g=mem[0]["layers.4.bias"]
eta=torch.cat([b2g.reshape(-1),W3g.reshape(-1),b3g,W4g.reshape(-1),b4g,W5g.reshape(-1),b5g])   # FROZEN downstream
W2gp=torch.linalg.pinv(W2guess)
# ---- honest B: sealed rowspace recovery (fd=1e-4) ----
def recover_B(fd=1e-4,reps=2):
    gq=torch.Generator(device=dev).manual_seed(5); rows=[]
    for rep in range(reps):
        for j in range(80):
            tt=(torch.randn(80,generator=gq,device=dev)*2.0); tt[j]=0.0
            q0=torch.clamp(W2gp@(tt-b2g),1e-3,1-1e-3); E=torch.eye(128,device=dev)
            Jp=hq((q0+fd*E).clamp(1e-4,1-1e-4)); Jm=hq((q0-fd*E).clamp(1e-4,1-1e-4)); rows.append(((Jp-Jm)/(2*fd)).t())
    M=torch.cat(rows,0); U,S,Vh=torch.linalg.svd(M,full_matrices=False); return Vh[:80]
B=recover_B()
print(f"[solver] B recovered; queries so far {NQ[0]}",flush=True)
# ---- probes + candidate observ (value + jets along B) ----
gq=torch.Generator(device=dev).manual_seed(1); Hs=[]
for j in range(40):
    tt=torch.randn(80,generator=gq,device=dev)*2.0; tt[j%80]=0.0; Hs.append(torch.clamp(W2gp@(tt-b2g),1e-3,1-1e-3))
H=torch.stack(Hs); fd=1e-4
# sealed TARGET: value + FD jets along B at each probe
val_t=hq(H)                                                  # (P,10)
jt=[]
for p in range(H.shape[0]):
    Qp=(H[p][None,:]+fd*B).clamp(1e-4,1-1e-4); Qm=(H[p][None,:]-fd*B).clamp(1e-4,1-1e-4)
    jt.append(((hq(Qp)-hq(Qm))/(2*fd)))                      # (80,10)
jt=torch.stack(jt)                                           # (P,80,10)
target=torch.cat([val_t.reshape(-1),jt.reshape(-1)]); vsc=float(target.abs().max())
print(f"[solver] targets built (sealed); queries {NQ[0]}",flush=True)
def ueta(e): return e[:80],e[80:3280].reshape(40,80),e[3280:3320],e[3320:4600].reshape(32,40),e[4600:4632],e[4632:4952].reshape(10,32),e[4952:4962]
def observ(A,e):
    b2,W3,b3,W4,b4,W5,b5=ueta(e); W2=A@B
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2)
    z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4)
    out=s4@W5.t()+b5; U=(W2@B.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3
    dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    return torch.cat([out.reshape(-1),dout.reshape(-1)])
def lsqr(Aop,Atop,b,n,damp,iters=60):
    beta=b.norm()
    if float(beta)==0: return torch.zeros(n,device=b.device)
    u=b/beta;v=Atop(u);alfa=v.norm();v=v/alfa.clamp_min(1e-30);w=v.clone();x=torch.zeros(n,device=b.device);phibar=beta;rhobar=alfa
    for _ in range(iters):
        u=Aop(v)-alfa*u;beta=u.norm();u=u/beta.clamp_min(1e-30)
        v=Atop(u)-beta*v;alfa=v.norm();v=v/alfa.clamp_min(1e-30)
        rb1=(rhobar**2+damp**2).sqrt();c1=rhobar/rb1;phibar=c1*phibar
        rho=(rb1**2+beta**2).sqrt();cc=rb1/rho;s=beta/rho;theta=s*alfa;rhobar=-cc*alfa;phi=cc*phibar;phibar=s*phibar
        x=x+(phi/rho)*w;w=v-(theta/rho)*w
    return x
# ---- PROJECTED-A: step A only, quotient candidate downstream tangent, eta frozen ----
A=(W2guess@B.t()).clone(); snaps={}
for it in range(20):
    r=(observ(A,eta)-target)/vsc
    Jeta=jacrev(lambda e:(observ(A,e)-target)/vsc,chunk_size=256)(eta)   # candidate nuisance tangent
    Q,_=torch.linalg.qr(Jeta)
    fA=lambda a:(observ(a.reshape(80,80),eta)-target)/vsc
    Jv=lambda v:jvp(fA,(A.reshape(-1),),(v,))[1]; Jt=lambda u:vjp(fA,A.reshape(-1))[1](u)[0]
    Aop=lambda v:(lambda j:j-Q@(Q.t()@j))(Jv(v)); Atop=lambda u:Jt(u-Q@(Q.t()@u))
    rp=r-Q@(Q.t()@r); base=float(rp@rp); damp=1e-3; ok=False
    for _ in range(10):
        d=lsqr(Aop,Atop,-rp,6400,damp); An=A+d.reshape(80,80)
        rn=(observ(An,eta)-target)/vsc; rpn=rn-Q@(Q.t()@rn); cn=float(rpn@rpn)
        if cn<base: A=An;ok=True;break
        damp*=4
    if not ok: break
    if (it+1) in (1,2,4,8,20): snaps[it+1]=(A@B).detach().cpu()
W2_est=(A@B).detach().cpu()
torch.save({"W2_est":W2_est,"W2_guess":(W2guess).detach().cpu(),"snaps":snaps}, PEEL+"W2_est.pt")
print(f"[solver] done, total queries {NQ[0]}, wrote W2_est.pt",flush=True)
_send({"cmd":"stop"}); c.close()
