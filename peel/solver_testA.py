"""TEST A (reviewer): remove the Frankenstein init. For each committee member m, use its OWN coherent
pair (W2^m projected into B, eta^m = its own downstream). Take ONE projected-A step. Score after.
Does any coherent member pair give a TRUTH-WARD A step? Sealed (socket only, no truth in process).
NOTE: this does NOT fix the q->h_true coordinate warp (confound #2) -- that's Test B.
"""
import socket, io, torch
from scipy.optimize import linear_sum_assignment
from torch.func import jvp, vjp, jacrev
torch.set_default_dtype(torch.float64); dev="cuda"
SOCK="/tmp/claude-1001/-home-judah/2cb1357d-51f9-4a19-9d02-8f39bb198f3b/scratchpad/bb.sock"; PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
c=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); c.connect(SOCK)
def _rall(n):
    b=b""
    while len(b)<n: b+=c.recv(n-len(b))
    return b
def _send(o):
    buf=io.BytesIO(); torch.save(o,buf); d=buf.getvalue(); c.sendall(len(d).to_bytes(8,"big")+d)
def _recv():
    n=int.from_bytes(_rall(8),"big"); return torch.load(io.BytesIO(_rall(n)),weights_only=False)
NQ=[0]
def query(x):
    xx=x if x.dim()>1 else x.unsqueeze(0); NQ[0]+=xx.shape[0]; _send({"x":xx.cpu()}); return _recv()["out"].to(dev)
inp=torch.load(PEEL+"solver_inputs.pt",map_location="cpu",weights_only=False)
mem=[{k:v.to(dev).double() for k,v in sd.items()} for sd in inp["pop_states"]]
def match(A,Bm):
    Cp=torch.cdist(A,Bm);Cm=torch.cdist(-A,Bm);C=torch.minimum(Cp,Cm);r,cc=linear_sum_assignment(C.cpu().numpy())
    r=torch.tensor(r,device=dev);cc=torch.tensor(cc,device=dev);return r,cc
W1c=mem[0]["layers.0.weight"]; b1c=mem[0]["layers.0.bias"]; W1cp=W1c.t()@torch.linalg.inv(W1c@W1c.t())
def hq(Q): return query((torch.log(Q/(1-Q))-b1c)@W1cp.t())
ref=mem[0]["layers.1.weight"]; AW=[ref.clone()]
for sd in mem[1:]:
    Wm=sd["layers.1.weight"]; r,cc=match(Wm,ref); s=torch.sign((Wm[r]*ref[cc]).sum(1)); Wa=torch.zeros_like(Wm); Wa[cc]=Wm[r]*s[:,None]; AW.append(Wa)
W2guess=torch.stack(AW).median(0).values; b2g=mem[0]["layers.1.bias"]; W2gp=torch.linalg.pinv(W2guess)
def recover_B(fd=1e-4,reps=2):
    gq=torch.Generator(device=dev).manual_seed(5); rows=[]
    for rep in range(reps):
        for j in range(80):
            tt=(torch.randn(80,generator=gq,device=dev)*2.0); tt[j]=0.0
            q0=torch.clamp(W2gp@(tt-b2g),1e-3,1-1e-3); E=torch.eye(128,device=dev)
            rows.append(((hq((q0+fd*E).clamp(1e-4,1-1e-4))-hq((q0-fd*E).clamp(1e-4,1-1e-4)))/(2*fd)).t())
    M=torch.cat(rows,0); U,S,Vh=torch.linalg.svd(M,full_matrices=False); return Vh[:80]
B=recover_B(); print(f"[testA] B recovered; q={NQ[0]}",flush=True)
gq=torch.Generator(device=dev).manual_seed(1); Hs=[]
for j in range(40):
    tt=torch.randn(80,generator=gq,device=dev)*2.0; tt[j%80]=0.0; Hs.append(torch.clamp(W2gp@(tt-b2g),1e-3,1-1e-3))
H=torch.stack(Hs); fd=1e-4
val_t=hq(H); jt=[]
for p in range(H.shape[0]):
    jt.append(((hq((H[p][None,:]+fd*B).clamp(1e-4,1-1e-4))-hq((H[p][None,:]-fd*B).clamp(1e-4,1-1e-4)))/(2*fd)))
target=torch.cat([val_t.reshape(-1),torch.stack(jt).reshape(-1)]); vsc=float(target.abs().max())
print(f"[testA] targets built; q={NQ[0]}",flush=True)
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
def proj_step(A0,em):
    A=A0.clone(); r=(observ(A,em)-target)/vsc
    Jeta=jacrev(lambda e:(observ(A,e)-target)/vsc,chunk_size=256)(em); Q,_=torch.linalg.qr(Jeta)
    fA=lambda a:(observ(a.reshape(80,80),em)-target)/vsc
    Jv=lambda v:jvp(fA,(A.reshape(-1),),(v,))[1]; Jt=lambda u:vjp(fA,A.reshape(-1))[1](u)[0]
    Aop=lambda v:(lambda j:j-Q@(Q.t()@j))(Jv(v)); Atop=lambda u:Jt(u-Q@(Q.t()@u))
    rp=r-Q@(Q.t()@r); base=float(rp@rp); damp=1e-3
    for _ in range(10):
        d=lsqr(Aop,Atop,-rp,6400,damp); An=A+d.reshape(80,80); rn=(observ(An,em)-target)/vsc; rpn=rn-Q@(Q.t()@rn)
        if float(rpn@rpn)<base: return An
        damp*=4
    return A0
results={}
for m in range(8):
    W2m=mem[m]["layers.1.weight"]
    em=torch.cat([mem[m]["layers.1.bias"].reshape(-1),mem[m]["layers.2.weight"].reshape(-1),mem[m]["layers.2.bias"],mem[m]["layers.3.weight"].reshape(-1),mem[m]["layers.3.bias"],mem[m]["layers.4.weight"].reshape(-1),mem[m]["layers.4.bias"]])
    A0=W2m@B.t(); A1=proj_step(A0,em)
    results[m]=((A0@B).cpu(),(A1@B).cpu())
    print(f"[testA] member {m} step done",flush=True)
torch.save({"results":results,"guess_proj":(W2guess@B.t()@B).cpu(),"guess_raw":W2guess.cpu()},PEEL+"testA_out.pt")
print(f"[testA] total q={NQ[0]}",flush=True); _send({"cmd":"stop"}); c.close()
