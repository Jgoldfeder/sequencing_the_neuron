"""SEALED sampled-adversarial projected-A step (reviewer's milestone test). Server owns teacher; solver
uses ONLY socket queries + committee. Honest B_hat (sealed FD). Coherent member 0 (W2=Rm+A B_hat, own eta).
Sampled-adversarial probes: sample q, D0(q)=||member0(x(q)) - bb.query(x(q))|| (member fwd in-solver,
bb via socket) -> top-60. BB FD value+jets targets. One projected-A step (ordinary vs adversarial probes).
Writes W2 estimates. Score afterward. Does the oracle +0.12 signal survive the seal?"""
import socket, io, torch
from torch.func import jvp, vjp, jacrev
from scipy.optimize import linear_sum_assignment
torch.set_default_dtype(torch.float64); dev="cuda"
import sys; sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
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
inp=torch.load(PEEL+"solver_inputs.pt",map_location="cpu",weights_only=False); dims=inp["dims"]
mem=[{k:v.to(dev).double() for k,v in sd.items()} for sd in inp["pop_states"]]
m0net=MLP(dims,act="sigmoid").to(dev).double(); m0net.load_state_dict(mem[0]); m0net.eval()  # committee member (ours)
W1c=mem[0]["layers.0.weight"]; b1c=mem[0]["layers.0.bias"]; W1cp=W1c.t()@torch.linalg.inv(W1c@W1c.t())
def xofq(Q): return (torch.log(Q/(1-Q))-b1c)@W1cp.t()
# honest B_hat (sealed FD)
b2g=mem[0]["layers.1.bias"]; W2g=mem[0]["layers.1.weight"]; W2gp=torch.linalg.pinv(W2g)
def recover_B(fd=3e-4):
    gq=torch.Generator(device=dev).manual_seed(5); rows=[]
    for rep in range(2):
        for j in range(80):
            tt=(torch.randn(80,generator=gq,device=dev)*2.0); tt[j]=0.0; q0=torch.clamp(W2gp@(tt-b2g),1e-3,1-1e-3); E=torch.eye(128,device=dev)
            rows.append(((query(xofq((q0+fd*E).clamp(1e-4,1-1e-4)))-query(xofq((q0-fd*E).clamp(1e-4,1-1e-4))))/(2*fd)).t())
    Msvd=torch.cat(rows,0); U,S,Vh=torch.linalg.svd(Msvd,full_matrices=False); return Vh[:80]
B=recover_B(); print(f"[adv] B recovered q={NQ[0]}",flush=True)
# member 0 coherent
W2m0=mem[0]["layers.1.weight"]; Rm=W2m0-(W2m0@B.t())@B; A0=W2m0@B.t()
em0=torch.cat([mem[0]["layers.1.bias"].reshape(-1),mem[0]["layers.2.weight"].reshape(-1),mem[0]["layers.2.bias"],mem[0]["layers.3.weight"].reshape(-1),mem[0]["layers.3.bias"],mem[0]["layers.4.weight"].reshape(-1),mem[0]["layers.4.bias"]])
# sampled adversarial probe selection (sealed)
gg=torch.Generator(device=dev).manual_seed(1); poolz=torch.cat([torch.randn(1500,128,generator=gg,device=dev)*s for s in [0.5,1,2,4,8]]); pool=torch.sigmoid(poolz).clamp(1e-4,1-1e-4)
with torch.no_grad():
    xx=xofq(pool); fm=m0net(xx); fb=query(xx); D0=(fm-fb).norm(dim=1)
NP=60; H_adv=pool[torch.topk(D0,NP).indices]
gm=torch.Generator(device=dev).manual_seed(9); H_ord=torch.sigmoid(torch.randn(NP,128,generator=gm,device=dev)*1.8).clamp(1e-4,1-1e-4)
print(f"[adv] probes selected q={NQ[0]}; D0 top mean {float(D0.topk(NP).values.mean()):.3e}",flush=True)
def ueta(e): return e[:80],e[80:3280].reshape(40,80),e[3280:3320],e[3320:4600].reshape(32,40),e[4600:4632],e[4632:4952].reshape(10,32),e[4952:4962]
def obs_cand(A,e,H):
    b2,W3,b3,W4,b4,W5,b5=ueta(e); W2=Rm+A@B
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2); z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4); out=s4@W5.t()+b5; U=(W2@B.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3; dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    return torch.cat([out.reshape(-1),dout.reshape(-1)])
def bb_target(H,fd=3e-4):   # sealed value + FD jets along B
    val=query(xofq(H)); jt=[]
    for p in range(H.shape[0]):
        jt.append((query(xofq((H[p][None,:]+fd*B).clamp(1e-4,1-1e-4)))-query(xofq((H[p][None,:]-fd*B).clamp(1e-4,1-1e-4))))/(2*fd))
    return torch.cat([val.reshape(-1),torch.stack(jt).reshape(-1)])
def lsqr(Aop,Atop,b,n,damp,iters=70):
    beta=b.norm()
    if float(beta)==0: return torch.zeros(n,device=b.device)
    u=b/beta;v=Atop(u);alfa=v.norm();v=v/alfa.clamp_min(1e-30);w=v.clone();x=torch.zeros(n,device=b.device);phibar=beta;rhobar=alfa
    for _ in range(iters):
        u=Aop(v)-alfa*u;beta=u.norm();u=u/beta.clamp_min(1e-30); v=Atop(u)-beta*v;alfa=v.norm();v=v/alfa.clamp_min(1e-30)
        rb1=(rhobar**2+damp**2).sqrt();c1=rhobar/rb1;phibar=c1*phibar; rho=(rb1**2+beta**2).sqrt();cc=rb1/rho;sg=beta/rho;th=sg*alfa;rhobar=-cc*alfa;phi=cc*phibar;phibar=sg*phibar
        x=x+(phi/rho)*w;w=v-(th/rho)*w
    return x
def step(H):
    target=bb_target(H).detach(); sc=float(target.abs().max()); r=(obs_cand(A0,em0,H)-target)/sc
    Jn=jacrev(lambda ee:(obs_cand(A0,ee,H)-target)/sc,chunk_size=256)(em0); Q,_=torch.linalg.qr(Jn)
    fA=lambda a:(obs_cand(a.reshape(80,80),em0,H)-target)/sc
    Jv=lambda v:jvp(fA,(A0.reshape(-1),),(v,))[1]; Jt=lambda u:vjp(fA,A0.reshape(-1))[1](u)[0]
    Aop=lambda v:(lambda j:j-Q@(Q.t()@j))(Jv(v)); Atop=lambda u:Jt(u-Q@(Q.t()@u)); rp=r-Q@(Q.t()@r); d=lsqr(Aop,Atop,-rp,6400,1e-3).reshape(80,80)
    return {l:(Rm+(A0+l*d)@B).cpu() for l in (0.03,0.1,0.3,1.0)}
out={"start":(Rm+A0@B).cpu(),"ordinary":step(H_ord),"adversarial":step(H_adv)}
torch.save(out,PEEL+"adv_sealed_out.pt"); print(f"[adv] done q={NQ[0]}",flush=True); _send({"cmd":"stop"}); c.close()
