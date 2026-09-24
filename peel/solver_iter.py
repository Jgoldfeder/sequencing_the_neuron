"""ITERATED sealed ordinary-probe trust-region projected-A (reviewer). Small steps, re-linearize each
iter, fresh ordinary probes, TRUTH-FREE lambda selection on the sealed projected residual
L_perp(A)=||P_eta^perp (Phi(A,eta_m)-Y_BB)||^2. Coherent member 0. Save W2 trajectory; score after.
Does 23.44% descend under iteration, or improve-once-then-stall?"""
import socket, io, torch
from torch.func import jvp, vjp, jacrev
torch.set_default_dtype(torch.float64); dev="cuda"
import sys; sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
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
W1c=mem[0]["layers.0.weight"]; b1c=mem[0]["layers.0.bias"]; W1cp=W1c.t()@torch.linalg.inv(W1c@W1c.t())
def xofq(Q): return (torch.log(Q/(1-Q))-b1c)@W1cp.t()
b2g=mem[0]["layers.1.bias"]; W2g=mem[0]["layers.1.weight"]; W2gp=torch.linalg.pinv(W2g)
def recover_B(fd=3e-4):
    gq=torch.Generator(device=dev).manual_seed(5); rows=[]
    for rep in range(2):
        for j in range(80):
            tt=(torch.randn(80,generator=gq,device=dev)*2.0); tt[j]=0.0; q0=torch.clamp(W2gp@(tt-b2g),1e-3,1-1e-3); E=torch.eye(128,device=dev)
            rows.append(((query(xofq((q0+fd*E).clamp(1e-4,1-1e-4)))-query(xofq((q0-fd*E).clamp(1e-4,1-1e-4))))/(2*fd)).t())
    Msvd=torch.cat(rows,0); U,S,Vh=torch.linalg.svd(Msvd,full_matrices=False); return Vh[:80]
B=recover_B(); print(f"[iter] B recovered q={NQ[0]}",flush=True)
W2m0=mem[0]["layers.1.weight"]; Rm=W2m0-(W2m0@B.t())@B; A=W2m0@B.t()
em0=torch.cat([mem[0]["layers.1.bias"].reshape(-1),mem[0]["layers.2.weight"].reshape(-1),mem[0]["layers.2.bias"],mem[0]["layers.3.weight"].reshape(-1),mem[0]["layers.3.bias"],mem[0]["layers.4.weight"].reshape(-1),mem[0]["layers.4.bias"]])
def ueta(e): return e[:80],e[80:3280].reshape(40,80),e[3280:3320],e[3320:4600].reshape(32,40),e[4600:4632],e[4632:4952].reshape(10,32),e[4952:4962]
def obs_cand(Aa,e,H):
    b2,W3,b3,W4,b4,W5,b5=ueta(e); W2=Rm+Aa@B
    z2=H@W2.t()+b2; s2=torch.sigmoid(z2); sp2=s2*(1-s2); z3=s2@W3.t()+b3; s3=torch.sigmoid(z3); sp3=s3*(1-s3)
    z4=s3@W4.t()+b4; s4=torch.sigmoid(z4); sp4=s4*(1-s4); out=s4@W5.t()+b5; U=(W2@B.t()).t()
    ds2=sp2[:,None,:]*U[None,:,:]; dz3=ds2@W3.t(); ds3=sp3[:,None,:]*dz3; dz4=ds3@W4.t(); ds4=sp4[:,None,:]*dz4; dout=ds4@W5.t()
    return torch.cat([out.reshape(-1),dout.reshape(-1)])
def bb_target(H,fd=3e-4):
    P=H.shape[0]; val=query(xofq(H))
    Hp=(H[:,None,:]+fd*B[None,:,:]).reshape(-1,128).clamp(1e-4,1-1e-4); Hm=(H[:,None,:]-fd*B[None,:,:]).reshape(-1,128).clamp(1e-4,1-1e-4)
    op=query(xofq(Hp)).reshape(P,80,10); om=query(xofq(Hm)).reshape(P,80,10); jt=(op-om)/(2*fd)
    return torch.cat([val.reshape(-1),jt.reshape(-1)])
def lsqr(Aop,Atop,b,n,damp,iters=70):
    beta=b.norm()
    if float(beta)==0: return torch.zeros(n,device=b.device)
    u=b/beta;v=Atop(u);alfa=v.norm();v=v/alfa.clamp_min(1e-30);w=v.clone();x=torch.zeros(n,device=b.device);phibar=beta;rhobar=alfa
    for _ in range(iters):
        u=Aop(v)-alfa*u;beta=u.norm();u=u/beta.clamp_min(1e-30); v=Atop(u)-beta*v;alfa=v.norm();v=v/alfa.clamp_min(1e-30)
        rb1=(rhobar**2+damp**2).sqrt();c1=rhobar/rb1;phibar=c1*phibar; rho=(rb1**2+beta**2).sqrt();cc=rb1/rho;sg=beta/rho;th=sg*alfa;rhobar=-cc*alfa;phi=cc*phibar;phibar=sg*phibar
        x=x+(phi/rho)*w;w=v-(th/rho)*w
    return x
traj=[(Rm+A@B).cpu()]; lams=[]
for it in range(60):
    gm=torch.Generator(device=dev).manual_seed(100+it); H=torch.sigmoid(torch.randn(60,128,generator=gm,device=dev)*1.8).clamp(1e-4,1-1e-4)
    target=bb_target(H).detach(); sc=float(target.abs().max()); r=(obs_cand(A,em0,H)-target)/sc
    Jn=jacrev(lambda ee:(obs_cand(A,ee,H)-target)/sc,chunk_size=256)(em0); Q,_=torch.linalg.qr(Jn)
    fA=lambda a:(obs_cand(a.reshape(80,80),em0,H)-target)/sc
    Jv=lambda v:jvp(fA,(A.reshape(-1),),(v,))[1]; Jt=lambda u:vjp(fA,A.reshape(-1))[1](u)[0]
    Aop=lambda v:(lambda j:j-Q@(Q.t()@j))(Jv(v)); Atop=lambda u:Jt(u-Q@(Q.t()@u)); rp=r-Q@(Q.t()@r)
    d=lsqr(Aop,Atop,-rp,6400,1e-3).reshape(80,80)
    # TRUTH-FREE line search on sealed projected residual
    def Lperp(Aa):
        rr=(obs_cand(Aa,em0,H)-target)/sc; return float(((rr-Q@(Q.t()@rr))**2).sum())
    cand=[(0.0,Lperp(A))]+[(l,Lperp(A+l*d)) for l in (0.03,0.1,0.3)]
    lam=min(cand,key=lambda z:z[1])[0]; A=A+lam*d; lams.append(lam); traj.append((Rm+A@B).cpu())
    torch.save({"traj":traj,"lams":lams},PEEL+"iter_out.pt")
    print(f"[iter {it:2d}] lambda={lam} q={NQ[0]}",flush=True)
torch.save({"traj":traj,"lams":lams},PEEL+"iter_out.pt"); print(f"[iter] done q={NQ[0]}",flush=True); _send({"cmd":"stop"}); c.close()
