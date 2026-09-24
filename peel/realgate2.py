import torch, numpy as np
from torch.func import jvp, vjp
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
def germ_deriv(s,W2,b2,W3,b3,w4,b4):
    p2=s@W2.T+b2; q=sig(p2); qd=q*(1-q); qdd=qd*(1-2*q)
    u=q@W3.T+b3; h=sig(u); hd=h*(1-h); hdd=hd*(1-2*h)
    dq=qd[:,:,None]*W2[None,:,:]; du=torch.einsum('nr,prk->pnk',W3,dq)
    G1=torch.einsum('on,pn,pnk->pok',w4,hd,du)
    d2u=torch.einsum('nr,pr,ri,rj->pnij',W3,qdd,W2,W2)
    G2=torch.einsum('on,pn,pni,pnj->poij',w4,hdd,du,du)+torch.einsum('on,pn,pnij->poij',w4,hd,d2u)
    return G1,G2
def raw_jet_S(S,beta,down,tps,k):
    z=tps@S.T+beta; s=sig(z); D=s*(1-s); sdd=D*(1-2*s)
    G1,G2=germ_deriv(s,*down)
    Ft=torch.einsum('ai,pa,poa->poi',S,D,G1)
    M=D[:,None,:,None]*G2*D[:,None,None,:]+torch.diag_embed(sdd[:,None,:]*G1)
    Fij=torch.einsum('ai,poab,bj->poij',S,M,S); iu=torch.triu_indices(k,k)
    return torch.cat([Ft.reshape(-1),Fij[:,:,iu[0],iu[1]].reshape(-1)])
def make(dims,seed):
    g=torch.Generator().manual_seed(seed); k,W,m,O=dims
    S=torch.eye(k)+8e-5*torch.randn(k,k,generator=g)
    b1=(2*torch.rand(k,generator=g)-1)*0.5
    down=[torch.randn(W,k,generator=g)/np.sqrt(k),(2*torch.rand(W,generator=g)-1)*0.6,
          torch.randn(m,W,generator=g)/np.sqrt(W),(2*torch.rand(m,generator=g)-1)*0.5,
          torch.randn(O,m,generator=g)/np.sqrt(m),(2*torch.rand(O,generator=g)-1)*0.5]
    return S,b1,down
def flat(S,beta,down): return torch.cat([S.reshape(-1),beta]+[d.reshape(-1) for d in down])
def unpack(th,dims,shapes):
    k=dims[0]; S=th[:k*k].reshape(k,k); beta=th[k*k:k*k+k]; i=k*k+k; down=[]
    for sh in shapes:
        n=int(np.prod(sh)); down.append(th[i:i+n].reshape(sh)); i+=n
    return S,beta,down
def cg(mv,b,iters=120,tol=1e-15):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rs=r@r
    for _ in range(iters):
        Ap=mv(p);a=rs/(p@Ap+1e-300);x=x+a*p;r=r-a*Ap;rs2=r@r
        if rs2.sqrt()<tol: break
        p=r+(rs2/rs)*p;rs=rs2
    return x
def gncg(th0,rf,iters=25):
    th=th0.clone();lam=1e-6;r=rf(th);c=float(r@r)
    for it in range(iters):
        rr,vjpf=vjp(rf,th);g=vjpf(rr)[0];ok=False
        for _ in range(12):
            JTJ=lambda v:vjpf(jvp(rf,(th,),(v,))[1])[0]+lam*v
            d=cg(JTJ,-g);thn=th+d;rn=rf(thn);cn=float(rn@rn)
            if cn<c:th=thn;c=cn;lam=max(lam*0.5,1e-13);ok=True;break
            lam*=4
            if lam>1e11:break
        if not ok or c<1e-26:break
    return th,c
def gate(dims,seeds,alphas,ndir,nprobe):
    k=dims[0]
    for sd in seeds:
        Sstar,b1,down=make(dims,sd); shapes=[tuple(d.shape) for d in down]
        gp=torch.Generator().manual_seed(sd+9); tps=(2*torch.rand(nprobe,k,generator=gp)-1)*1.2
        with torch.no_grad(): Fm=raw_jet_S(Sstar,b1,down,tps,k); sc=Fm.abs().max()
        rng=np.random.default_rng(sd); b0=b1+0.05*torch.tensor(2*rng.random(k)-1)
        for al in alphas:
            eb=[];es=[]
            for dd in range(ndir):
                g2=torch.Generator().manual_seed(sd*500+int(al*1e5)+dd+1)
                down0=[W+al*(W.norm()/(X:=torch.randn(*W.shape,generator=g2)).norm())*X for W in down]
                th0=flat(torch.eye(k),b0,down0)   # S0=I, beta0=guess, downstream truth+alpha
                rf=lambda th: (raw_jet_S(*unpack(th,dims,shapes),tps,k)-Fm)/sc
                thM,c=gncg(th0,rf)
                Sh,bh,_=unpack(thM,dims,shapes); eb.append(float((bh-b1).abs().max())); es.append(float((Sh-Sstar).abs().max()))
            eb=np.array(eb)
            print(f"  {dims} s{sd} a={al:.2f}: P(bias<1e-4)={(eb<1e-4).mean():.2f} med-bias={np.median(eb):.1e} max||S-S*||={max(es):.1e}",flush=True)
print("JOINT (S,beta,theta) GN-CG-LM, S0=I (non-oracle coords):",flush=True)
print("TOY 6->8->4->4 multi-direction basin:",flush=True)
gate([6,8,4,4],[0,1],[0.05,0.10,0.20],10,12)
print("REAL 24->32->16->8 multi-direction:",flush=True)
gate([24,32,16,8],[0],[0.0,0.10,0.20],5,8)
