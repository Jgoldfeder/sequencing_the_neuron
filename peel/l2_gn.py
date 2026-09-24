"""Redo the B-test with a proper 2nd-order solver (matrix-free Gauss-Newton / LM), since Adam only
plateaus on this ill-conditioned (weak-gradient) problem. deeper=true => W2_true is the UNIQUE
loss-0 solution, so if GN-LM reaches it from 15%, the consensus guesses ARE refinable."""
import sys, torch
from torch.func import jvp, vjp
torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
dev = "cuda" if torch.cuda.is_available() else "cpu"
teacher = MLP([784,128,80,40,32,10], act="sigmoid").to(dev); teacher.load_state_dict(torch.load("/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt", map_location=dev, weights_only=False))
Wl=[teacher.layers[i].weight.detach() for i in range(5)]; bl=[teacher.layers[i].bias.detach() for i in range(5)]
W2t=Wl[1]; nt=W2t.norm(dim=1); sh=W2t.shape
def G(h,W2):
    q2=torch.sigmoid(h@W2.t()+bl[1]); q3=torch.sigmoid(q2@Wl[2].t()+bl[2]); q4=torch.sigmoid(q3@Wl[3].t()+bl[3]); return q4@Wl[4].t()+bl[4]
g=torch.Generator(device=dev).manual_seed(0)
H=torch.rand(1500,128,generator=g,device=dev); Yt=G(H,W2t).detach(); sc=Yt.abs().max()
def resid(wf): return ((G(H,wf.reshape(sh))-Yt)/sc).reshape(-1)
def cg(Afun,b,iters=40,tol=1e-10):
    x=torch.zeros_like(b); r=b.clone(); p=r.clone(); rs=r@r
    for _ in range(iters):
        Ap=Afun(p); a=rs/(p@Ap+1e-30); x=x+a*p; r=r-a*Ap; rs2=r@r
        if rs2.sqrt()<tol: break
        p=r+(rs2/rs)*p; rs=rs2
    return x
def relerr(wf): W2=wf.reshape(sh); return float(((W2-W2t).norm(dim=1)/nt).mean())
print(f"loss at TRUE W2 = {float(resid(W2t.reshape(-1))@resid(W2t.reshape(-1))):.2e}  (confirms signal: 0)")
print("GN-LM refinement of W2 (deeper=true), function-value residuals:")
for alpha in [0.05,0.15,0.30]:
    gg=torch.Generator(device=dev).manual_seed(1); P=torch.randn(80,128,generator=gg,device=dev); P=P/P.norm(dim=1,keepdim=True)
    wf=(W2t+alpha*nt[:,None]*P).reshape(-1).clone(); lam=1e-3; r=resid(wf); c=float(r@r)
    for it in range(60):
        _,vjpf=vjp(resid,wf); Jt=lambda u: vjpf(u)[0]
        Jv=lambda v: jvp(resid,(wf,),(v,))[1]
        gvec=Jt(r); A=lambda v: Jt(Jv(v))+lam*v; ok=False
        for _ in range(8):
            dwf=cg(A,-gvec); wn=wf+dwf; rn=resid(wn); cn=float(rn@rn)
            if cn<c: wf=wn; r=rn; c=cn; lam=max(lam*0.4,1e-12); ok=True; break
            lam*=4
        if not ok or c<1e-24: break
    print(f"  alpha={alpha:.2f}: W2 err {alpha:.3f} -> {relerr(wf):.4f}   final loss {c:.2e}", flush=True)
