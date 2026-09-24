"""B-test: local-refinement of W2 by matching G(h) VALUES (not just jets) at reachable cube points.
Seed W2 = true + alpha*perturbation (deeper frozen true), refine by function-value fit, and see if
it converges to true. Sweeps alpha incl. ~0.15 (the consensus-guess quality). If it converges from
0.15, the 34 consensus neurons WERE refinable; if it stalls, function-fit has the same flat valley."""
import sys, torch, numpy as np
torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
dev = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/teachers/teacher_784x128x80x40x32x10_e25_s0_sigmoid.pt"
teacher = MLP([784,128,80,40,32,10], act="sigmoid").to(dev); teacher.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False))
Wl=[teacher.layers[i].weight.detach() for i in range(5)]; bl=[teacher.layers[i].bias.detach() for i in range(5)]
W2t=Wl[1]; nt=W2t.norm(dim=1)
def G(h, W2):
    q2=torch.sigmoid(h@W2.t()+bl[1]); q3=torch.sigmoid(q2@Wl[2].t()+bl[2]); q4=torch.sigmoid(q3@Wl[3].t()+bl[3]); return q4@Wl[4].t()+bl[4]
g=torch.Generator(device=dev).manual_seed(0)
Hc=torch.rand(3000,128,generator=g,device=dev)                    # reachable cube probes (uniform)
Yt=G(Hc,W2t).detach(); yn=Yt.pow(2).mean().sqrt()
def relerr(W2): return float(((W2-W2t).norm(dim=1)/nt).mean())
print("B-test: function-value local refinement of W2 (deeper=true), reachable cube probes:")
print(f"{'alpha':>6} {'W2 err before':>14} {'W2 err after':>13} {'fit rmse/||Y||':>15} {'verdict':>10}")
for alpha in [0.0, 0.05, 0.15, 0.30]:
    gg=torch.Generator(device=dev).manual_seed(1); P=torch.randn(80,128,generator=gg,device=dev); P=P/P.norm(dim=1,keepdim=True)
    W2=(W2t+alpha*nt[:,None]*P).clone().requires_grad_(True)
    opt=torch.optim.Adam([W2],lr=3e-3)
    for step in range(3000):
        idx=torch.randint(0,Hc.shape[0],(512,),generator=g,device=dev)
        loss=((G(Hc[idx],W2)-Yt[idx])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        fit=float((G(Hc,W2)-Yt).pow(2).mean().sqrt()/yn); e0=alpha; e1=relerr(W2)
    v="CONVERGED" if e1<0.02 else ("improved" if e1<0.7*max(e0,1e-9) else "STUCK")
    print(f"{alpha:>6.2f} {e0:>14.3f} {e1:>13.3f} {fit:>15.2e} {v:>10}", flush=True)
