"""Basin map: how accurate must the row guess be for deflation to REFINE (not degrade) the dual dirs?
Inject controlled relative row error eps onto the TRUE W2, run honest deflation off those noisy rows,
compare refined dual-dir angle to the noisy-guess angle. Crossover = viability threshold."""
import sys, time, torch, numpy as np
torch.set_default_dtype(torch.float32)
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
dev="cuda"; P="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
pk=torch.load(P+"peel_committee.pt",map_location=dev,weights_only=False); dims=[784,128,80,40,32,10]
_t=MLP(dims,act="sigmoid").to(dev); _t.load_state_dict(pk["teacher_state"]); _t.eval()
for p in _t.parameters(): p.requires_grad_(False)
Wt=[_t.layers[i].weight.detach().float() for i in range(5)]; bt=[_t.layers[i].bias.detach().float() for i in range(5)]
def oracle(h):
    q2=torch.sigmoid(h@Wt[1].t()+bt[1]);q3=torch.sigmoid(q2@Wt[2].t()+bt[2])
    q4=torch.sigmoid(q3@Wt[3].t()+bt[3]);return q4@Wt[4].t()+bt[4]
W2t=Wt[1]; b2t=bt[1]; nt=W2t.norm(dim=1)
Vt=torch.linalg.pinv(W2t); Vt=Vt/Vt.norm(dim=0,keepdim=True)
def angles(Vrec):
    Cabs=(Vrec.t()@Vt).abs(); r,c=linear_sum_assignment((1-Cabs).cpu().numpy())
    return torch.rad2deg(torch.arccos(Cabs[r,c].clamp(0,1)))
def deflate(W2g,b2g,base_pts=8,fd=1e-3):
    W2gp=torch.linalg.pinv(W2g); Vrec=torch.zeros(128,80,device=dev); gg=torch.Generator(device="cpu").manual_seed(0)
    rows=[torch.cat([W2g[:j],W2g[j+1:]],0) for j in range(80)]
    for j in range(80):
        Qb,_=torch.linalg.qr(rows[j].t()); G=torch.zeros(128,128,device=dev)
        for _ in range(base_pts):
            t=(torch.randn(80,generator=gg)*2.0).to(dev); t[j]=0.0
            h0=torch.clamp(W2gp@(t-b2g),1e-3,1-1e-3); E=torch.eye(128,device=dev)
            Hp=(h0+fd*E).clamp(1e-6,1-1e-6); Hm=(h0-fd*E).clamp(1e-6,1-1e-6)
            with torch.no_grad(): J=((oracle(Hp)-oracle(Hm))/(2*fd)).t()
            JP=J-(J@Qb)@Qb.t(); G+=JP.t()@JP
        w,V=torch.linalg.eigh(G); Vrec[:,j]=V[:,-1]
    return Vrec
print("eps = injected relative row error on true W2.  dual-dir angle (deg), guess vs deflation-refined.")
print(f"{'eps':>6} {'guess ang':>10} {'refine ang':>11} {'improved':>9}")
gN=torch.Generator(device="cpu").manual_seed(7)
for eps in [0.0,0.02,0.04,0.08,0.12,0.16]:
    D=torch.randn(80,128,generator=gN).to(dev); D=D/D.norm(dim=1,keepdim=True)
    W2g=W2t+eps*nt[:,None]*D; b2g=b2t.clone()
    Vg=torch.linalg.pinv(W2g); Vg=Vg/Vg.norm(dim=0,keepdim=True); ag=angles(Vg)
    Vr=deflate(W2g,b2g); ar=angles(Vr)
    print(f"{eps:>6.2f} {float(ag.mean()):>10.2f} {float(ar.mean()):>11.2f} {int((ar<ag).sum()):>7}/80")
