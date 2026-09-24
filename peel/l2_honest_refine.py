"""HONEST L2 solve attempt: refine the layer-2 weight against a BLACK-BOX oracle only.

No cheating:
 - TARGET = the real subnetwork's value+Jacobian at chosen h (a black-box query; the true
   net evaluated at h). We never read teacher weights into the model.
 - MODEL  = OUR student (seeded from ONE committee member: its own layers 2..4 are the
   downstream GUESS, its layer-2 the weight guess). b2 + downstream are the committee's, NOT true.
 - Correspondence: none used in the solve; the student lives in its own (member-0) neuron order.
   True W2 is unlocked ONLY at the end for scoring (perm/sign alignment).

Two conditions:
  (A) refine W2 only, downstream+b2 FROZEN at the guess  -> does W2 converge with a wrong downstream?
  (B) refine the whole subnetwork (W2,b2,downstream) jointly -> does first-layer identifiability pin W2?
"""
import sys, torch
import torch.nn.functional as F
from torch.func import jvp, vjp
torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from scipy.optimize import linear_sum_assignment
dev = "cuda" if torch.cuda.is_available() else "cpu"
P = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
pk = torch.load(P + "peel_committee.pt", map_location=dev, weights_only=False)
dims = [784,128,80,40,32,10]

# ---- true teacher: ORACLE (value+Jac targets) + SCORING only. Model never reads its weights. ----
teacher = MLP(dims, act="sigmoid").to(dev); teacher.load_state_dict(pk["teacher_state"]); teacher.eval()
for p in teacher.parameters(): p.requires_grad_(False)
Wt = [teacher.layers[i].weight.detach() for i in range(5)]; bt = [teacher.layers[i].bias.detach() for i in range(5)]
W2t = Wt[1]; nt = W2t.norm(dim=1)
def oracle_fwd(h):                                   # black-box: true subnetwork layers 2..4 (h -> out)
    q2 = torch.sigmoid(h @ Wt[1].t() + bt[1]); q3 = torch.sigmoid(q2 @ Wt[2].t() + bt[2])
    q4 = torch.sigmoid(q3 @ Wt[3].t() + bt[3]); return q4 @ Wt[4].t() + bt[4]

# ---- honest seed = committee MEMBER 0 (a single consistent guess of layers 2..4) ----
sd0 = pk["pop_states"][0]
g_W2 = sd0["layers.1.weight"].to(dev).double().clone(); g_b2 = sd0["layers.1.bias"].to(dev).double().clone()
g_W3 = sd0["layers.2.weight"].to(dev).double().clone(); g_b3 = sd0["layers.2.bias"].to(dev).double().clone()
g_W4 = sd0["layers.3.weight"].to(dev).double().clone(); g_b4 = sd0["layers.3.bias"].to(dev).double().clone()
g_W5 = sd0["layers.4.weight"].to(dev).double().clone(); g_b5 = sd0["layers.4.bias"].to(dev).double().clone()
FIX = (g_b2, g_W3, g_b3, g_W4, g_b4, g_W5, g_b5)      # frozen guess for condition A

def model_A(h, W2):                                   # only W2 free; rest = frozen guess
    b2, W3, b3, W4, b4, W5, b5 = FIX
    q2 = torch.sigmoid(h @ W2.t() + b2); q3 = torch.sigmoid(q2 @ W3.t() + b3)
    q4 = torch.sigmoid(q3 @ W4.t() + b4); return q4 @ W5.t() + b5
def model_B(h, ps):                                   # all of layers 2..4 free
    W2,b2,W3,b3,W4,b4,W5,b5 = ps
    q2 = torch.sigmoid(h @ W2.t() + b2); q3 = torch.sigmoid(q2 @ W3.t() + b3)
    q4 = torch.sigmoid(q3 @ W4.t() + b4); return q4 @ W5.t() + b5

def wscore(W2, tag):                                  # SCORING ONLY (truth here, nowhere else)
    Cp = torch.cdist(W2, W2t); Cm = torch.cdist(-W2, W2t); C = torch.minimum(Cp, Cm).cpu().numpy()
    ri, ci = linear_sum_assignment(C); e = torch.tensor([C[ri[t], ci[t]] for t in range(len(ri))])
    rel = e / nt[ci].cpu()
    print(f"    {tag:22s} W rel-row-err: mean {float(rel.mean()):.3e}  max {float(rel.max()):.3e}  <1%:{int((rel<0.01).sum())}/80")

# ---- full-cube h + full-Jacobian targets from the ORACLE ----
N = 400
gz = torch.Generator(device="cpu").manual_seed(0); scales = torch.tensor([1.5,4.0,10.0])
sidx = torch.randint(0,3,(N,),generator=gz); z = torch.randn(N,128,generator=gz)*scales[sidx][:,None]
h = torch.sigmoid(z).clamp(1e-5,1-1e-5).to(dev)
Qg = torch.Generator(device="cpu").manual_seed(3); Q,_ = torch.linalg.qr(torch.randn(128,128,generator=Qg)); Q = Q[:64].to(dev)
valt = oracle_fwd(h).detach(); scv = valt.abs().max()
def jac_oracle():
    f = lambda hb: oracle_fwd(hb)
    return torch.func.vmap(lambda u: jvp(f,(h,),(u.expand(N,128),))[1])(Q)
ddt = jac_oracle().detach(); scd = ddt.abs().max()

def cg(A,b,it=50,tol=1e-14):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rs=r@r
    for _ in range(it):
        Ap=A(p);a=rs/(p@Ap+1e-30);x=x+a*p;r=r-a*Ap;rs2=r@r
        if rs2.sqrt()<tol:break
        p=r+(rs2/rs)*p;rs=rs2
    return x
def refine(w0, resid, iters=60):
    wf=w0.clone();lam=1e-3;r=resid(wf);c=float(r@r)
    for it in range(iters):
        _,vjpf=vjp(resid,wf);Jt=lambda u:vjpf(u)[0];Jv=lambda v:jvp(resid,(wf,),(v,))[1]
        gvec=Jt(r);A=lambda v:Jt(Jv(v))+lam*v;ok=False
        for _ in range(12):
            dwf=cg(A,-gvec);wn=wf+dwf;rn=resid(wn);cn=float(rn@rn)
            if cn<c:wf=wn;r=rn;c=cn;lam=max(lam*0.3,1e-15);ok=True;break
            lam*=5
        if not ok or c<1e-28:break
    return wf,c

# ================= condition A: W2 only, frozen guessed downstream =================
print("condition A: refine W2 ONLY (b2+downstream frozen at committee guess), match black-box oracle")
wscore(g_W2, "init (member-0 guess)")
def resid_A(wf):
    W2 = wf.reshape(80,128)
    fh = lambda hb: model_A(hb, W2)
    val = model_A(h, W2)
    dd = torch.func.vmap(lambda u: jvp(fh,(h,),(u.expand(N,128),))[1])(Q)
    return torch.cat([((val-valt)/scv).reshape(-1), ((dd-ddt)/scd).reshape(-1)])
wfA,cA = refine(g_W2.reshape(-1).clone(), resid_A, iters=60)
wscore(wfA.reshape(80,128), "after A"); print(f"    final loss {cA:.3e}\n")

# ================= condition B: joint refine of whole subnetwork =================
print("condition B: refine W2 + b2 + downstream JOINTLY, match black-box oracle")
seed = [g_W2,g_b2,g_W3,g_b3,g_W4,g_b4,g_W5,g_b5]
shapes = [t.shape for t in seed]; numels = [t.numel() for t in seed]
def unpack(v):
    out=[]; o=0
    for sh,n in zip(shapes,numels): out.append(v[o:o+n].reshape(sh)); o+=n
    return out
def resid_B(v):
    ps = unpack(v)
    fh = lambda hb: model_B(hb, ps)
    val = model_B(h, ps)
    dd = torch.func.vmap(lambda u: jvp(fh,(h,),(u.expand(N,128),))[1])(Q)
    return torch.cat([((val-valt)/scv).reshape(-1), ((dd-ddt)/scd).reshape(-1)])
v0 = torch.cat([t.reshape(-1) for t in seed]).clone()
wfB,cB = refine(v0, resid_B, iters=60)
wscore(unpack(wfB)[0], "after B"); print(f"    final loss {cB:.3e}")
