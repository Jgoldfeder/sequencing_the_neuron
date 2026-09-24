"""Chat's upgrade: (1) JOINT common-denominator rational fit -- all contexts x all
output coords share ONE denominator Q (the first-layer singularity); variable-
projection eliminates the per-(context,output) numerators, leaving a linear LS in
Q's coefficients. (2) Rank contexts by downstream VISIBILITY ||J_f(x_m) n_j|| and
keep only the strongest. Pole = stable root of Q across denom degrees."""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch, numpy as np
from nets import MLP
dev="cuda"; torch.manual_seed(0)
dims=[128,24,32,16,8]; d=dims[0]; k=dims[1]; O=dims[-1]
teacher=MLP(dims,act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]: l.bias.uniform_(-0.5,0.5)
W1t=teacher.layers[0].weight.detach(); b1t=teacher.layers[0].bias.detach()
tn=W1t/W1t.norm(dim=1,keepdim=True); g=torch.Generator(device=dev).manual_seed(1)
@torch.no_grad()
def J_at(x,fd=5e-5):
    E=torch.eye(d,device=dev,dtype=torch.float64)
    return ((teacher(x.unsqueeze(0)+fd*E)-teacher(x.unsqueeze(0)-fd*E))/(2*fd)).t()
Wg=torch.empty(k,d,device=dev,dtype=torch.float64); bg0=torch.empty(k,device=dev,dtype=torch.float64)
for j in range(k):
    u=tn[j]; v=torch.randn(d,generator=g,device=dev,dtype=torch.float64); v=v-(v@u)*u; v=v/v.norm()
    Wg[j]=(math.cos(math.radians(5))*u+math.sin(math.radians(5))*v)*W1t[j].norm()*(1+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64)-1))
    bg0[j]=b1t[j]+0.08*(2*torch.rand(1,generator=g,device=dev,dtype=torch.float64).item()-1)
Wg_pinv=Wg.t()@torch.linalg.inv(Wg@Wg.t())
N=torch.empty_like(Wg)
for j in range(k):
    t=torch.full((k,),20.0,device=dev,dtype=torch.float64); t[j]=0.0
    x0=Wg_pinv@(t-bg0); U,Sv,Vh=torch.linalg.svd(J_at(x0),full_matrices=False)
    nv=Vh[0]; N[j]=nv if float(nv@Wg[j])>0 else -nv

def joint_denominator_roots(tnp, G, R, P, L):
    """variable-projection joint fit: g_mr * Q = P_mr, shared Q (deg R, Q(0)=1).
    G: (K,N). Returns roots of Q (in t)."""
    s=tnp/L; V=np.vander(s,P+1,increasing=True)         # numerator basis (N,P+1)
    Qo,_=np.linalg.qr(V)                                 # orthonormal cols
    def proj(M): return M-Qo@(Qo.T@M)
    Wv=np.vander(s,R+1,increasing=True)[:,1:]            # (N,R) degrees 1..R
    A=[]; b=[]
    for gm in G:
        A.append(proj(gm[:,None]*Wv)); b.append(proj(gm))
    A=np.vstack(A); b=np.concatenate(b)
    q,_,_,_=np.linalg.lstsq(A,-b,rcond=None)
    coef=np.concatenate([[1.0],q])                       # Q(s)=1+sum q_r s^r
    return np.roots(coef[::-1])*L

S=20.0; M=24; L=3.0; NP=160; delta=1e-3
tanp=np.cos(np.pi*(np.arange(NP)+0.5)/NP)*L
ta=torch.tensor(tanp,device=dev,dtype=torch.float64)
a_cur=Wg.norm(dim=1).clone(); b_cur=bg0.clone()
for it in range(2):
    W1cur=N*a_cur[:,None]; W1cur_pinv=W1cur.t()@torch.linalg.inv(W1cur@W1cur.t())
    a_new=a_cur.clone(); b_new=b_cur.clone()
    for j in range(k):
        im_exp=math.pi/float(a_cur[j]); nx=-float(b_cur[j])/float(a_cur[j])
        # --- (2) rank ~300 contexts by visibility ||d f/d n_j|| ---
        SG=(torch.randint(0,2,(300,k),generator=g,device=dev)*2-1).double()*S; SG[:,j]=0.0
        Xm=(SG-b_cur)@W1cur_pinv.t()                     # (300,d)
        with torch.no_grad():
            fp=teacher(Xm+delta*N[j]); fm=teacher(Xm-delta*N[j])
        vis=((fp-fm)/(2*delta)).norm(dim=1)
        top=torch.topk(vis,M).indices
        # --- sample all output coords along the top contexts ---
        G=[]
        for mi in top.tolist():
            pts=Xm[mi].unsqueeze(0)+ta.unsqueeze(1)*N[j].unsqueeze(0)
            with torch.no_grad(): F=teacher(pts)         # (NP,O)
            for r in range(O): G.append(F[:,r].cpu().numpy())
        # --- (1) joint fit across degrees; stable common pole cloud ---
        cand=[]
        for R in (10,14,18):
            try: roots=joint_denominator_roots(tanp,G,R,min(R,12),L)
            except Exception: continue
            sel=roots[(roots.imag>0.5*im_exp)&(roots.imag<1.8*im_exp)&(np.abs(roots.real)<0.6)]
            cand.extend(sel.tolist())
        if not cand: continue
        cand=np.array(cand)
        # center: pole with most neighbors, then median
        cnt=np.array([np.sum(np.abs(cand-p)<0.05*im_exp) for p in cand]); best=cand[cnt.argmax()]
        cl=cand[np.abs(cand-best)<0.05*im_exp]
        c=complex(np.median(cl.real),np.median(cl.imag))
        a_new[j]=math.pi/c.imag; b_new[j]=-(math.pi/c.imag)*(c.real+nx)
    me=(a_new-W1t.norm(dim=1)).abs(); be=(b_new-b1t).abs()
    print(f"iter {it}: mag worst {me.max():.3e} med {me.median():.3e} | bias worst {be.max():.3e} med {be.median():.3e}")
    a_cur=a_new; b_cur=b_new
