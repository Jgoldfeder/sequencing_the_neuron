"""ITEM C make-or-break: numerical/algebraic implicitization of the architecture jet
manifold. Sample many random 6->8->4->1 downstreams (offline, teacher NOT included);
compute germ-Hessian off-diagonal jets J across FIXED probes. Search for degree-2
relations c^T psi(J)=0 holding across the manifold (null space of the feature Gram).
Then held-out teacher: L(b)=sum_r (c_r^T psi(J(b)))^2, J(b)=dewarped at candidate b.
Does L bottom out at the true bias (delta-parabola)?"""
import numpy as np
sig=lambda x:1/(1+np.exp(-x))
k,W,m=6,8,4; P=6
rng0=np.random.default_rng(0); cs=np.stack([(2*rng0.random(k)-1)*1.3 for _ in range(P)])  # FIXED probes
iu=np.triu_indices(k,1)
def germ_off_batch(W2,b2,W3,b3,w4,beta):     # all (M,...) -> (M, P*15) off-diag germ Hessians
    M=W2.shape[0]; outs=[]
    for c in cs:
        s=sig(c[None,:]+beta); p2=np.einsum('mwk,mk->mw',W2,s)+b2
        sp=sig(p2); spd=sp*(1-sp); spdd=spd*(1-2*sp); q=sp
        u=np.einsum('mnw,mw->mn',W3,q)+b3; hh=sig(u); hd=hh*(1-hh); hdd=hd*(1-2*hh)
        rho=np.einsum('mnw,mn->mw',W3,w4*hd); K=np.einsum('mnw,mn,mnv->mwv',W3,w4*hdd,W3)
        S=np.zeros((M,W,W)); di=np.arange(W); S[:,di,di]=rho*spdd
        S=S+spd[:,:,None]*K*spd[:,None,:]
        G=np.einsum('mwk,mwv,mvj->mkj',W2,S,W2)
        outs.append(G[:,iu[0],iu[1]])
    return np.concatenate(outs,axis=1)       # (M, P*15)
def sample(M,seed):
    r=np.random.default_rng(seed)
    return (r.standard_normal((M,W,k))*0.9,(2*r.random((M,W))-1)*0.6,r.standard_normal((M,m,W))*0.7,
            (2*r.random((M,m))-1)*0.5,r.standard_normal((M,m))*0.8,(2*r.random((M,k))-1)*0.5)
M=14000; W2,b2,W3,b3,w4,beta=sample(M,1); J=germ_off_batch(W2,b2,W3,b3,w4,beta)  # (M,90)
d=J.shape[1]
mu=J.mean(0); sc=J.std(0); Jn=(J-mu)/sc                                          # standardize
def feat(Jn):                                                                    # [1, J, upper(J⊗J)]
    n=Jn.shape[0]; quad=(Jn[:,:,None]*Jn[:,None,:]).reshape(n,-1)
    ti=np.triu_indices(d); quad=quad.reshape(n,d,d)[:,ti[0],ti[1]]
    return np.concatenate([np.ones((n,1)),Jn,quad],axis=1)
Phi=feat(Jn); Gram=Phi.T@Phi/M
ev,V=np.linalg.eigh(Gram)
print(f'feature dim={Phi.shape[1]}, samples={M}')
print(f'smallest 8 eigenvalues of degree-2 feature Gram: {np.array2string(ev[:8],precision=2)}')
print(f'largest: {ev[-1]:.2e};  ratio ev[0]/ev[-1]={ev[0]/ev[-1]:.2e}')
nrel=int((ev<ev[-1]*1e-6).sum())
print(f'# near-exact degree-2 relations (ev<1e-6*max): {nrel}')
if nrel>0:
    Crel=V[:,:min(nrel,40)]                                                       # relation covectors
    # held-out teacher
    W2t,b2t,W3t,b3t,w4t,bt=sample(1,9999); bt=bt[0]
    def teacher_jet_at(bcand):
        # measured F off-diag at true bt; reconstruct germ at candidate bcand
        Jtrue=germ_off_batch(W2t,b2t,W3t,b3t,w4t,bt[None,:])[0]                    # (90,) = germ at bt
        out=[]
        for pi,c in enumerate(cs):
            st=sig(c+bt); dt=st*(1-st); sc_=sig(c+bcand); dc=sc_*(1-sc_)
            blk=Jtrue[pi*15:(pi+1)*15]
            warp=np.array([dt[iu[0][q]]*dt[iu[1][q]]/(dc[iu[0][q]]*dc[iu[1][q]]) for q in range(15)])
            out.append(blk*warp)
        return np.concatenate(out)
    def L(bcand):
        Jn2=(teacher_jet_at(bcand)-mu)/sc; f=feat(Jn2[None,:])[0]; return float(((Crel.T@f)**2).sum())
    print('held-out teacher L(b) around true bias:')
    for dlt in (0.0,1e-4,1e-3,1e-2,5e-2):
        b=bt.copy(); b[0]+=dlt; print(f'   delta={dlt:+.0e}: L={L(b):.3e}')
