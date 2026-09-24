"""Scaling law for the MIXED-derivative bias recovery.
First layer = 2 neurons reading 2 inputs (t_i controls z_i=a_i t_i+b_i).
Downstream = MLP of given hidden sizes. Question: what mixed-jet ORDER makes the
biases b1,b2 IDENTIFIABLE (downstream as nuisance)? Test vs second-layer WIDTH and
DEPTH. Method: jet-vector J(theta,t) of all mixed partials up to order n; Jacobian
dJ/dtheta; null space = downstream-compensating perturbations; b_i identifiable iff
its row in the null basis is ~0. Report min order per architecture."""
import torch, itertools, numpy as np
torch.set_default_dtype(torch.float64)

def make_net(hidden, seed):
    g=torch.Generator().manual_seed(seed)
    a1,a2=1.4,1.1; b1,b2=0.35,-0.2
    sizes=[2]+hidden+[1]; Ws=[]; bs=[]
    for i in range(len(sizes)-1):
        Ws.append((2*torch.rand(sizes[i+1],sizes[i],generator=g)-1)*1.3)
        bs.append((2*torch.rand(sizes[i+1],generator=g)-1)*0.6)
    # flat theta = [b1,b2, all Ws, all bs]
    theta=torch.cat([torch.tensor([b1,b2])]+[W.reshape(-1) for W in Ws]+[b for b in bs])
    shapes=[('b',2)]+[('W',W.shape) for W in Ws]+[('bb',b.shape) for b in bs]
    return theta, shapes, (a1,a2)

def forward(theta, shapes, a, t1, t2):
    a1,a2=a; idx=0
    b1=theta[0]; b2=theta[1]; idx=2
    s1=torch.sigmoid(a1*t1+b1); s2=torch.sigmoid(a2*t2+b2)
    h=torch.stack([s1,s2]).reshape(2,1)
    Ws=[]; bs=[]
    for typ,sh in shapes[1:]:
        if typ=='W':
            n=sh[0]*sh[1]; Ws.append(theta[idx:idx+n].reshape(sh)); idx+=n
    for typ,sh in shapes[1:]:
        if typ=='bb':
            n=sh[0]; bs.append(theta[idx:idx+n]); idx+=n
    for i,(W,bb) in enumerate(zip(Ws,bs)):
        h=W@h+bb.reshape(-1,1)
        if i<len(Ws)-1: h=torch.sigmoid(h)
    return h.squeeze()

def jet_vec(theta, shapes, a, t1v, t2v, n):
    t1=torch.tensor(t1v,requires_grad=True); t2=torch.tensor(t2v,requires_grad=True)
    f=forward(theta,shapes,a,t1,t2)
    # build all mixed partials up to order n by BFS
    cur={(0,0):f}; comps=[]
    for order in range(1,n+1):
        nxt={}
        for (i,j),val in cur.items():
            if i+j==order-1:
                # differentiate wrt t1 and t2
                d1=torch.autograd.grad(val,t1,create_graph=True,retain_graph=True)[0]
                d2=torch.autograd.grad(val,t2,create_graph=True,retain_graph=True)[0]
                nxt[(i+1,j)]=d1; nxt[(i,j+1)]=d2
        cur.update(nxt)
    for order in range(1,n+1):
        for i in range(order,-1,-1):
            comps.append(cur[(i,order-i)])
    return torch.stack(comps)

def min_order(hidden, seed=0):
    theta,shapes,a=make_net(hidden,seed)
    for n in (2,3,4,5):
        theta_l=theta.clone().requires_grad_(True)
        J=torch.autograd.functional.jacobian(
            lambda th: jet_vec(th,shapes,a,0.2,-0.3,n), theta_l, vectorize=True)
        J=J.detach().numpy()                       # (num_jet_comps, num_params)
        U,S,Vt=np.linalg.svd(J)
        tol=S.max()*1e-9
        null=Vt[S.shape[0]:] if J.shape[0]<J.shape[1] else np.zeros((0,J.shape[1]))
        # include right-sing vectors with tiny sing val too
        rankdef=[Vt[i] for i in range(len(S)) if S[i]<tol]
        nb=np.array(list(null)+rankdef)
        if nb.shape[0]==0:
            return n
        b1row=np.abs(nb[:,0]).max(); b2row=np.abs(nb[:,1]).max()
        if b1row<1e-6 and b2row<1e-6:
            return n
    return '>5'

print("MIXED-jet minimum order for bias identifiability (downstream = nuisance):")
for hidden,label in [([1],'2->1->1 (bottleneck)'),
                     ([2],'2->2->1 (width 2)'),
                     ([3],'2->3->1 (width 3)'),
                     ([4],'2->4->1 (width 4)'),
                     ([2,2],'2->2->2->1 (deeper, width 2)'),
                     ([2,3,2],'2->2->3->2->1 (deep, width 2)')]:
    mo=min_order(hidden)
    print(f"   {label:32s}: min mixed order = {mo}")
