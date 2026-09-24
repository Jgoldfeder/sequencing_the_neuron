"""Recursive peel of a random 200x...x100 net: for each hidden layer l, guess = recovered
prefix (from earlier peels) + teacher layer l with 1e-2 relative noise; recover; freeze; next."""
import torch, sys, time, math
sys.path.insert(0,'.')
from nets import MLP
from align import scale_normalize_
from verify_layer1 import _Oracle
import kink_solve
dev='cuda'
dims=[200]*8+[100]
torch.manual_seed(0)
T=MLP(dims).double()
with torch.no_grad():
    for L in T.layers: L.bias.normal_(0,0.1)           # random init has zero biases; make them generic
scale_normalize_(T)                                    # canonical gauge: unit-norm hidden rows
T=T.to(dev).eval()
G=T.clone().eval()                                     # student: filled layer by layer
gen=torch.Generator(device=dev).manual_seed(1)
with torch.no_grad():
    for L in G.layers: L.weight.zero_(); L.bias.zero_()
nh=len(dims)-2; total_q=0; t_all=time.time(); worst=0.0
for l in range(nh):
    with torch.no_grad():                              # guess for layer l: teacher + 1e-2 noise
        Wt=T.layers[l].weight; bt=T.layers[l].bias
        G.layers[l].weight.copy_(Wt+1e-2*torch.randn(Wt.shape,device=dev,dtype=Wt.dtype,generator=gen)/math.sqrt(Wt.shape[1]))
        G.layers[l].bias.copy_(bt+5e-3*torch.randn(bt.shape,device=dev,dtype=bt.dtype,generator=gen))
        for k in range(l+1,len(dims)-1):               # downstream: garbage
            G.layers[k].weight.copy_(0.1*torch.randn_like(G.layers[k].weight)); G.layers[k].bias.copy_(0.1*torch.randn_like(G.layers[k].bias))
    t0=time.time()
    W,b,mask,nq=kink_solve.recover_layer(T,G,l,dev,verbose=1)
    total_q+=nq
    if not bool(mask.all()):
        missing = (~mask).nonzero()[:, 0].tolist()
        raise RuntimeError(f"Layer {l} incomplete: {len(missing)} unsolved neurons {missing}; refusing to freeze an inaccurate prefix")
    with torch.no_grad():                              # freeze in the unit-w gauge (matches the canonical teacher)
        n=W.norm(dim=1,keepdim=True); G.layers[l].weight.copy_(W/n); G.layers[l].bias.copy_(b/n[:,0])
        e=torch.maximum((G.layers[l].weight-Wt).abs().max(1).values,(G.layers[l].bias-bt).abs())
    worst=max(worst,e.max().item())
    print(f"LAYER {l}: refined {int(mask.sum())}/{len(mask)}  max err {e.max():.1e}  median {e.median():.1e}  {nq} q  {time.time()-t0:.1f}s", flush=True)
t0=time.time()                                          # output layer: closed-form least squares
X=torch.randn(4000,dims[0],device=dev,dtype=torch.float64,generator=gen)
orc=_Oracle(T); Y=orc(X)
with torch.no_grad():
    H=X
    for L in G.layers[:-1]: H=G.act(L(H))
    A=torch.cat([H,torch.ones(len(H),1,device=dev,dtype=H.dtype)],1)
    sol=torch.linalg.lstsq(A.cpu(),Y.cpu()).solution.to(dev)
    G.layers[-1].weight.copy_(sol[:-1].T); G.layers[-1].bias.copy_(sol[-1])
    eo=max((G.layers[-1].weight-T.layers[-1].weight).abs().max().item(),(G.layers[-1].bias-T.layers[-1].bias).abs().max().item())
    total_q+=orc.n
    Xt=torch.randn(2000,dims[0],device=dev,dtype=torch.float64,generator=gen)
    fgap=(G(Xt)-T(Xt)).abs().max().item()
print(f"OUTPUT layer: max err {eo:.1e}  {orc.n} q  {time.time()-t0:.1f}s")
print(f"TOTAL: worst hidden max err {worst:.1e}  output {eo:.1e}  function gap {fgap:.1e}  {total_q} queries  {time.time()-t_all:.1f}s")
