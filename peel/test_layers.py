"""Single-layer tests with an EXACT prefix and a synthetic 1e-2 guess: python test_layers.py 3 4 1"""
import torch, sys, math, time
sys.path.insert(0,'.')
from nets import MLP
from align import scale_normalize_
import kink_solve as K
dev='cuda'; dims=[200]*8+[100]
torch.manual_seed(0)
T=MLP(dims).double()
with torch.no_grad():
    for L in T.layers: L.bias.normal_(0,0.1)
scale_normalize_(T); T=T.to(dev).eval()
for l in [int(a) for a in sys.argv[1:]] or [3]:
    G=T.clone().eval(); gen=torch.Generator(device=dev).manual_seed(1)
    with torch.no_grad():
        Wt=T.layers[l].weight; bt=T.layers[l].bias
        G.layers[l].weight.copy_(Wt+1e-2*torch.randn(Wt.shape,device=dev,dtype=Wt.dtype,generator=gen)/math.sqrt(200))
        G.layers[l].bias.copy_(bt+5e-3*torch.randn(bt.shape,device=dev,dtype=bt.dtype,generator=gen))
    t0=time.time()
    W,b,mask,nq=K.recover_layer(T,G,l,dev,verbose=2)
    e=[]
    for c in range(200):
        v=torch.cat([W[c],b[c].reshape(1)]); v=v/v.norm(); g=torch.cat([Wt[c],bt[c].reshape(1)]); g=g/g.norm(); e.append((v-g).abs().max().item())
    e=torch.tensor(e)
    print(f"LAYER {l}: {int(mask.sum())}/200  max err {e.max():.1e} median {e.median():.1e}  {nq} q ({nq/200:.0f}/neuron)  {time.time()-t0:.1f}s", flush=True)
