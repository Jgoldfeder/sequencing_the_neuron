"""CPU regressions for CNN gauge transfer, frozen masks and fp64 delivery."""
import copy
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nets import ConvNet
from align import _push_to_next, cnn_canonicalize_
import method
from kink_solve import _cnn_fit_quality, _cnn_solve_points


def rescale(net, seed):
    g=torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for i,layer in enumerate(net.layers[:-1]):
            f=torch.exp(torch.randn(len(layer.bias),generator=g,dtype=torch.float64))
            layer.weight.mul_(f.reshape(-1,*([1]*(layer.weight.ndim-1))))
            layer.bias.mul_(f)
            _push_to_next(net.layers[i+1],1/f,layer.weight.ndim==4)


def main():
    torch.set_num_threads(1); torch.manual_seed(42)
    cfg=[(2,3,3,1,1,2),(3,4,3,1,1,0)]
    source=ConvNet((2,8,10),cfg,(5,),3,'leaky_relu').double()
    with torch.no_grad():
        for layer in source.layers: layer.bias.normal_(0,.1)
    target=copy.deepcopy(source); rescale(source,2); rescale(target,3)
    x=torch.randn(32,160,dtype=torch.float64)
    expected=target(x).detach()
    assert torch.allclose(source(x),expected,atol=1e-12,rtol=1e-12)
    for l in range(3):
        idx=torch.arange(len(target.layers[l].bias))
        method._cnn_transfer_rows_(target,source,l,idx)
        assert torch.allclose(target(x),expected,atol=1e-12,rtol=1e-12),l
    frozen={l:(layer.weight.detach().clone(),layer.bias.detach().clone(),torch.ones(len(layer.bias),dtype=torch.bool)) for l,layer in enumerate(source.layers[:-1])}
    warm=method._cnn_warm_reinit_population([target],source.input_shape,cfg,(5,),3,'leaky_relu','cpu',frozen)[0]
    assert torch.allclose(warm(x.float()).double(),expected,atol=1e-6,rtol=1e-5),'Warm freeze changed function'
    # Frozen and partial masks survive cold restarts and Adam updates.
    mask=torch.tensor([True,False,True])
    bank={0:(source.layers[0].weight,source.layers[0].bias,mask)}
    cold=method._cnn_reinit_frozen_population(source.input_shape,cfg,(5,),3,'leaky_relu','cpu',1,bank)[0]
    opt=torch.optim.Adam(cold.parameters(),lr=.01)
    before=cold.layers[0].weight.detach().clone()
    for _ in range(3):
        opt.zero_grad(); cold(x.float()).square().mean().backward(); opt.step()
    assert torch.equal(cold.layers[0].weight[mask],before[mask])
    assert not torch.equal(cold.layers[0].weight[~mask],before[~mask])
    # Permuting frozen slots must permute their gradient masks too.
    perm=torch.tensor([1,2,0]); pred=cold(x.float()).detach()
    method._cnn_apply_perms_(cold,[perm],0,opt=opt)
    assert torch.allclose(cold(x.float()),pred,atol=1e-6)
    assert torch.equal(cold._cnn_frozen_masks[0],mask[perm])
    before=cold.layers[0].weight.detach().clone(); mm=mask[perm]
    opt.zero_grad(); cold(x.float()).square().mean().backward(); opt.step()
    assert torch.equal(cold.layers[0].weight[mm],before[mm])
    # Delivery retains fp64 record precision despite fp32 training.
    delivered=method._cnn_solved_view(target.float(),{},source,{l:v[2] for l,v in frozen.items()})
    canonical=source.clone(); cnn_canonicalize_(canonical)
    for l in range(3):
        assert torch.equal(delivered.layers[l].weight,canonical.layers[l].weight)
    # Good hyperplanes pass; underdetermined and contaminated fits abstain.
    H=torch.randn(160,6,dtype=torch.float64); H[:,-1]=.3*H[:,0]-.2
    fit=_cnn_solve_points(H,torch.tensor([-.3,0,0,0,0,1.],dtype=torch.float64))
    assert fit is not None and _cnn_fit_quality(*[fit[2],fit[0],fit[1]])[0]
    deficient=H.clone(); deficient[:,1]=0
    assert not _cnn_fit_quality(deficient,fit[0],fit[1])[0]
    bad=H.clone(); bad[::4,-1]+=.01
    assert not _cnn_fit_quality(bad,fit[0],fit[1])[0]
    print('PASS: gauge transfer, warm/cold pinning, Adam/permutation masks, fp64 delivery, fit rank/validation')


if __name__=='__main__': main()
