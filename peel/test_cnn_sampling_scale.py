"""CNN search brackets and recovery must not depend on target-row magnitude."""
import copy
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import kink_solve as K
from nets import ConvNet


def main():
    torch.set_num_threads(1);torch.manual_seed(9)
    T=ConvNet((1,8,8),[(1,2,3,1,1,2),(2,3,3,1,1,0)],(4,),3,'leaky_relu').double()
    with torch.no_grad():
        for layer in T.layers:layer.bias.normal_(0,.05)
    for fr in range(3):
        G=copy.deepcopy(T)
        with torch.no_grad():
            G.layers[fr].weight.add_(.001*torch.randn_like(G.layers[fr].weight))
            G.layers[fr].bias.add_(.001*torch.randn_like(G.layers[fr].bias))
        expected=torch.cat((T.layers[fr].weight.flatten(1),T.layers[fr].bias[:,None]),1)
        expected=expected/expected.norm(dim=1,keepdim=True)
        base_model=K.ConvNetUnits(G,fr)
        units=base_model.units_of(0,32,torch.Generator().manual_seed(1))
        base_brackets=K.seed_brackets(base_model,fr,units,32,torch.Generator().manual_seed(2),isolate_prefix=False)
        for scale in (.02,1.,50.):
            model=copy.deepcopy(G)
            with torch.no_grad():
                model.layers[fr].weight.mul_(scale);model.layers[fr].bias.mul_(scale)
            saved={k:v.clone() for k,v in model.state_dict().items()}
            brackets=K.seed_brackets(K.ConvNetUnits(model,fr),fr,units,32,torch.Generator().manual_seed(2),isolate_prefix=False)
            for got,want in zip(brackets,base_brackets):
                assert got.shape==want.shape
                assert torch.allclose(got,want,atol=1e-11,rtol=1e-11),(fr,scale)
            W,b,mask,nq=K.recover_layer(T,model,fr,'cpu',sampling='track',gen=torch.Generator().manual_seed(3),verbose=False)
            got=torch.cat((W.flatten(1),b[:,None]),1);got=got/got.norm(dim=1,keepdim=True)
            error=(got-expected).abs().max().item()
            assert bool(mask.all()),(fr,scale,mask)
            assert error<1e-9,(fr,scale,error)
            assert all(torch.equal(v,saved[k]) for k,v in model.state_dict().items())
            print(f'PASS layer={fr} scale={scale}: {int(mask.sum())}/{len(mask)} error={error:.2e} queries={nq}',flush=True)
    # An unrequested stale row can be closer than the fresh requested guess.
    G=copy.deepcopy(T)
    with torch.no_grad():
        G.layers[0].weight[1].copy_(T.layers[0].weight[0])
        G.layers[0].bias[1].copy_(T.layers[0].bias[0])
        G.layers[0].weight[0].add_(.001*torch.randn_like(G.layers[0].weight[0]))
    W,b,mask,_=K.recover_layer(T,G,0,'cpu',only_channels=[0],sampling='track',
                              gen=torch.Generator().manual_seed(3),verbose=False)
    got=torch.cat((W[0].flatten(),b[0:1]));got=got/got.norm()
    want=torch.cat((T.layers[0].weight[0].flatten(),T.layers[0].bias[0:1]));want=want/want.norm()
    assert bool(mask[0]) and (got-want).abs().max().item()<1e-9
    print('PASS: stale unrequested guess does not veto a correct requested fit')
    print('PASS: CNN seeding and full recovery across 2500x target-row scale range')


if __name__=='__main__':main()
