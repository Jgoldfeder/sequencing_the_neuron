"""Check joint recovery and adjoint-driven LSMR against exact synthetic data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from nets import MLP
from peel.jump_kinks import joint_fit, prefix_masks, measure
from verify_layer1 import _Oracle

def main():
    rng=np.random.default_rng(3); d=12
    weights=[rng.normal(size=(d,d))/np.sqrt(d) for _ in range(2)]
    truth=rng.normal(size=d+1); truth/=np.linalg.norm(truth[:-1])
    guess=truth[:-1]+rng.normal(size=d)*.01; regions=[]
    for _ in range(8):
        masks=[rng.choice([.01,1.],size=d) for _ in weights]
        n=truth[:-1].copy()
        for w,m in reversed(list(zip(weights,masks))): n=w.T@(m*n)
        n/=np.linalg.norm(n)
        h=rng.normal(size=d); h-=truth[:-1]*(h@truth[:-1]+truth[-1])
        regions.append((masks,n,h))
    for dense in [True,False]:
        result,info=joint_fit(weights,regions,guess,dense=dense)
        error=np.max(abs(result-truth)); assert error<1e-10,(error,info)
    net=MLP([d,d,2]).double()
    masks=prefix_masks(net,torch.zeros(d,dtype=torch.float64),1)
    assert masks[0].dtype==np.float64
    n=torch.tensor(truth[:-1]); u=n.clone(); x=-truth[-1]*n
    output=torch.tensor([1.,-2.,.7])
    oracle=_Oracle(lambda z:torch.nn.functional.leaky_relu(z@n+truth[-1],.01)[:,None]*output)
    normal,diag=measure(oracle,x,u,torch.tensor(.1),49)
    assert normal is not None and abs(np.dot(normal,truth[:-1]))>1-1e-12
    assert oracle.n==8*d*diag['attempts']
    print('PASS: exact joint system, iterative fit, fp64 prefix masks, forward-only jump and query count')

if __name__=='__main__': main()
