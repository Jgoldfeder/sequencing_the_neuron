"""CNN recovery audit; teacher weights only construct fixtures and score results.
Run from code/: python peel/test_cnn_audit.py --suite small|trained --output PATH
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
import torch
sys.path.insert(0, '.')
from nets import ConvNet
from kink_solve import recover_layer


def unit_rows(weight, bias):
    v = torch.cat((weight.flatten(1), bias[:, None]), 1)
    return v / v.norm(dim=1, keepdim=True)


def run(args):
    torch.set_num_threads(1)
    results = []
    for act in ('leaky_relu', 'relu'):
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            if args.suite == 'trained':
                cfg = [(1,6,5,1,2,2),(6,16,5,1,0,2),(16,120,5,1,0,0)]
                T = ConvNet((1,28,28),cfg,(84,),10,act).double().to(args.device)
                p = f'teachers/teacher_cnn_1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0_fc84__o10_e25_s0_{act}.pt'
                T.load_state_dict(torch.load(p, map_location=args.device, weights_only=True))
            else:
                # Non-square, multichannel input; padding, stride, pooling, FC.
                cfg = [(2,3,3,1,1,2),(3,4,3,2,1,0)]
                T = ConvNet((2,12,14),cfg,(5,),3,act).double().to(args.device)
                with torch.no_grad():
                    for layer in T.layers:
                        layer.bias.normal_(0,.05)
            T.eval()
            recovered = copy.deepcopy(T)
            for fr in range(len(T.layers)-1):
                G = copy.deepcopy(recovered if args.recursive else T)
                gen = torch.Generator(device=args.device).manual_seed(1000+seed*10+fr)
                with torch.no_grad():
                    w,b = G.layers[fr].weight,G.layers[fr].bias
                    norm = w.flatten(1).norm(dim=1)
                    w.add_(args.noise*norm.reshape(-1,*([1]*(w.ndim-1)))*torch.randn(w.shape,device=args.device,dtype=w.dtype,generator=gen)/math.sqrt(w[0].numel()))
                    b.add_(args.noise*.5*norm*torch.randn(b.shape,device=args.device,dtype=b.dtype,generator=gen))
                    for layer in G.layers[fr+1:]:
                        layer.weight.zero_(); layer.bias.zero_()
                channels = list(range(len(w))) if args.suite=='small' else sorted(set([0,len(w)//2,len(w)-1]))
                before = {k:v.clone() for k,v in G.state_dict().items()}
                start=time.time()
                W,B,mask,nq=recover_layer(T,G,fr,args.device,only_channels=channels,gen=gen,sampling='track',verbose=True)
                target=unit_rows(T.layers[fr].weight,T.layers[fr].bias)
                errors=(unit_rows(W,B)-target).abs().amax(1)
                accepted=mask.nonzero().flatten()
                record=dict(act=act,seed=seed,frontier=fr,recursive=args.recursive,noise=args.noise,
                            requested=len(channels),accepted=len(accepted),queries=nq,seconds=time.time()-start,
                            max_error=float(errors[accepted].max()) if len(accepted) else None,
                            errors={str(c):float(errors[c]) for c in channels},mask=mask.tolist())
                assert all(torch.equal(before[k],v) for k,v in G.state_dict().items()), 'Mutated guess'
                assert torch.equal(W[~mask],G.layers[fr].weight[~mask]), 'Changed unsolved weights'
                assert torch.equal(B[~mask],G.layers[fr].bias[~mask]), 'Changed unsolved biases'
                results.append(record)
                Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
                print('AUDIT',json.dumps(record),flush=True)
                if args.recursive:
                    if not bool(mask.all()):
                        break
                    with torch.no_grad():
                        # Preserve teacher gauge for scoring subsequent rows;
                        # magnitude is unconstrained by a kink hyperplane.
                        scale=torch.cat((T.layers[fr].weight.flatten(1),T.layers[fr].bias[:,None]),1).norm(dim=1)
                        recovered.layers[fr].weight.copy_(W*scale.reshape(-1,*([1]*(W.ndim-1))))
                        recovered.layers[fr].bias.copy_(B*scale)
    bad=[r for r in results if r['max_error'] is not None and r['max_error']>1e-7]
    assert not bad, f'Incorrect accepted rows in {len(bad)} cases; see {args.output}'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['small','trained'],default='small')
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--noise',type=float,default=.01)
    p.add_argument('--recursive',action='store_true')
    p.add_argument('--output',default='peel/cnn_audit.json')
    run(p.parse_args())
